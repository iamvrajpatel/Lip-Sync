import asyncio
import gc
import logging
import mimetypes
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response


logger = logging.getLogger("lip_sync_api")

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_ROOT = BASE_DIR / "tmp"
LIVEPORTRAIT_DIR = BASE_DIR / "LivePortrait"
LIVEPORTRAIT_INFERENCE_SCRIPT = LIVEPORTRAIT_DIR / "inference.py"
LIVEPORTRAIT_IDLE_DRIVING_VIDEO = LIVEPORTRAIT_DIR / "assets" / "ideal" / "driving" / "ideal.mp4"
PROCESSING_FPS = 25
JOB_RETENTION_SECONDS = 60 * 60


def _default_inference_runner(
    video_path: str,
    audio_path: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    output_path: str,
) -> None:
    from inference import perform_inference

    perform_inference(video_path, audio_path, seed, num_steps, guidance_scale, output_path)


def _verify_default_inference_runtime() -> None:
    from inference import perform_inference

    if not callable(perform_inference):
        raise RuntimeError("Lip-sync inference runtime is unavailable.")


class LipSyncService:
    def __init__(
        self,
        tmp_root: Path | None = None,
        processing_fps: int = PROCESSING_FPS,
        job_retention_seconds: int = JOB_RETENTION_SECONDS,
        loop_video_from_endframe: bool = True,
        inference_runner=None,
    ) -> None:
        self._tmp_root = tmp_root or TMP_ROOT
        self._processing_fps = processing_fps
        self._job_retention_seconds = job_retention_seconds
        self._loop_video_from_endframe = loop_video_from_endframe
        self._inference_runner = inference_runner or _default_inference_runner
        self._job_store_lock = threading.Lock()
        self._job_store: dict[str, dict[str, object]] = {}
        self._job_tasks: set[asyncio.Task] = set()
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    @property
    def processing_fps(self) -> int:
        return self._processing_fps

    @property
    def tmp_root(self) -> Path:
        return self._tmp_root

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/jobs/generate-from-video", status_code=202)
        async def submit_video_job(
            request: Request,
            video: UploadFile = File(..., description=".mp4 file"),
            audio: UploadFile = File(..., description=".wav/.mp3/.aac/.flac"),
            base_url: str | None = Form(None),
            seed: int = Form(1247),
            num_steps: int = Form(40, ge=1, le=100),
            guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
            video_scale: float = Form(0.8, ge=0.1, le=1.0),
            output_fps: int = Form(30, ge=6, le=60),
        ) -> dict[str, object]:
            route_name = "/generate-from-video"
            mode, normalized_base_url = self.resolve_execution_mode(base_url, request)
            job = self.create_job_record(route_name, mode, normalized_base_url, "output_video.mp4")
            job_id = str(job["job_id"])
            job_dir = Path(str(job["job_dir"]))

            try:
                video.file.seek(0)
                audio.file.seek(0)
                video_path = self.save_upload(job_dir, video)
                audio_path = self.save_upload(job_dir, audio)
            except Exception:
                self.delete_job_record(job_id)
                raise

            self.schedule_job(
                self.process_video_job(
                    job_id,
                    route_name,
                    mode,
                    normalized_base_url,
                    video_path,
                    audio_path,
                    seed,
                    num_steps,
                    guidance_scale,
                    video_scale,
                    output_fps,
                )
            )
            return self.serialize_job_record(self.get_job_record(job_id) or job)

        @router.post("/jobs/generate-from-image", status_code=202)
        async def submit_image_job(
            request: Request,
            image: UploadFile = File(...),
            audio: UploadFile = File(...),
            base_url: str | None = Form(None),
            seed: int = Form(1247),
            num_steps: int | None = Form(None, ge=1, le=100),
            steps: int | None = Form(None, ge=1, le=100),
            guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
            video_scale: float = Form(0.8, ge=0.1, le=1.0),
            output_fps: int = Form(30, ge=6, le=60),
        ) -> dict[str, object]:
            resolved_num_steps = self.resolve_num_steps(num_steps, steps)
            route_name = "/generate-from-image"
            mode, normalized_base_url = self.resolve_execution_mode(base_url, request)
            job = self.create_job_record(route_name, mode, normalized_base_url, "lipsync_output.mp4")
            job_id = str(job["job_id"])
            job_dir = Path(str(job["job_dir"]))

            try:
                image.file.seek(0)
                audio.file.seek(0)
                image_path = self.save_upload(job_dir, image)
                audio_path = self.save_upload(job_dir, audio)
            except Exception:
                self.delete_job_record(job_id)
                raise

            self.schedule_job(
                self.process_image_job(
                    job_id,
                    route_name,
                    mode,
                    normalized_base_url,
                    image_path,
                    audio_path,
                    seed,
                    resolved_num_steps,
                    guidance_scale,
                    video_scale,
                    output_fps,
                )
            )
            return self.serialize_job_record(self.get_job_record(job_id) or job)

        @router.get("/jobs/{job_id}")
        async def get_job_status(job_id: str) -> dict[str, object]:
            self.cleanup_expired_jobs()
            job = self.get_job_record(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return self.serialize_job_record(job)

        @router.get("/jobs/{job_id}/download")
        async def download_job_result(job_id: str) -> FileResponse:
            self.cleanup_expired_jobs()
            job = self.get_job_record(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            if str(job["status"]) != "completed":
                raise HTTPException(status_code=409, detail="Job result is not ready yet")

            result_path = Path(str(job["result_path"]))
            if not result_path.exists():
                raise HTTPException(status_code=404, detail="Generated file is no longer available")

            duration_seconds = float(job["duration_seconds"]) if job["duration_seconds"] is not None else 0.0
            return self.build_job_file_response(
                result_path,
                str(job["result_filename"]),
                duration_seconds,
                str(job.get("media_type") or "video/mp4"),
            )

        @router.post("/generate-from-video")
        async def generate_from_video(
            request: Request,
            background_tasks: BackgroundTasks,
            video: UploadFile = File(..., description=".mp4 file"),
            audio: UploadFile = File(..., description=".wav/.mp3/.aac/.flac"),
            base_url: str | None = Form(None),
            seed: int = Form(1247),
            num_steps: int = Form(40, ge=1, le=100),
            guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
            video_scale: float = Form(0.8, ge=0.1, le=1.0),
            output_fps: int = Form(30, ge=6, le=60),
        ) -> Response:
            route_name = "/generate-from-video"
            mode, normalized_base_url = self.resolve_execution_mode(base_url, request)
            started_at = time.perf_counter()

            try:
                if normalized_base_url:
                    response = await self.proxy_request(
                        route_name,
                        normalized_base_url,
                        {"video": video, "audio": audio},
                        {
                            "seed": str(seed),
                            "num_steps": str(num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        "output_video.mp4",
                    )
                else:
                    response = self.run_local_video_generation(
                        video,
                        audio,
                        seed,
                        num_steps,
                        guidance_scale,
                        output_fps,
                        background_tasks,
                        0.0,
                    )

                duration_seconds = time.perf_counter() - started_at
                response.headers.update(self.get_processing_headers(duration_seconds))
                self.log_request_timing(route_name, mode, normalized_base_url, True, duration_seconds)
                return response
            except Exception as exc:
                duration_seconds = time.perf_counter() - started_at
                self.log_request_timing(route_name, mode, normalized_base_url, False, duration_seconds, str(exc))
                raise

        @router.post("/generate-from-image")
        async def generate_from_image(
            request: Request,
            background_tasks: BackgroundTasks,
            image: UploadFile = File(...),
            audio: UploadFile = File(...),
            base_url: str | None = Form(None),
            seed: int = Form(1247),
            num_steps: int | None = Form(None, ge=1, le=100),
            steps: int | None = Form(None, ge=1, le=100),
            guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
            video_scale: float = Form(0.8, ge=0.1, le=1.0),
            output_fps: int = Form(30, ge=6, le=60),
        ) -> Response:
            resolved_num_steps = self.resolve_num_steps(num_steps, steps)
            route_name = "/generate-from-image"
            mode, normalized_base_url = self.resolve_execution_mode(base_url, request)
            started_at = time.perf_counter()

            try:
                if normalized_base_url:
                    response = await self.proxy_request(
                        route_name,
                        normalized_base_url,
                        {"image": image, "audio": audio},
                        {
                            "seed": str(seed),
                            "num_steps": str(resolved_num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        "lipsync_output.mp4",
                    )
                else:
                    response = self.run_local_image_generation(
                        image,
                        audio,
                        seed,
                        resolved_num_steps,
                        guidance_scale,
                        output_fps,
                        background_tasks,
                        0.0,
                    )

                duration_seconds = time.perf_counter() - started_at
                response.headers.update(self.get_processing_headers(duration_seconds))
                self.log_request_timing(route_name, mode, normalized_base_url, True, duration_seconds)
                return response
            except Exception as exc:
                duration_seconds = time.perf_counter() - started_at
                self.log_request_timing(route_name, mode, normalized_base_url, False, duration_seconds, str(exc))
                raise

        return router

    def get_health_payload(self) -> dict[str, object]:
        with self._job_store_lock:
            jobs = list(self._job_store.values())

        counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
        for job in jobs:
            status = str(job.get("status", "queued"))
            if status in counts:
                counts[status] += 1

        return {
            "tmp_root": str(self._tmp_root),
            "processing_fps": self._processing_fps,
            "job_retention_seconds": self._job_retention_seconds,
            "active_job_count": len(jobs),
            "queued_job_count": counts["queued"],
            "processing_job_count": counts["processing"],
            "completed_job_count": counts["completed"],
            "failed_job_count": counts["failed"],
        }

    def get_processing_headers(self, duration_seconds: float) -> dict[str, str]:
        return {
            "X-Processing-Time": f"{duration_seconds:.3f}",
            "Access-Control-Expose-Headers": "Content-Disposition, X-Processing-Time",
        }

    def normalize_base_url(self, base_url: str | None) -> str | None:
        if not base_url:
            return None
        trimmed = base_url.strip()
        return trimmed.rstrip("/") if trimmed else None

    def resolve_execution_mode(
        self,
        base_url: str | None,
        request: Request | None = None,
    ) -> tuple[str, str | None]:
        normalized_base_url = self.normalize_base_url(base_url)
        if not normalized_base_url:
            return "local", None

        request_base_url = self.normalize_base_url(str(request.base_url)) if request else None
        if request_base_url and normalized_base_url == request_base_url:
            return "local", None

        return "proxy", normalized_base_url

    def log_request_timing(
        self,
        route_name: str,
        mode: str,
        base_url: str | None,
        success: bool,
        duration_seconds: float,
        error: str | None = None,
    ) -> None:
        logger.info(
            "route=%s mode=%s success=%s duration=%.3fs base_url=%s error=%s",
            route_name,
            mode,
            success,
            duration_seconds,
            base_url or "-",
            error or "-",
        )

    def cleanup_job_dir(self, job_dir: Path) -> None:
        shutil.rmtree(job_dir, ignore_errors=True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    def cleanup_expired_jobs(self) -> None:
        cutoff = time.time() - self._job_retention_seconds
        expired_jobs: list[tuple[str, Path]] = []

        with self._job_store_lock:
            for job_id, job in list(self._job_store.items()):
                status = str(job.get("status", "queued"))
                updated_at = float(job.get("updated_at", job.get("created_at", 0.0)))
                if status in {"completed", "failed"} and updated_at < cutoff:
                    expired_jobs.append((job_id, Path(str(job["job_dir"]))))
                    self._job_store.pop(job_id, None)

        for _, job_dir in expired_jobs:
            self.cleanup_job_dir(job_dir)

    def create_job_record(
        self,
        route_name: str,
        mode: str,
        base_url: str | None,
        result_filename: str,
        media_type: str = "video/mp4",
    ) -> dict[str, object]:
        self.cleanup_expired_jobs()

        job_id = uuid.uuid4().hex
        created_at = time.time()
        job_dir = self._tmp_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = {
            "job_id": job_id,
            "route_name": route_name,
            "mode": mode,
            "base_url": base_url,
            "status": "queued",
            "error": None,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "job_dir": str(job_dir),
            "result_path": None,
            "result_filename": result_filename,
            "media_type": media_type,
        }

        with self._job_store_lock:
            self._job_store[job_id] = job

        return dict(job)

    def delete_job_record(self, job_id: str) -> None:
        with self._job_store_lock:
            job = self._job_store.pop(job_id, None)

        if job:
            self.cleanup_job_dir(Path(str(job["job_dir"])))

    def update_job_record(self, job_id: str, **updates: object) -> None:
        with self._job_store_lock:
            job = self._job_store.get(job_id)
            if not job:
                return
            job.update(updates)
            job["updated_at"] = time.time()

    def get_job_record(self, job_id: str) -> dict[str, object] | None:
        with self._job_store_lock:
            job = self._job_store.get(job_id)
            return dict(job) if job else None

    def serialize_job_record(self, job: dict[str, object]) -> dict[str, object]:
        status = str(job["status"])
        duration_seconds = job.get("duration_seconds")
        return {
            "job_id": job["job_id"],
            "route_name": job["route_name"],
            "mode": job["mode"],
            "status": status,
            "error": job.get("error"),
            "duration_seconds": round(float(duration_seconds), 3) if duration_seconds is not None else None,
            "result_filename": job["result_filename"],
            "download_url": f"/jobs/{job['job_id']}/download" if status == "completed" else None,
            "status_url": f"/jobs/{job['job_id']}",
            "poll_after_ms": 3000 if status in {"queued", "processing"} else 0,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    def schedule_job(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._job_tasks.add(task)
        task.add_done_callback(self._job_tasks.discard)

    def save_upload(self, tmp_dir: Path, upload: UploadFile) -> Path:
        ext = Path(upload.filename or "").suffix
        out_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
        with out_path.open("wb") as file_obj:
            shutil.copyfileobj(upload.file, file_obj)
        return out_path

    def detect_video_size(self, video_path: Path) -> tuple[int, int]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Could not read video dimensions")

        return width, height

    def probe_audio_presence(self, video_path: Path) -> bool:
        import ffmpeg

        try:
            probe = ffmpeg.probe(str(video_path), v="error", select_streams="a")
            return len(probe.get("streams", [])) > 0
        except ffmpeg.Error:
            return False

    def convert_video_fps(self, input_path: Path, target_fps: int, output_path: Path) -> Path:
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise RuntimeError(f"Video file is missing or empty: {input_path}")

        audio_present = self.probe_audio_presence(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-filter:v",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        if audio_present:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"])
        else:
            cmd.append("-an")

        cmd.append(str(output_path))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_path

    def get_video_duration(self, video_path: Path) -> float:
        import ffmpeg

        try:
            probe = ffmpeg.probe(
                str(video_path),
                v="error",
                select_streams="v:0",
                show_entries="format=duration",
            )
            return float(probe["format"]["duration"])
        except Exception as exc:
            raise RuntimeError(f"Unable to fetch video duration for {video_path}: {exc}") from exc

    def reverse_video(self, video_path: Path, audio_exists: bool, output_path: Path) -> Path:
        import ffmpeg

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = ffmpeg.input(str(video_path))
            if audio_exists:
                stream.output(str(output_path), vf="reverse", af="areverse").run(
                    overwrite_output=True,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            else:
                stream.output(str(output_path), vf="reverse").run(
                    overwrite_output=True,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            return output_path
        except ffmpeg.Error as exc:
            message = exc.stderr.decode() if exc.stderr else str(exc)
            raise RuntimeError(f"Video reverse failed: {message}") from exc

    def trim_video(self, video_path: Path, target_duration: float, output_path: Path) -> Path:
        import ffmpeg

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Video file is missing or empty: {video_path}")
        if target_duration <= 0:
            raise RuntimeError(f"Invalid target duration: {target_duration}")

        original_duration = self.get_video_duration(video_path)
        if original_duration <= target_duration:
            return video_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_audio = self.probe_audio_presence(video_path)
        input_stream = ffmpeg.input(str(video_path), ss=0, to=target_duration)

        output_args = {
            "c:v": "libx264",
            "preset": "fast",
            "crf": "18",
            "pix_fmt": "yuv420p",
            "movflags": "+faststart",
        }
        if has_audio:
            output_args["c:a"] = "aac"
            output_args["b:a"] = "192k"
            output_args["ar"] = "44100"
            output_args["ac"] = "2"

        cmd = input_stream.output(str(output_path), **output_args).compile()
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_path

    def extend_video(self, video_path: Path, target_duration: float, workdir: Path) -> Path:
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Video file is missing or empty: {video_path}")

        original_duration = self.get_video_duration(video_path)
        if original_duration >= target_duration:
            return video_path

        audio_exists = self.probe_audio_presence(video_path)
        clips = [video_path]
        total_duration = original_duration
        extension_index = 0

        while total_duration < target_duration:
            extension_index += 1
            if self._loop_video_from_endframe:
                reversed_clip = workdir / f"reversed_{extension_index}.mp4"
                clips.append(self.reverse_video(clips[-1], audio_exists, reversed_clip))
            else:
                clips.append(clips[-1])
            total_duration += original_duration

        concat_list_path = workdir / "concat_list.txt"
        extended_video_path = workdir / "extended_video.mp4"

        with concat_list_path.open("w", encoding="utf-8") as file_obj:
            for clip in clips:
                file_obj.write(f"file '{clip.resolve()}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(extended_video_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return extended_video_path

    def pad_audio_to_multiple_of_16_for_video(
        self,
        audio_path: Path,
        target_fps: int,
        output_path: Path,
    ) -> tuple[Path, int, float]:
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio_duration = waveform.shape[1] / sample_rate
        num_frames = int(audio_duration * target_fps)
        remainder = num_frames % 16

        if remainder > 0:
            pad_frames = 16 - remainder
            pad_samples = int((pad_frames / target_fps) * sample_rate)
            pad_waveform = torch.zeros((waveform.shape[0], pad_samples))
            waveform = torch.cat((waveform, pad_waveform), dim=1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), waveform, sample_rate)
            padded_audio_path = output_path
        else:
            padded_audio_path = audio_path

        padded_duration = waveform.shape[1] / sample_rate
        padded_frames = int(padded_duration * target_fps)
        return padded_audio_path, padded_frames, padded_duration

    def generate_liveportrait_reference_video(self, image_path: Path, job_dir: Path) -> Path:
        if not LIVEPORTRAIT_DIR.exists():
            raise RuntimeError(f"LivePortrait directory is missing: {LIVEPORTRAIT_DIR}")
        if not LIVEPORTRAIT_INFERENCE_SCRIPT.exists():
            raise RuntimeError(f"LivePortrait inference script is missing: {LIVEPORTRAIT_INFERENCE_SCRIPT}")
        if not LIVEPORTRAIT_IDLE_DRIVING_VIDEO.exists():
            raise RuntimeError(f"LivePortrait idle driving video is missing: {LIVEPORTRAIT_IDLE_DRIVING_VIDEO}")

        output_dir = job_dir / "liveportrait_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(LIVEPORTRAIT_INFERENCE_SCRIPT),
            "-s",
            str(image_path),
            "-d",
            str(LIVEPORTRAIT_IDLE_DRIVING_VIDEO),
            "-o",
            str(output_dir),
        ]

        try:
            subprocess.run(
                cmd,
                cwd=str(LIVEPORTRAIT_DIR),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"LivePortrait reference video generation failed: {details}") from exc

        reference_video = output_dir / f"{image_path.stem}--{LIVEPORTRAIT_IDLE_DRIVING_VIDEO.stem}.mp4"
        if not reference_video.exists() or reference_video.stat().st_size == 0:
            raise RuntimeError(
                f"LivePortrait did not produce the expected reference video: {reference_video}"
            )

        return reference_video

    def build_local_file_response(
        self,
        file_path: Path,
        filename: str,
        duration_seconds: float,
        background_tasks: BackgroundTasks,
        job_dir: Path,
    ) -> FileResponse:
        background_tasks.add_task(self.cleanup_job_dir, job_dir)
        return FileResponse(
            str(file_path),
            media_type="video/mp4",
            filename=filename,
            headers=self.get_processing_headers(duration_seconds),
            background=background_tasks,
        )

    def build_job_file_response(
        self,
        file_path: Path,
        filename: str,
        duration_seconds: float,
        media_type: str,
    ) -> FileResponse:
        return FileResponse(
            str(file_path),
            media_type=media_type,
            filename=filename,
            headers=self.get_processing_headers(duration_seconds),
        )

    def format_upstream_error(self, status_code: int, body_text: str) -> str:
        stripped = body_text.strip()
        if not stripped:
            return "Upstream request failed"
        lowered = stripped.lower()
        if "<html" in lowered or "<!doctype html" in lowered:
            return (
                f"Upstream request failed with status {status_code}. "
                "The remote server likely timed out before returning the video."
            )
        return stripped

    async def proxy_request(
        self,
        route_path: str,
        base_url: str,
        uploads: dict[str, UploadFile],
        form_fields: dict[str, str],
        download_filename: str,
        duration_seconds: float | None = None,
    ) -> Response:
        target_url = urljoin(f"{base_url}/", route_path.lstrip("/"))

        for upload in uploads.values():
            await upload.seek(0)

        files = {
            field_name: (
                upload.filename or field_name,
                upload.file,
                upload.content_type or "application/octet-stream",
            )
            for field_name, upload in uploads.items()
        }

        timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_response = await client.post(target_url, data=form_fields, files=files)

        if upstream_response.status_code >= 400:
            detail = self.format_upstream_error(upstream_response.status_code, upstream_response.text)
            try:
                payload = upstream_response.json()
                if isinstance(payload, dict) and "detail" in payload:
                    detail = str(payload["detail"])
            except ValueError:
                pass
            raise HTTPException(status_code=upstream_response.status_code, detail=detail)

        headers = {}
        if duration_seconds is not None:
            headers.update(self.get_processing_headers(duration_seconds))
        headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'

        return Response(
            content=upstream_response.content,
            media_type=upstream_response.headers.get("content-type", "video/mp4"),
            headers=headers,
        )

    async def proxy_request_to_file(
        self,
        route_path: str,
        base_url: str,
        uploads: dict[str, Path],
        form_fields: dict[str, str],
        output_path: Path,
    ) -> tuple[Path, str]:
        target_url = urljoin(f"{base_url}/", route_path.lstrip("/"))
        timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)
        file_handles = []

        try:
            files = {}
            for field_name, upload_path in uploads.items():
                file_obj = upload_path.open("rb")
                file_handles.append(file_obj)
                files[field_name] = (
                    upload_path.name,
                    file_obj,
                    mimetypes.guess_type(upload_path.name)[0] or "application/octet-stream",
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", target_url, data=form_fields, files=files) as upstream_response:
                    if upstream_response.status_code >= 400:
                        body = (await upstream_response.aread()).decode("utf-8", errors="replace")
                        detail = self.format_upstream_error(upstream_response.status_code, body)
                        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("wb") as file_obj:
                        async for chunk in upstream_response.aiter_bytes():
                            file_obj.write(chunk)

                    media_type = upstream_response.headers.get("content-type", "video/mp4")
                    return output_path, media_type
        finally:
            for file_obj in file_handles:
                file_obj.close()

    def build_target_url(self, base_url: str, route_path: str) -> str:
        return urljoin(f"{base_url}/", route_path.lstrip("/"))

    def resolve_remote_url(self, base_url: str, route_or_url: str) -> str:
        if route_or_url.startswith("http://") or route_or_url.startswith("https://"):
            return route_or_url
        return self.build_target_url(base_url, route_or_url)

    async def proxy_job_to_file(
        self,
        submit_route_path: str,
        base_url: str,
        uploads: dict[str, Path],
        form_fields: dict[str, str],
        output_path: Path,
    ) -> tuple[Path, str]:
        target_url = self.build_target_url(base_url, submit_route_path)
        timeout = httpx.Timeout(connect=30.0, read=60.0, write=None, pool=None)
        file_handles = []

        try:
            files = {}
            for field_name, upload_path in uploads.items():
                file_obj = upload_path.open("rb")
                file_handles.append(file_obj)
                files[field_name] = (
                    upload_path.name,
                    file_obj,
                    mimetypes.guess_type(upload_path.name)[0] or "application/octet-stream",
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                submit_response = await client.post(target_url, data=form_fields, files=files)

                if submit_response.status_code in {404, 405}:
                    raise HTTPException(
                        status_code=submit_response.status_code,
                        detail="Remote server does not support async job submission.",
                    )

                if submit_response.status_code >= 400:
                    detail = self.format_upstream_error(submit_response.status_code, submit_response.text)
                    raise HTTPException(status_code=submit_response.status_code, detail=detail)

                job_payload = submit_response.json()
                status_url = self.resolve_remote_url(base_url, str(job_payload.get("status_url") or ""))

                if not status_url:
                    raise RuntimeError("Remote server did not return a status URL.")

                while True:
                    status_response = await client.get(status_url)
                    if status_response.status_code >= 400:
                        detail = self.format_upstream_error(status_response.status_code, status_response.text)
                        raise HTTPException(status_code=status_response.status_code, detail=detail)

                    job_status = status_response.json()
                    status = str(job_status.get("status") or "")

                    if status == "completed":
                        download_url = self.resolve_remote_url(base_url, str(job_status.get("download_url") or ""))
                        if not download_url:
                            raise RuntimeError("Remote server completed the job without a download URL.")

                        async with client.stream("GET", download_url) as download_response:
                            if download_response.status_code >= 400:
                                body = (await download_response.aread()).decode("utf-8", errors="replace")
                                detail = self.format_upstream_error(download_response.status_code, body)
                                raise HTTPException(status_code=download_response.status_code, detail=detail)

                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with output_path.open("wb") as file_obj:
                                async for chunk in download_response.aiter_bytes():
                                    file_obj.write(chunk)

                            media_type = download_response.headers.get("content-type", "video/mp4")
                            return output_path, media_type

                    if status == "failed":
                        raise RuntimeError(str(job_status.get("error") or "Remote generation failed"))

                    poll_after_ms = int(job_status.get("poll_after_ms") or 3000)
                    await asyncio.sleep(max(poll_after_ms, 500) / 1000)
        finally:
            for file_obj in file_handles:
                file_obj.close()

    def resolve_num_steps(
        self,
        num_steps: int | None,
        steps: int | None,
        default_value: int = 40,
    ) -> int:
        return num_steps if num_steps is not None else (steps if steps is not None else default_value)

    def generate_video_to_file(
        self,
        video_path: Path,
        audio_path: Path,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        output_fps: int,
        job_dir: Path,
    ) -> Path:
        if self._inference_runner is _default_inference_runner:
            _verify_default_inference_runtime()

        self.detect_video_size(video_path)

        processing_video = self.convert_video_fps(video_path, self._processing_fps, job_dir / "processing_video.mp4")
        padded_audio_path, _, audio_duration = self.pad_audio_to_multiple_of_16_for_video(
            audio_path,
            self._processing_fps,
            job_dir / "padded_audio.wav",
        )
        video_duration = self.get_video_duration(processing_video)

        if audio_duration > video_duration:
            processing_video = self.extend_video(processing_video, audio_duration, job_dir)
            video_duration = self.get_video_duration(processing_video)
            if video_duration > audio_duration:
                processing_video = self.trim_video(processing_video, audio_duration, job_dir / "trimmed_video.mp4")
        elif video_duration > audio_duration:
            processing_video = self.trim_video(processing_video, audio_duration, job_dir / "trimmed_video.mp4")

        inference_output = job_dir / "inference_output.mp4"
        self._inference_runner(
            str(processing_video),
            str(padded_audio_path),
            seed,
            num_steps,
            guidance_scale,
            str(inference_output),
        )

        return self.convert_video_fps(inference_output, output_fps, job_dir / "output_video.mp4")

    def generate_image_to_file(
        self,
        image_path: Path,
        audio_path: Path,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        output_fps: int,
        job_dir: Path,
    ) -> Path:
        if self._inference_runner is _default_inference_runner:
            _verify_default_inference_runtime()

        reference_video = self.generate_liveportrait_reference_video(image_path, job_dir)
        return self.generate_video_to_file(
            reference_video,
            audio_path,
            seed,
            num_steps,
            guidance_scale,
            output_fps,
            job_dir,
        )

    def run_local_video_generation(
        self,
        video: UploadFile,
        audio: UploadFile,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        output_fps: int,
        background_tasks: BackgroundTasks,
        duration_seconds: float,
    ) -> FileResponse:
        job_dir = self._tmp_root / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            video.file.seek(0)
            audio.file.seek(0)
            video_path = self.save_upload(job_dir, video)
            audio_path = self.save_upload(job_dir, audio)
            final_output = self.generate_video_to_file(
                video_path,
                audio_path,
                seed,
                num_steps,
                guidance_scale,
                output_fps,
                job_dir,
            )
            return self.build_local_file_response(
                final_output,
                "output_video.mp4",
                duration_seconds,
                background_tasks,
                job_dir,
            )
        except Exception:
            self.cleanup_job_dir(job_dir)
            raise

    def run_local_image_generation(
        self,
        image: UploadFile,
        audio: UploadFile,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        output_fps: int,
        background_tasks: BackgroundTasks,
        duration_seconds: float,
    ) -> FileResponse:
        job_dir = self._tmp_root / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            image.file.seek(0)
            audio.file.seek(0)
            image_path = self.save_upload(job_dir, image)
            audio_path = self.save_upload(job_dir, audio)
            final_output = self.generate_image_to_file(
                image_path,
                audio_path,
                seed,
                num_steps,
                guidance_scale,
                output_fps,
                job_dir,
            )
            return self.build_local_file_response(
                final_output,
                "lipsync_output.mp4",
                duration_seconds,
                background_tasks,
                job_dir,
            )
        except Exception:
            self.cleanup_job_dir(job_dir)
            raise

    async def process_video_job(
        self,
        job_id: str,
        route_name: str,
        mode: str,
        base_url: str | None,
        video_path: Path,
        audio_path: Path,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        video_scale: float,
        output_fps: int,
    ) -> None:
        started_at = time.perf_counter()
        self.update_job_record(job_id, status="processing", started_at=time.time(), error=None)
        job = self.get_job_record(job_id)
        if not job:
            return

        job_dir = Path(str(job["job_dir"]))

        try:
            if mode == "proxy" and base_url:
                try:
                    result_path, media_type = await self.proxy_job_to_file(
                        "/jobs/generate-from-video",
                        base_url,
                        {"video": video_path, "audio": audio_path},
                        {
                            "seed": str(seed),
                            "num_steps": str(num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        job_dir / "output_video.mp4",
                    )
                except HTTPException as exc:
                    if exc.status_code not in {404, 405}:
                        raise
                    result_path, media_type = await self.proxy_request_to_file(
                        route_name,
                        base_url,
                        {"video": video_path, "audio": audio_path},
                        {
                            "seed": str(seed),
                            "num_steps": str(num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        job_dir / "output_video.mp4",
                    )
            else:
                result_path = await asyncio.to_thread(
                    self.generate_video_to_file,
                    video_path,
                    audio_path,
                    seed,
                    num_steps,
                    guidance_scale,
                    output_fps,
                    job_dir,
                )
                media_type = "video/mp4"

            duration_seconds = time.perf_counter() - started_at
            self.update_job_record(
                job_id,
                status="completed",
                completed_at=time.time(),
                duration_seconds=duration_seconds,
                result_path=str(result_path),
                media_type=media_type,
                error=None,
            )
            self.log_request_timing(route_name, mode, base_url, True, duration_seconds)
        except Exception as exc:
            duration_seconds = time.perf_counter() - started_at
            self.update_job_record(
                job_id,
                status="failed",
                completed_at=time.time(),
                duration_seconds=duration_seconds,
                error=str(exc),
            )
            self.log_request_timing(route_name, mode, base_url, False, duration_seconds, str(exc))

    async def process_image_job(
        self,
        job_id: str,
        route_name: str,
        mode: str,
        base_url: str | None,
        image_path: Path,
        audio_path: Path,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        video_scale: float,
        output_fps: int,
    ) -> None:
        started_at = time.perf_counter()
        self.update_job_record(job_id, status="processing", started_at=time.time(), error=None)
        job = self.get_job_record(job_id)
        if not job:
            return

        job_dir = Path(str(job["job_dir"]))

        try:
            if mode == "proxy" and base_url:
                try:
                    result_path, media_type = await self.proxy_job_to_file(
                        "/jobs/generate-from-image",
                        base_url,
                        {"image": image_path, "audio": audio_path},
                        {
                            "seed": str(seed),
                            "num_steps": str(num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        job_dir / "lipsync_output.mp4",
                    )
                except HTTPException as exc:
                    if exc.status_code not in {404, 405}:
                        raise
                    result_path, media_type = await self.proxy_request_to_file(
                        route_name,
                        base_url,
                        {"image": image_path, "audio": audio_path},
                        {
                            "seed": str(seed),
                            "num_steps": str(num_steps),
                            "guidance_scale": str(guidance_scale),
                            "video_scale": str(video_scale),
                            "output_fps": str(output_fps),
                        },
                        job_dir / "lipsync_output.mp4",
                    )
            else:
                result_path = await asyncio.to_thread(
                    self.generate_image_to_file,
                    image_path,
                    audio_path,
                    seed,
                    num_steps,
                    guidance_scale,
                    output_fps,
                    job_dir,
                )
                media_type = "video/mp4"

            duration_seconds = time.perf_counter() - started_at
            self.update_job_record(
                job_id,
                status="completed",
                completed_at=time.time(),
                duration_seconds=duration_seconds,
                result_path=str(result_path),
                media_type=media_type,
                error=None,
            )
            self.log_request_timing(route_name, mode, base_url, True, duration_seconds)
        except Exception as exc:
            duration_seconds = time.perf_counter() - started_at
            self.update_job_record(
                job_id,
                status="failed",
                completed_at=time.time(),
                duration_seconds=duration_seconds,
                error=str(exc),
            )
            self.log_request_timing(route_name, mode, base_url, False, duration_seconds, str(exc))
