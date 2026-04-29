import gc
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import cv2
import ffmpeg
import httpx
import torch
import torchaudio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates

from inference import perform_inference


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

print("running on : ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lip_sync_api")

BASE_DIR = Path(__file__).resolve().parent
TMP_ROOT = BASE_DIR / "tmp"
TEMPLATE_DIR = BASE_DIR / "templates"
PROCESSING_FPS = 25
loop_vid_from_endframe = True

TMP_ROOT.mkdir(exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Processing-Time"],
)


def get_processing_headers(duration_seconds: float) -> dict[str, str]:
    return {
        "X-Processing-Time": f"{duration_seconds:.3f}",
        "Access-Control-Expose-Headers": "Content-Disposition, X-Processing-Time",
    }


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    trimmed = base_url.strip()
    return trimmed.rstrip("/") if trimmed else None


def log_request_timing(
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


def cleanup_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_upload(tmp_dir: Path, upload: UploadFile) -> Path:
    ext = Path(upload.filename or "").suffix
    out_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
    with out_path.open("wb") as file_obj:
        shutil.copyfileobj(upload.file, file_obj)
    return out_path


def detect_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Could not read video dimensions")

    return width, height


def probe_audio_presence(video_path: Path) -> bool:
    try:
        probe = ffmpeg.probe(str(video_path), v="error", select_streams="a")
        return len(probe.get("streams", [])) > 0
    except ffmpeg.Error:
        return False


def convert_video_fps(input_path: Path, target_fps: int, output_path: Path) -> Path:
    if not input_path.exists() or input_path.stat().st_size == 0:
        raise RuntimeError(f"Video file is missing or empty: {input_path}")

    audio_present = probe_audio_presence(input_path)
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


def get_video_duration(video_path: Path) -> float:
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


def reverse_video(video_path: Path, audio_exists: bool, output_path: Path) -> Path:
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


def trim_video(video_path: Path, target_duration: float, output_path: Path) -> Path:
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video file is missing or empty: {video_path}")
    if target_duration <= 0:
        raise RuntimeError(f"Invalid target duration: {target_duration}")

    original_duration = get_video_duration(video_path)
    if original_duration <= target_duration:
        return video_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = probe_audio_presence(video_path)
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


def extend_video(video_path: Path, target_duration: float, workdir: Path) -> Path:
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video file is missing or empty: {video_path}")

    original_duration = get_video_duration(video_path)
    if original_duration >= target_duration:
        return video_path

    audio_exists = probe_audio_presence(video_path)
    clips = [video_path]
    total_duration = original_duration
    extension_index = 0

    while total_duration < target_duration:
        extension_index += 1
        if loop_vid_from_endframe:
            reversed_clip = workdir / f"reversed_{extension_index}.mp4"
            clips.append(reverse_video(clips[-1], audio_exists, reversed_clip))
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
    audio_path: Path,
    target_fps: int,
    output_path: Path,
) -> tuple[Path, int, float]:
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


def pad_audio_to_multiple_of_16_for_audio(
    audio_path: Path,
    target_fps: int,
    output_path: Path,
) -> tuple[Path, int]:
    padded_audio_path, padded_num_frames, _ = pad_audio_to_multiple_of_16_for_video(
        audio_path,
        target_fps,
        output_path,
    )
    return padded_audio_path, padded_num_frames


def create_video_from_image(image_path: Path, output_video_path: Path, num_frames: int, fps: int = 25) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Unable to read the image file")

    height, width, _ = image.shape
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for _ in range(num_frames):
        writer.write(image)

    writer.release()
    return output_video_path


def build_local_file_response(
    file_path: Path,
    filename: str,
    duration_seconds: float,
    background_tasks: BackgroundTasks,
    job_dir: Path,
) -> FileResponse:
    background_tasks.add_task(cleanup_job_dir, job_dir)
    response = FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers=get_processing_headers(duration_seconds),
        background=background_tasks,
    )
    return response


async def proxy_request(
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
        detail = upstream_response.text.strip() or "Upstream request failed"
        try:
            payload = upstream_response.json()
            if isinstance(payload, dict) and "detail" in payload:
                detail = str(payload["detail"])
        except ValueError:
            pass
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    headers = {}
    if duration_seconds is not None:
        headers.update(get_processing_headers(duration_seconds))
    headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'

    return Response(
        content=upstream_response.content,
        media_type=upstream_response.headers.get("content-type", "video/mp4"),
        headers=headers,
    )


def resolve_num_steps(num_steps: int | None, steps: int | None, default_value: int = 40) -> int:
    return num_steps if num_steps is not None else (steps if steps is not None else default_value)


def run_local_video_generation(
    video: UploadFile,
    audio: UploadFile,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    output_fps: int,
    background_tasks: BackgroundTasks,
    duration_seconds: float,
) -> FileResponse:
    job_dir = TMP_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        video.file.seek(0)
        audio.file.seek(0)

        video_path = save_upload(job_dir, video)
        audio_path = save_upload(job_dir, audio)

        detect_video_size(video_path)

        processing_video = convert_video_fps(video_path, PROCESSING_FPS, job_dir / "processing_video.mp4")
        padded_audio_path, _, audio_duration = pad_audio_to_multiple_of_16_for_video(
            audio_path,
            PROCESSING_FPS,
            job_dir / "padded_audio.wav",
        )
        video_duration = get_video_duration(processing_video)

        if audio_duration > video_duration:
            processing_video = extend_video(processing_video, audio_duration, job_dir)
            video_duration = get_video_duration(processing_video)
            if video_duration > audio_duration:
                processing_video = trim_video(processing_video, audio_duration, job_dir / "trimmed_video.mp4")
        elif video_duration > audio_duration:
            processing_video = trim_video(processing_video, audio_duration, job_dir / "trimmed_video.mp4")

        inference_output = job_dir / "inference_output.mp4"
        perform_inference(
            str(processing_video),
            str(padded_audio_path),
            seed,
            num_steps,
            guidance_scale,
            str(inference_output),
        )

        final_output = convert_video_fps(inference_output, output_fps, job_dir / "output_video.mp4")
        return build_local_file_response(
            final_output,
            "output_video.mp4",
            duration_seconds,
            background_tasks,
            job_dir,
        )
    except Exception:
        cleanup_job_dir(job_dir)
        raise


def run_local_image_generation(
    image: UploadFile,
    audio: UploadFile,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    output_fps: int,
    background_tasks: BackgroundTasks,
    duration_seconds: float,
) -> FileResponse:
    job_dir = TMP_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        image.file.seek(0)
        audio.file.seek(0)

        image_path = save_upload(job_dir, image)
        audio_path = save_upload(job_dir, audio)

        padded_audio_path, num_frames = pad_audio_to_multiple_of_16_for_audio(
            audio_path,
            PROCESSING_FPS,
            job_dir / "padded_audio.wav",
        )

        raw_video = create_video_from_image(
            image_path,
            job_dir / "input_video.mp4",
            num_frames,
            fps=PROCESSING_FPS,
        )

        inference_output = job_dir / "generated_video.mp4"
        perform_inference(
            str(raw_video),
            str(padded_audio_path),
            seed,
            num_steps,
            guidance_scale,
            str(inference_output),
        )

        final_output = convert_video_fps(inference_output, output_fps, job_dir / "lipsync_output.mp4")
        return build_local_file_response(
            final_output,
            "lipsync_output.mp4",
            duration_seconds,
            background_tasks,
            job_dir,
        )
    except Exception:
        cleanup_job_dir(job_dir)
        raise


@app.get("/")
async def root():
    return {"status": "Lip-Sync Running!!"}


@app.get("/ui")
async def ui_page(request: Request):
    return templates.TemplateResponse(request, "lipsync.html", {})


@app.post("/generate-from-video")
async def generate_from_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description=".mp4 file"),
    audio: UploadFile = File(..., description=".wav/.mp3/.aac/.flac"),
    base_url: str | None = Form(None),
    seed: int = Form(1247),
    num_steps: int = Form(40, ge=1, le=100),
    guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
    video_scale: float = Form(0.8, ge=0.1, le=1.0),
    output_fps: int = Form(30, ge=6, le=60),
):
    route_name = "/generate-from-video"
    normalized_base_url = normalize_base_url(base_url)
    mode = "proxy" if normalized_base_url else "local"
    started_at = time.perf_counter()

    try:
        if normalized_base_url:
            response = await proxy_request(
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
            response = run_local_video_generation(
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
        response.headers.update(get_processing_headers(duration_seconds))
        log_request_timing(route_name, mode, normalized_base_url, True, duration_seconds)
        return response
    except Exception as exc:
        duration_seconds = time.perf_counter() - started_at
        log_request_timing(route_name, mode, normalized_base_url, False, duration_seconds, str(exc))
        raise


@app.post("/generate-from-image")
async def generate_from_image(
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
):
    resolved_num_steps = resolve_num_steps(num_steps, steps)
    route_name = "/generate-from-image"
    normalized_base_url = normalize_base_url(base_url)
    mode = "proxy" if normalized_base_url else "local"
    started_at = time.perf_counter()

    try:
        if normalized_base_url:
            response = await proxy_request(
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
            response = run_local_image_generation(
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
        response.headers.update(get_processing_headers(duration_seconds))
        log_request_timing(route_name, mode, normalized_base_url, True, duration_seconds)
        return response
    except Exception as exc:
        duration_seconds = time.perf_counter() - started_at
        log_request_timing(route_name, mode, normalized_base_url, False, duration_seconds, str(exc))
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")
