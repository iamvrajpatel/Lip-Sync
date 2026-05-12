import io
import tempfile
import types
import unittest
from pathlib import Path

from tts_api.catalog import UnsupportedLanguageError, build_default_catalog

try:
    import httpx

    from app_factory import create_app

    FASTAPI_TESTS_AVAILABLE = True
except ModuleNotFoundError:
    httpx = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    FASTAPI_TESTS_AVAILABLE = False


class FakeWebService:
    def __init__(self, catalog) -> None:  # type: ignore[no-untyped-def]
        self.catalog = catalog

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_generating(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "fake-web-model"

    @property
    def load_error(self) -> str | None:
        return None

    @property
    def device(self) -> str | None:
        return "cpu"

    async def load(self) -> None:
        return None

    async def synthesize(
        self,
        language_code: str,
        text: str,
        speaker_name: str | None = None,
        voice_description: str | None = None,
        cancellation=None,  # type: ignore[no-untyped-def]
    ) -> bytes:
        if voice_description:
            self.catalog.get_language(language_code)
        else:
            self.catalog.get_speaker(language_code, speaker_name)
        return b"RIFFfakewav"


class FakeCloneWebService:
    def __init__(self, catalog) -> None:  # type: ignore[no-untyped-def]
        self.catalog = catalog

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_generating(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "fake-clone-model"

    @property
    def load_error(self) -> str | None:
        return None

    @property
    def device(self) -> str | None:
        return "cpu"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("en", "hi")

    async def load(self) -> None:
        return None

    async def synthesize(
        self,
        language_code: str,
        text: str,
        reference_audio: bytes,
        reference_filename: str | None = None,
        cancellation=None,  # type: ignore[no-untyped-def]
    ) -> bytes:
        if not reference_audio:
            raise ValueError("Reference audio file is empty.")
        language = self.catalog.get_language(language_code)
        if language.code not in self.supported_languages:
            raise UnsupportedLanguageError(
                f"Voice cloning is not available for {language.display_name}. "
                "Supported clone languages: English, Hindi."
            )
        return b"RIFFclonewav"


@unittest.skipUnless(FASTAPI_TESTS_AVAILABLE, "fastapi is not installed")
class CombinedAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_default_catalog()
        self.tts_service = FakeWebService(self.catalog)
        self.clone_service = FakeCloneWebService(self.catalog)

    def create_app_instance(self, lip_sync_service=None):  # type: ignore[no-untyped-def]
        return create_app(
            service=self.tts_service,
            clone_service=self.clone_service,
            catalog=self.catalog,
            lip_sync_service=lip_sync_service or FakeLipSyncStatusService(),
        )

    def request(self, app, method: str, path: str, **kwargs):  # type: ignore[no-untyped-def]
        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        import asyncio

        return asyncio.run(scenario())

    def test_combined_root_and_ui_render_both_workspaces(self) -> None:
        app = self.create_app_instance()
        root_response = self.request(app, "GET", "/")
        ui_response = self.request(app, "GET", "/ui")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(ui_response.status_code, 200)
        self.assertIn("Lip-sync", root_response.text)
        self.assertIn("TTS / Voice Clone", root_response.text)
        self.assertIn("Backend sync", root_response.text)
        self.assertIn("Clone from reference audio", root_response.text)
        self.assertIn("Lip-sync", ui_response.text)
        self.assertIn("TTS / Voice Clone", ui_response.text)

    def test_health_includes_lip_sync_status(self) -> None:
        response = self.request(self.create_app_instance(), "GET", "/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertIn("voice_clone", payload)
        self.assertIn("lip_sync", payload)
        self.assertIn("active_job_count", payload["lip_sync"])

    def test_expected_routes_are_registered(self) -> None:
        from lipsync_app import LipSyncService

        app = create_app(
            service=self.tts_service,
            clone_service=self.clone_service,
            catalog=self.catalog,
            lip_sync_service=LipSyncService(tmp_root=Path(tempfile.mkdtemp(prefix="lip-routes-"))),
        )
        route_paths = {route.path for route in app.routes}
        expected_paths = {
            "/",
            "/ui",
            "/health",
            "/tts/",
            "/clone-voice",
            "/generate-from-video",
            "/generate-from-image",
            "/jobs/generate-from-video",
            "/jobs/generate-from-image",
            "/jobs/{job_id}",
            "/jobs/{job_id}/download",
        }
        self.assertTrue(expected_paths.issubset(route_paths))

    def test_lip_sync_local_generation_methods_return_file_responses(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lip-jobs-") as tmp_dir:
            from lipsync_app import LipSyncService
            from fastapi import BackgroundTasks
            from fastapi import UploadFile

            service = LipSyncService(tmp_root=Path(tmp_dir))

            def fake_generate_video_to_file(self, video_path, audio_path, seed, num_steps, guidance_scale, output_fps, job_dir):  # type: ignore[no-untyped-def]
                del video_path, audio_path, seed, num_steps, guidance_scale, output_fps
                output_path = Path(job_dir) / "output_video.mp4"
                output_path.write_bytes(b"fake-video")
                return output_path

            def fake_generate_image_to_file(self, image_path, audio_path, seed, num_steps, guidance_scale, output_fps, job_dir):  # type: ignore[no-untyped-def]
                del image_path, audio_path, seed, num_steps, guidance_scale, output_fps
                output_path = Path(job_dir) / "lipsync_output.mp4"
                output_path.write_bytes(b"fake-image-video")
                return output_path

            service.generate_video_to_file = types.MethodType(fake_generate_video_to_file, service)
            service.generate_image_to_file = types.MethodType(fake_generate_image_to_file, service)

            video_response = service.run_local_video_generation(
                video=UploadFile(filename="face.mp4", file=io.BytesIO(b"video-bytes")),
                audio=UploadFile(filename="voice.wav", file=io.BytesIO(b"audio-bytes")),
                seed=1247,
                num_steps=40,
                guidance_scale=1.0,
                output_fps=30,
                background_tasks=BackgroundTasks(),
                duration_seconds=0.0,
            )
            self.assertEqual(video_response.filename, "output_video.mp4")
            self.assertEqual(Path(video_response.path).read_bytes(), b"fake-video")

            image_response = service.run_local_image_generation(
                image=UploadFile(filename="face.png", file=io.BytesIO(b"image-bytes")),
                audio=UploadFile(filename="voice.wav", file=io.BytesIO(b"audio-bytes")),
                seed=1247,
                num_steps=40,
                guidance_scale=1.0,
                output_fps=30,
                background_tasks=BackgroundTasks(),
                duration_seconds=0.0,
            )
            self.assertEqual(image_response.filename, "lipsync_output.mp4")
            self.assertEqual(Path(image_response.path).read_bytes(), b"fake-image-video")

    def test_lip_sync_job_record_serialization_and_download_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lip-job-status-") as tmp_dir:
            from lipsync_app import LipSyncService

            service = LipSyncService(tmp_root=Path(tmp_dir))
            job = service.create_job_record(
                route_name="/generate-from-video",
                mode="local",
                base_url=None,
                result_filename="output_video.mp4",
            )
            job_dir = Path(str(job["job_dir"]))
            output_path = job_dir / "output_video.mp4"
            output_path.write_bytes(b"prepared-video")
            service.update_job_record(
                str(job["job_id"]),
                status="completed",
                duration_seconds=0.123,
                result_path=str(output_path),
                media_type="video/mp4",
                error=None,
            )

            payload = service.serialize_job_record(service.get_job_record(str(job["job_id"])) or job)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["download_url"], f"/jobs/{job['job_id']}/download")

            response = service.build_job_file_response(
                output_path,
                "output_video.mp4",
                0.123,
                "video/mp4",
            )
            self.assertEqual(response.filename, "output_video.mp4")
            self.assertEqual(response.media_type, "video/mp4")


class FakeLipSyncStatusService:
    def build_router(self):  # type: ignore[no-untyped-def]
        from fastapi import APIRouter

        return APIRouter()

    def get_health_payload(self) -> dict[str, object]:
        return {"active_job_count": 0, "processing_fps": 25}
