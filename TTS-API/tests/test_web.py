import unittest

from tts_api.catalog import build_default_catalog
from tts_api.services import InvalidReferenceAudioError
from tts_api.services import ModelLoadError
from tts_api.services import ModelNotReadyError
from tts_api.services import VoiceCloneLoadError
from tts_api.catalog import UnsupportedLanguageError

try:
    from fastapi import APIRouter
    import httpx

    from app_factory import create_app

    FASTAPI_TESTS_AVAILABLE = True
except ModuleNotFoundError:
    httpx = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    FASTAPI_TESTS_AVAILABLE = False


class FakeWebService:
    def __init__(
        self,
        catalog,
        ready: bool = True,
        load_error: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        self.catalog = catalog
        self.ready = ready
        self._load_error = load_error
        self.calls: list[tuple[str, str, str]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def is_generating(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "fake-web-model"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return "cpu" if self.ready else None

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
        if self._load_error:
            raise ModelLoadError(self._load_error)
        if not self.ready:
            raise ModelNotReadyError("The TTS model is still loading.")
        if voice_description:
            self.catalog.get_language(language_code)
        else:
            self.catalog.get_speaker(language_code, speaker_name)
        self.calls.append((language_code, speaker_name or "custom-description", text))
        return b"RIFFfakewav"


class FakeCloneWebService:
    def __init__(
        self,
        catalog,
        ready: bool = False,
        load_error: str | None = None,
        supported_languages: tuple[str, ...] | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        self.catalog = catalog
        self.ready = ready
        self._load_error = load_error
        self._supported_languages = supported_languages or ("en", "hi")
        self.calls: list[tuple[str, str, bytes]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def is_generating(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "fake-clone-model"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return "cpu" if self.ready else None

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self._supported_languages

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
        if self._load_error:
            raise VoiceCloneLoadError(self._load_error)
        if not reference_audio:
            raise InvalidReferenceAudioError("Reference audio file is empty.")
        language = self.catalog.get_language(language_code)
        if language.code not in self._supported_languages:
            raise UnsupportedLanguageError(
                f"Voice cloning is not available for {language.display_name}. "
                "Supported clone languages: English, Hindi."
            )
        self.calls.append((language_code, reference_filename or "", reference_audio))
        return b"RIFFclonewav"


class FakeLipSyncService:
    def build_router(self):  # type: ignore[no-untyped-def]
        return APIRouter()

    def get_health_payload(self) -> dict[str, object]:
        return {"active_job_count": 0, "processing_fps": 25}


@unittest.skipUnless(FASTAPI_TESTS_AVAILABLE, "fastapi is not installed")
class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_default_catalog()
        self.clone_service = FakeCloneWebService(self.catalog, ready=True)

    def create_app_instance(self, service=None, clone_service=None):  # type: ignore[no-untyped-def]
        return create_app(
            service=service or FakeWebService(self.catalog),
            clone_service=clone_service or self.clone_service,
            catalog=self.catalog,
            lip_sync_service=FakeLipSyncService(),
        )

    def request(self, app, method: str, path: str, **kwargs):  # type: ignore[no-untyped-def]
        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        import asyncio

        return asyncio.run(scenario())

    def test_root_renders_combined_page(self) -> None:
        response = self.request(self.create_app_instance(), "GET", "/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Lip-sync", response.text)
        self.assertIn("TTS / Voice Clone", response.text)
        self.assertIn("Backend sync", response.text)
        self.assertIn("Clone from reference audio", response.text)

    def test_ui_alias_renders_combined_page(self) -> None:
        response = self.request(self.create_app_instance(), "GET", "/ui")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Lip-sync", response.text)
        self.assertIn("TTS / Voice Clone", response.text)

    def test_health_reflects_loaded_state(self) -> None:
        ready_response = self.request(
            self.create_app_instance(
                service=FakeWebService(self.catalog, ready=True),
                clone_service=FakeCloneWebService(self.catalog, ready=True),
            ),
            "GET",
            "/health",
        )
        loading_response = self.request(
            self.create_app_instance(
                service=FakeWebService(self.catalog, ready=False),
                clone_service=FakeCloneWebService(self.catalog, ready=False),
            ),
            "GET",
            "/health",
        )

        self.assertEqual(ready_response.status_code, 200)
        self.assertTrue(ready_response.json()["ready"])
        self.assertFalse(ready_response.json()["generation_in_progress"])
        self.assertIn("en", ready_response.json()["voice_clone"]["supported_languages"])
        self.assertIn("hi", ready_response.json()["voice_clone"]["supported_languages"])
        self.assertIn("lip_sync", ready_response.json())
        self.assertEqual(loading_response.status_code, 503)
        self.assertFalse(loading_response.json()["ready"])

    def test_health_reports_model_error(self) -> None:
        response = self.request(
            self.create_app_instance(
                service=FakeWebService(
                    self.catalog,
                    ready=False,
                    load_error="Model loading failed: torchaudio mismatch",
                )
            ),
            "GET",
            "/health",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "error")
        self.assertIn("torchaudio mismatch", response.json()["model_error"])

    def test_tts_returns_audio_for_valid_request(self) -> None:
        service = FakeWebService(self.catalog, ready=True)
        response = self.request(
            self.create_app_instance(service=service),
            "POST",
            "/tts/",
            json={
                "text": "Namaste",
                "language": "hi",
                "voice_mode": "speaker",
                "speaker": "Divya",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertIn("X-Generation-Time-Ms", response.headers)
        self.assertEqual(service.calls, [("hi", "Divya", "Namaste")])

    def test_tts_rejects_invalid_speaker(self) -> None:
        response = self.request(
            self.create_app_instance(),
            "POST",
            "/tts/",
            json={
                "text": "Namaste",
                "language": "hi",
                "voice_mode": "speaker",
                "speaker": "Mary",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_tts_rejects_empty_text(self) -> None:
        response = self.request(
            self.create_app_instance(),
            "POST",
            "/tts/",
            json={
                "text": "   ",
                "language": "hi",
                "voice_mode": "speaker",
                "speaker": "Divya",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_tts_accepts_custom_description_mode(self) -> None:
        service = FakeWebService(self.catalog, ready=True)
        response = self.request(
            self.create_app_instance(service=service),
            "POST",
            "/tts/",
            json={
                "text": "Namaste",
                "language": "hi",
                "voice_mode": "description",
                "voice_description": "warm custom voice with a calm pace",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(
            service.calls,
            [("hi", "custom-description", "Namaste")],
        )

    def test_clone_voice_accepts_reference_audio_upload(self) -> None:
        clone_service = FakeCloneWebService(self.catalog, ready=True)
        response = self.request(
            self.create_app_instance(clone_service=clone_service),
            "POST",
            "/clone-voice",
            data={
                "text": "Hello from the cloned voice",
                "language": "en",
            },
            files={"reference_audio": ("voice.wav", b"wave-bytes", "audio/wav")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["X-Voice-Mode"], "clone")
        self.assertEqual(
            clone_service.calls,
            [("en", "voice.wav", b"wave-bytes")],
        )

    def test_clone_voice_rejects_empty_reference_audio(self) -> None:
        response = self.request(
            self.create_app_instance(),
            "POST",
            "/clone-voice",
            data={
                "text": "Hello from the cloned voice",
                "language": "en",
            },
            files={"reference_audio": ("voice.wav", b"", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)

    def test_clone_voice_rejects_unsupported_language(self) -> None:
        response = self.request(
            self.create_app_instance(),
            "POST",
            "/clone-voice",
            data={
                "text": "Kem cho?",
                "language": "gu",
            },
            files={"reference_audio": ("voice.wav", b"wave-bytes", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Voice cloning is not available for Gujarati", response.json()["detail"])
