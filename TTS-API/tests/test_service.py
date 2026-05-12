import asyncio
import logging
import types
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np

from tts_api.catalog import UnsupportedLanguageError, build_default_catalog
from tts_api.services import (
    InvalidReferenceAudioError,
    IndicParlerTTSService,
    LoadedModelBundle,
    LoadedVoiceCloneBundle,
    ModelLoadError,
    ModelNotReadyError,
    ReferenceAudioPreprocessor,
    RequestCancellation,
    RequestCancelledError,
    ServiceBusyError,
    SynthesisConcurrencyGate,
    XttsVoiceCloneService,
    _ensure_generation_mixin_support,
    _suppress_generation_mixin_log_warning,
)


class FakeBatch:
    def __init__(self) -> None:
        self.input_ids = [[1, 2, 3]]
        self.attention_mask = [[1, 1, 1]]

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> FakeBatch:
        self.last_text = text
        self.last_return_tensors = return_tensors
        return FakeBatch()


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = np.array(values, dtype=np.float32)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_calls.append(kwargs)
        return FakeTensor([[0.1, -0.1, 0.2, -0.2]])


def fake_loader() -> LoadedModelBundle:
    return LoadedModelBundle(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        description_tokenizer=FakeTokenizer(),
        device="cpu",
        sampling_rate=24000,
        model_name="fake-model",
    )


class BusyGate:
    @asynccontextmanager
    async def claim(self):
        raise ServiceBusyError("The server is currently processing another request.")
        yield

    @property
    def is_generating(self) -> bool:
        return True


class FakeCloneModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def tts(self, text: str, speaker_wav: str, language: str) -> list[float]:
        self.calls.append(
            {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
            }
        )
        return [0.1, -0.1, 0.2, -0.2]


class CancellingCloneModel:
    def __init__(self, cancellation: RequestCancellation) -> None:
        self._cancellation = cancellation
        self.calls: list[str] = []

    def tts(self, text: str, speaker_wav: str, language: str) -> list[float]:
        self.calls.append(text)
        if len(self.calls) == 1:
            self._cancellation.cancel()
        return [0.1, -0.1, 0.2, -0.2]


def fake_clone_loader() -> LoadedVoiceCloneBundle:
    return LoadedVoiceCloneBundle(
        model=FakeCloneModel(),
        device="cpu",
        sampling_rate=24000,
        model_name="fake-clone-model",
    )


class FakeReferenceAudioPreprocessor:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None]] = []

    def prepare(self, audio_bytes: bytes, filename: str | None = None):  # type: ignore[no-untyped-def]
        import tempfile
        from pathlib import Path

        if not audio_bytes:
            raise InvalidReferenceAudioError("Reference audio file is empty.")

        self.calls.append((audio_bytes, filename))
        temp_dir = tempfile.mkdtemp(prefix="clone-test-")
        audio_path = Path(temp_dir) / "reference.wav"
        audio_path.write_bytes(b"fake-reference")
        return type(
            "PreparedReferenceAudioStub",
            (),
            {
                "path": str(audio_path),
                "temp_dir": temp_dir,
            },
        )()


class ServiceTests(unittest.TestCase):
    def test_generation_mixin_log_warning_filter_suppresses_target_message(self) -> None:
        transformer_logger = logging.getLogger("transformers.modeling_utils")

        with self.assertLogs("transformers.modeling_utils", level="WARNING") as captured:
            with _suppress_generation_mixin_log_warning():
                transformer_logger.warning(
                    "GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. "
                    "However, it doesn't directly inherit from `GenerationMixin`."
                )
                transformer_logger.warning("A different warning should still be visible.")

        self.assertEqual(captured.output, ["WARNING:transformers.modeling_utils:A different warning should still be visible."])

    def test_generation_mixin_helper_updates_non_generating_classes(self) -> None:
        class FakeGenerationMixin:
            pass

        class FakePreTrainedModel:
            pass

        class FakeModel(FakePreTrainedModel):
            pass

        _ensure_generation_mixin_support(
            FakeModel,
            generation_mixin_class=FakeGenerationMixin,
        )

        self.assertTrue(issubclass(FakeModel, FakeGenerationMixin))

    def test_generation_mixin_helper_uses_transformers_when_available(self) -> None:
        class FakeGenerationMixin:
            pass

        class FakePreTrainedModel:
            pass

        class FakeModel(FakePreTrainedModel):
            pass

        fake_generation_utils = types.SimpleNamespace(GenerationMixin=FakeGenerationMixin)

        with patch.dict(
            "sys.modules",
            {"transformers.generation.utils": fake_generation_utils},
        ):
            _ensure_generation_mixin_support(FakeModel)

        self.assertTrue(issubclass(FakeModel, FakeGenerationMixin))

    def create_service(self) -> IndicParlerTTSService:
        return IndicParlerTTSService(
            catalog=build_default_catalog(),
            model_loader=fake_loader,
        )

    def test_startup_preload_succeeds(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            self.assertTrue(service.is_ready)
            self.assertEqual(service.model_name, "fake-model")

        asyncio.run(scenario())

    def test_model_not_ready_raises(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            with self.assertRaises(ModelNotReadyError):
                await service.synthesize("hi", "Divya", "Namaste")

        asyncio.run(scenario())

    def test_synthesis_uses_speaker_description(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"WAV"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="hi",
                speaker_name="Divya",
                text="Namaste",
            )

            self.assertEqual(wav_bytes, b"WAV")
            self.assertEqual(service._bundle.description_tokenizer.last_return_tensors, "pt")
            self.assertIn(
                "Divya's voice sounds",
                service._bundle.description_tokenizer.last_text,
            )

        asyncio.run(scenario())

    def test_synthesis_stops_when_request_is_cancelled(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            cancellation = RequestCancellation()
            cancellation.cancel()

            with self.assertRaises(RequestCancelledError):
                await service.synthesize(
                    language_code="hi",
                    speaker_name="Divya",
                    text="Namaste",
                    cancellation=cancellation,
                )

            self.assertEqual(service._bundle.model.generate_calls, [])

        asyncio.run(scenario())

    def test_concurrency_guard_rejects_parallel_generation(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._generation_gate = BusyGate()  # type: ignore[assignment]

            with self.assertRaises(ServiceBusyError):
                await service.synthesize(
                    language_code="hi",
                    speaker_name="Divya",
                    text="Second request",
                )

        asyncio.run(scenario())

    def test_loader_failure_is_retained_as_model_error(self) -> None:
        async def scenario() -> None:
            def broken_loader() -> LoadedModelBundle:
                raise OSError("libcudart.so.13: cannot open shared object file")

            service = IndicParlerTTSService(
                catalog=build_default_catalog(),
                model_loader=broken_loader,
            )
            await service.load()

            self.assertFalse(service.is_ready)
            self.assertIsNotNone(service.load_error)
            self.assertIn("torchaudio", service.load_error.lower())

            with self.assertRaises(ModelLoadError):
                await service.synthesize(
                    language_code="hi",
                    speaker_name="Divya",
                    text="Namaste",
                )

        asyncio.run(scenario())

    def test_custom_description_bypasses_catalog_speaker_lookup(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"WAV"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="hi",
                text="Namaste",
                voice_description="custom calm voice with clean studio sound",
            )

            self.assertEqual(wav_bytes, b"WAV")
            self.assertEqual(
                service._bundle.description_tokenizer.last_text,
                "custom calm voice with clean studio sound",
            )

        asyncio.run(scenario())

    def test_generation_gate_reports_global_busy_state(self) -> None:
        gate = SynthesisConcurrencyGate()

        async def scenario() -> None:
            self.assertFalse(gate.is_generating)
            async with gate.claim():
                self.assertTrue(gate.is_generating)
            self.assertFalse(gate.is_generating)

        asyncio.run(scenario())

    def test_clone_service_synthesizes_with_reference_audio(self) -> None:
        async def scenario() -> None:
            preprocessor = FakeReferenceAudioPreprocessor()
            service = XttsVoiceCloneService(
                catalog=build_default_catalog(),
                model_loader=fake_clone_loader,
                reference_audio_preprocessor=preprocessor,  # type: ignore[arg-type]
            )
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"CLONE"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="hi",
                text="Namaste duniya. Kaise ho?",
                reference_audio=b"reference-bytes",
                reference_filename="voice.wav",
            )

            self.assertEqual(wav_bytes, b"CLONE")
            self.assertEqual(preprocessor.calls, [(b"reference-bytes", "voice.wav")])
            self.assertEqual(service._bundle.model.calls[0]["language"], "hi")

        asyncio.run(scenario())

    def test_clone_service_accepts_catalog_language_beyond_hindi_and_english(self) -> None:
        async def scenario() -> None:
            preprocessor = FakeReferenceAudioPreprocessor()
            service = XttsVoiceCloneService(
                catalog=build_default_catalog(),
                model_loader=fake_clone_loader,
                reference_audio_preprocessor=preprocessor,  # type: ignore[arg-type]
                supported_languages=("en", "hi", "bn"),
            )
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"CLONE"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="bn",
                text="Nomoskar",
                reference_audio=b"reference-bytes",
                reference_filename="voice.wav",
            )

            self.assertEqual(wav_bytes, b"CLONE")
            self.assertEqual(preprocessor.calls, [(b"reference-bytes", "voice.wav")])
            self.assertEqual(service._bundle.model.calls[0]["language"], "bn")

        asyncio.run(scenario())

    def test_clone_service_rejects_unsupported_xtts_language(self) -> None:
        async def scenario() -> None:
            preprocessor = FakeReferenceAudioPreprocessor()
            service = XttsVoiceCloneService(
                catalog=build_default_catalog(),
                model_loader=fake_clone_loader,
                reference_audio_preprocessor=preprocessor,  # type: ignore[arg-type]
            )
            await service.load()

            with self.assertRaises(UnsupportedLanguageError):
                await service.synthesize(
                    language_code="gu",
                    text="Kem cho?",
                    reference_audio=b"reference-bytes",
                    reference_filename="voice.wav",
                )

            self.assertEqual(preprocessor.calls, [])

        asyncio.run(scenario())

    def test_clone_service_stops_between_chunks_when_request_is_cancelled(self) -> None:
        cancellation = RequestCancellation()
        model = CancellingCloneModel(cancellation)
        bundle = LoadedVoiceCloneBundle(
            model=model,
            device="cpu",
            sampling_rate=24000,
            model_name="fake-clone-model",
        )
        service = XttsVoiceCloneService(
            catalog=build_default_catalog(),
            model_loader=fake_clone_loader,
        )

        with self.assertRaises(RequestCancelledError):
            service._synthesize_sync(
                bundle,
                ["first chunk", "second chunk"],
                "en",
                "/tmp/reference.wav",
                cancellation,
            )

        self.assertEqual(model.calls, ["first chunk"])
