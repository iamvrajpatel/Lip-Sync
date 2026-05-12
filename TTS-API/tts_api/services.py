import asyncio
import importlib
import io
import logging
import math
import re
import shutil
import tempfile
import threading
import warnings
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from tts_api.catalog import LanguageCatalog, UnsupportedLanguageError


logger = logging.getLogger(__name__)
_GENERATION_MIXIN_WARNING_FRAGMENT = "doesn't directly inherit from `GenerationMixin`"
XTTS_V2_SUPPORTED_LANGUAGE_CODES = ("en", "hi")
_CANCELLED_SYNTHESIS = object()


class _MessageFilter(logging.Filter):
    def __init__(self, blocked_fragment: str) -> None:
        super().__init__()
        self._blocked_fragment = blocked_fragment

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return self._blocked_fragment not in message


def _resolve_optional_class(module_name: str, class_name: str) -> type[Any] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    resolved_class = getattr(module, class_name, None)
    return resolved_class if isinstance(resolved_class, type) else None


def _ensure_generation_mixin_support(
    *model_classes: type[Any] | None,
    generation_mixin_class: type[Any] | None = None,
) -> None:
    if generation_mixin_class is None:
        try:
            from transformers.generation.utils import GenerationMixin
        except Exception:
            return
        generation_mixin_class = GenerationMixin

    for model_class in model_classes:
        if model_class is None or not isinstance(model_class, type):
            continue
        if issubclass(model_class, generation_mixin_class):
            continue

        current_bases = tuple(model_class.__bases__)
        if current_bases == (object,):
            logger.debug(
                "Skipping GenerationMixin compatibility patch for %s because it has no framework base class.",
                model_class,
            )
            continue

        try:
            model_class.__bases__ = current_bases + (generation_mixin_class,)
        except TypeError:
            logger.debug(
                "Skipping GenerationMixin compatibility patch for %s",
                model_class,
                exc_info=True,
            )


@contextmanager
def _suppress_generation_mixin_log_warning() -> Iterator[None]:
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    warning_filter = _MessageFilter(_GENERATION_MIXIN_WARNING_FRAGMENT)
    transformers_logger.addFilter(warning_filter)
    try:
        yield
    finally:
        transformers_logger.removeFilter(warning_filter)


class ModelNotReadyError(RuntimeError):
    """Raised when synthesis is requested before the model is loaded."""


class ServiceBusyError(RuntimeError):
    """Raised when the model is already handling another request."""


class ModelLoadError(RuntimeError):
    """Raised when model startup failed due to dependency or runtime issues."""


class VoiceCloneLoadError(RuntimeError):
    """Raised when the voice clone model failed to load."""


class InvalidReferenceAudioError(ValueError):
    """Raised when the uploaded clone reference audio is not usable."""


class RequestCancelledError(RuntimeError):
    """Raised when speech generation should stop because the client disconnected."""


class RequestCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RequestCancelledError(
                "Speech generation was cancelled because the client disconnected."
            )


@dataclass
class LoadedModelBundle:
    model: Any
    tokenizer: Any
    description_tokenizer: Any
    device: str
    sampling_rate: int
    model_name: str


@dataclass
class LoadedVoiceCloneBundle:
    model: Any
    device: str
    sampling_rate: int
    model_name: str


@dataclass(frozen=True)
class PreparedReferenceAudio:
    path: str
    temp_dir: str


class SynthesisConcurrencyGate:
    def __init__(self, max_concurrent_requests: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._active_requests = 0

    @property
    def is_generating(self) -> bool:
        return self._active_requests > 0

    @asynccontextmanager
    async def claim(self) -> Iterator[None]:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.01)
        except asyncio.TimeoutError as exc:
            raise ServiceBusyError("The server is currently processing another request.") from exc

        try:
            self._active_requests += 1
            yield
        finally:
            self._active_requests -= 1
            self._semaphore.release()


class ReferenceAudioPreprocessor:
    def __init__(
        self,
        target_sample_rate: int = 24000,
        minimum_duration_seconds: float = 5.0,
    ) -> None:
        self._target_sample_rate = target_sample_rate
        self._minimum_duration_seconds = minimum_duration_seconds

    def prepare(self, audio_bytes: bytes, filename: str | None = None) -> PreparedReferenceAudio:
        if not audio_bytes:
            raise InvalidReferenceAudioError("Reference audio file is empty.")

        temp_dir = tempfile.mkdtemp(prefix="tts-clone-")
        raw_suffix = Path(filename or "reference.wav").suffix or ".wav"
        raw_path = Path(temp_dir) / f"reference{raw_suffix.lower()}"
        raw_path.write_bytes(audio_bytes)

        try:
            cleaned_audio = self._normalize_to_wav(audio_bytes)
        except InvalidReferenceAudioError as exc:
            logger.warning("Reference audio cleanup skipped: %s", exc)
            return PreparedReferenceAudio(path=str(raw_path), temp_dir=temp_dir)

        cleaned_path = Path(temp_dir) / "reference-cleaned.wav"
        cleaned_path.write_bytes(cleaned_audio)
        return PreparedReferenceAudio(path=str(cleaned_path), temp_dir=temp_dir)

    def _normalize_to_wav(self, audio_bytes: bytes) -> bytes:
        import soundfile as sf

        try:
            audio_array, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise InvalidReferenceAudioError(
                "Uploaded reference audio could not be decoded for cleanup. "
                "A WAV file is the safest option."
            ) from exc

        normalized_audio = self._coerce_audio_array(audio_array, sample_rate)
        buffer = io.BytesIO()
        sf.write(buffer, normalized_audio, self._target_sample_rate, format="WAV")
        return buffer.getvalue()

    def _coerce_audio_array(self, audio_array: Any, sample_rate: int) -> np.ndarray:
        array = np.asarray(audio_array, dtype=np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)

        if array.size == 0:
            raise InvalidReferenceAudioError("Reference audio has no samples.")

        array = self._resample(array, sample_rate)
        max_amplitude = float(np.max(np.abs(array)))
        if max_amplitude > 0:
            array = np.clip(array / max_amplitude, -1.0, 1.0)

        minimum_samples = int(self._minimum_duration_seconds * self._target_sample_rate)
        if array.size < minimum_samples:
            repeats = math.ceil(minimum_samples / array.size)
            array = np.tile(array, repeats)[:minimum_samples]

        return array.astype(np.float32)

    def _resample(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate <= 0:
            raise InvalidReferenceAudioError("Reference audio sample rate is invalid.")

        if sample_rate == self._target_sample_rate:
            return audio_array

        if audio_array.size == 1:
            return np.repeat(audio_array, self._target_sample_rate).astype(np.float32)

        target_size = max(
            1,
            int(round(audio_array.size * self._target_sample_rate / sample_rate)),
        )
        original_positions = np.linspace(0.0, 1.0, num=audio_array.size, endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        return np.interp(target_positions, original_positions, audio_array).astype(np.float32)


def build_default_model_loader(
    model_name: str = "ai4bharat/indic-parler-tts",
) -> Callable[[], LoadedModelBundle]:
    def load() -> LoadedModelBundle:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )
        return LoadedModelBundle(
            model=model,
            tokenizer=tokenizer,
            description_tokenizer=description_tokenizer,
            device=device,
            sampling_rate=model.config.sampling_rate,
            model_name=model_name,
        )

    return load


def build_default_voice_clone_model_loader(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> Callable[[], LoadedVoiceCloneBundle]:
    def load() -> LoadedVoiceCloneBundle:
        import torch
        from TTS.api import TTS
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig
        from transformers import GPT2Model

        _ensure_generation_mixin_support(
            GPT2Model,
            _resolve_optional_class("TTS.tts.layers.xtts.gpt", "GPT2InferenceModel"),
            _resolve_optional_class("TTS.tts.layers.xtts.gpt_inference", "GPT2InferenceModel"),
        )

        # Torch versions prior to the newer safe-unpickling API do not expose
        # add_safe_globals; those versions also do not require this registration.
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with _suppress_generation_mixin_log_warning(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(f".*{re.escape(_GENERATION_MIXIN_WARNING_FRAGMENT)}.*"),
                category=UserWarning,
            )
            model = TTS(model_name=model_name, progress_bar=False).to(device)
        sample_rate = getattr(getattr(model, "synthesizer", None), "output_sample_rate", 24000)
        return LoadedVoiceCloneBundle(
            model=model,
            device=device,
            sampling_rate=sample_rate,
            model_name=model_name,
        )

    return load


class IndicParlerTTSService:
    def __init__(
        self,
        catalog: LanguageCatalog,
        model_loader: Callable[[], LoadedModelBundle] | None = None,
        generation_gate: SynthesisConcurrencyGate | None = None,
    ) -> None:
        self._catalog = catalog
        self._model_loader = model_loader or build_default_model_loader()
        self._bundle: LoadedModelBundle | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._generation_gate = generation_gate or SynthesisConcurrencyGate()

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    @property
    def is_generating(self) -> bool:
        return self._generation_gate.is_generating

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return self._bundle.device if self._bundle else None

    @property
    def model_name(self) -> str:
        return self._bundle.model_name if self._bundle else "ai4bharat/indic-parler-tts"

    @property
    def generation_gate(self) -> SynthesisConcurrencyGate:
        return self._generation_gate

    async def load(self) -> None:
        if self._bundle is not None or self._load_error is not None:
            return

        async with self._load_lock:
            if self._bundle is not None or self._load_error is not None:
                return

            try:
                self._bundle = self._model_loader()
            except Exception as exc:
                self._load_error = self._build_load_error_message(exc)
                logger.exception("Failed to load Indic Parler TTS model")

    def _build_load_error_message(self, exc: Exception) -> str:
        base_message = f"Model loading failed: {exc}"
        raw_message = str(exc)

        if "libcudart.so.13" in raw_message:
            return (
                f"{base_message}. Detected a Torch/Torchaudio CUDA mismatch. "
                "Reinstall matching wheels for torch==2.7.1 and torchaudio==2.7.1, "
                "using either the CPU index or the cu126 index."
            )

        return base_message

    async def synthesize(
        self,
        language_code: str,
        text: str,
        speaker_name: str | None = None,
        voice_description: str | None = None,
        cancellation: RequestCancellation | None = None,
    ) -> bytes:
        if self._load_error is not None:
            raise ModelLoadError(self._load_error)

        if self._bundle is None:
            raise ModelNotReadyError("The TTS model is still loading.")

        description = self._resolve_description(
            language_code=language_code,
            speaker_name=speaker_name,
            voice_description=voice_description,
        )
        request_cancellation = cancellation or RequestCancellation()

        async with self._generation_gate.claim():
            wav_bytes = await asyncio.to_thread(
                self._safely_synthesize_sync,
                self._bundle,
                text,
                description,
                request_cancellation,
            )
            if wav_bytes is _CANCELLED_SYNTHESIS:
                raise RequestCancelledError(
                    "Speech generation was cancelled because the client disconnected."
                )
            return wav_bytes

    def _resolve_description(
        self,
        language_code: str,
        speaker_name: str | None,
        voice_description: str | None,
    ) -> str:
        if voice_description:
            self._catalog.get_language(language_code)
            return voice_description

        if not speaker_name:
            raise ValueError("Either speaker_name or voice_description must be provided.")

        speaker = self._catalog.get_speaker(language_code, speaker_name)
        return speaker.description

    def _synthesize_sync(
        self,
        bundle: LoadedModelBundle,
        text: str,
        description: str,
        cancellation: RequestCancellation,
    ) -> bytes:
        cancellation.raise_if_cancelled()
        description_inputs = bundle.description_tokenizer(
            description, return_tensors="pt"
        ).to(bundle.device)
        prompt_inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)

        cancellation.raise_if_cancelled()
        generation = bundle.model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

        cancellation.raise_if_cancelled()
        audio_array = self._coerce_audio_array(generation)
        return self._encode_wav(audio_array, bundle.sampling_rate)

    def _safely_synthesize_sync(
        self,
        bundle: LoadedModelBundle,
        text: str,
        description: str,
        cancellation: RequestCancellation,
    ) -> bytes | object:
        try:
            return self._synthesize_sync(bundle, text, description, cancellation)
        except RequestCancelledError:
            return _CANCELLED_SYNTHESIS

    def _coerce_audio_array(self, generation: Any) -> np.ndarray:
        if hasattr(generation, "cpu"):
            generation = generation.cpu()
        if hasattr(generation, "numpy"):
            generation = generation.numpy()

        audio_array = np.asarray(generation, dtype=np.float32).squeeze()
        if audio_array.ndim != 1:
            audio_array = audio_array.reshape(-1)
        return audio_array

    def _encode_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        return buffer.getvalue()


class XttsVoiceCloneService:
    def __init__(
        self,
        catalog: LanguageCatalog,
        model_loader: Callable[[], LoadedVoiceCloneBundle] | None = None,
        reference_audio_preprocessor: ReferenceAudioPreprocessor | None = None,
        generation_gate: SynthesisConcurrencyGate | None = None,
        supported_languages: tuple[str, ...] | None = None,
    ) -> None:
        self._catalog = catalog
        self._model_loader = model_loader or build_default_voice_clone_model_loader()
        self._reference_audio_preprocessor = (
            reference_audio_preprocessor or ReferenceAudioPreprocessor()
        )
        self._generation_gate = generation_gate or SynthesisConcurrencyGate()
        catalog_language_codes = {
            language.code.strip().lower() for language in self._catalog.list_languages()
        }
        resolved_supported_languages = supported_languages or tuple(
            code for code in XTTS_V2_SUPPORTED_LANGUAGE_CODES if code in catalog_language_codes
        )
        self._supported_languages = tuple(
            sorted({code.strip().lower() for code in resolved_supported_languages})
        )
        self._bundle: LoadedVoiceCloneBundle | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    @property
    def is_generating(self) -> bool:
        return self._generation_gate.is_generating

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return self._bundle.device if self._bundle else None

    @property
    def model_name(self) -> str:
        return self._bundle.model_name if self._bundle else "tts_models/multilingual/multi-dataset/xtts_v2"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self._supported_languages

    async def load(self) -> None:
        if self._bundle is not None or self._load_error is not None:
            return

        async with self._load_lock:
            if self._bundle is not None or self._load_error is not None:
                return

            try:
                self._bundle = self._model_loader()
            except Exception as exc:
                self._load_error = self._build_load_error_message(exc)
                logger.exception("Failed to load XTTS voice clone model")

    def _build_load_error_message(self, exc: Exception) -> str:
        return (
            f"Voice clone model loading failed: {exc}. "
            "Install Coqui TTS and matching Torch dependencies to enable cloning."
        )

    async def synthesize(
        self,
        language_code: str,
        text: str,
        reference_audio: bytes,
        reference_filename: str | None = None,
        cancellation: RequestCancellation | None = None,
    ) -> bytes:
        normalized_language = language_code.strip().lower()
        language = self._catalog.get_language(normalized_language)
        self._validate_supported_language(language)
        request_cancellation = cancellation or RequestCancellation()
        request_cancellation.raise_if_cancelled()

        await self.load()
        if self._load_error is not None:
            raise VoiceCloneLoadError(self._load_error)
        if self._bundle is None:
            raise VoiceCloneLoadError("The voice clone model is still loading.")

        prepared_reference = self._reference_audio_preprocessor.prepare(
            reference_audio,
            filename=reference_filename,
        )
        try:
            async with self._generation_gate.claim():
                chunks = self._chunk_text(text, normalized_language)
                wav_bytes = await asyncio.to_thread(
                    self._safely_synthesize_sync,
                    self._bundle,
                    chunks,
                    normalized_language,
                    prepared_reference.path,
                    request_cancellation,
                )
                if wav_bytes is _CANCELLED_SYNTHESIS:
                    raise RequestCancelledError(
                        "Speech generation was cancelled because the client disconnected."
                    )
                return wav_bytes
        finally:
            shutil.rmtree(prepared_reference.temp_dir, ignore_errors=True)

    def _validate_supported_language(self, language: Any) -> None:
        if language.code in self._supported_languages:
            return

        supported_display_names = [
            self._catalog.get_language(code).display_name
            for code in self._supported_languages
        ]
        supported_list = ", ".join(supported_display_names) or "none"
        raise UnsupportedLanguageError(
            f"Voice cloning is not available for {language.display_name}. "
            f"Supported clone languages: {supported_list}."
        )

    def _chunk_text(self, text: str, language_code: str) -> list[str]:
        max_lengths = {"hi": 230, "en": 280}
        max_length = max_lengths.get(language_code, 260)
        normalized_text = re.sub(r"\s+", " ", text).strip()
        if not normalized_text:
            raise ValueError("Text must not be empty.")
        if len(normalized_text) <= max_length:
            return [normalized_text]

        sentence_like_parts = re.split(r"(?<=[.!?।])\s+", normalized_text)
        chunks: list[str] = []
        current_chunk = ""

        for part in sentence_like_parts:
            part = part.strip()
            if not part:
                continue
            current_chunk = self._append_or_flush_chunk(
                chunks,
                current_chunk,
                part,
                max_length,
            )

        if current_chunk:
            chunks.append(current_chunk)

        return chunks or [normalized_text]

    def _append_or_flush_chunk(
        self,
        chunks: list[str],
        current_chunk: str,
        text_part: str,
        max_length: int,
    ) -> str:
        if len(text_part) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            words = text_part.split(" ")
            word_chunk = ""
            for word in words:
                candidate = f"{word_chunk} {word}".strip()
                if len(candidate) <= max_length:
                    word_chunk = candidate
                    continue
                if word_chunk:
                    chunks.append(word_chunk)
                word_chunk = word

            return word_chunk

        candidate = f"{current_chunk} {text_part}".strip()
        if candidate and len(candidate) <= max_length:
            return candidate

        if current_chunk:
            chunks.append(current_chunk)
        return text_part

    def _synthesize_sync(
        self,
        bundle: LoadedVoiceCloneBundle,
        chunks: list[str],
        language_code: str,
        reference_path: str,
        cancellation: RequestCancellation,
    ) -> bytes:
        waves: list[np.ndarray] = []
        for chunk in chunks:
            cancellation.raise_if_cancelled()
            waves.append(
                np.asarray(
                    bundle.model.tts(
                        text=chunk,
                        speaker_wav=reference_path,
                        language=language_code,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            )

        if not waves:
            raise RuntimeError("No audio generated by the voice clone model.")

        cancellation.raise_if_cancelled()
        combined_audio = waves[0]
        silence = np.zeros(max(1, int(bundle.sampling_rate * 0.005)), dtype=np.float32)
        for wave in waves[1:]:
            cancellation.raise_if_cancelled()
            combined_audio = np.concatenate([combined_audio, silence, wave])

        cancellation.raise_if_cancelled()
        return self._encode_wav(combined_audio, bundle.sampling_rate)

    def _safely_synthesize_sync(
        self,
        bundle: LoadedVoiceCloneBundle,
        chunks: list[str],
        language_code: str,
        reference_path: str,
        cancellation: RequestCancellation,
    ) -> bytes | object:
        try:
            return self._synthesize_sync(
                bundle,
                chunks,
                language_code,
                reference_path,
                cancellation,
            )
        except RequestCancelledError:
            return _CANCELLED_SYNTHESIS

    def _encode_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        return buffer.getvalue()
