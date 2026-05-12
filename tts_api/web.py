import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from tts_api.catalog import (
    LanguageCatalog,
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
    build_default_catalog,
)
from tts_api.schemas import TTSRequest, VoiceCloneRequest
from tts_api.services import (
    InvalidReferenceAudioError,
    IndicParlerTTSService,
    ModelLoadError,
    ModelNotReadyError,
    RequestCancellation,
    RequestCancelledError,
    ServiceBusyError,
    VoiceCloneLoadError,
    XttsVoiceCloneService,
)


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
CUSTOM_DESCRIPTION_PLACEHOLDER = (
    "A warm, expressive female voice with a slightly brisk pace, "
    "clear pronunciation, and a clean studio recording with almost no background noise."
)
logger = logging.getLogger(__name__)


def _build_filename(language_code: str, voice_label: str) -> str:
    safe_label = voice_label.strip().replace(" ", "_").lower()
    return f"{language_code}-{safe_label}.wav"


async def _watch_for_disconnect(
    request: Request,
    cancellation: RequestCancellation,
    poll_interval_seconds: float = 0.25,
) -> None:
    while not cancellation.is_cancelled:
        if await request.is_disconnected():
            cancellation.cancel()
            return
        await asyncio.sleep(poll_interval_seconds)


def build_ui_context(
    catalog: LanguageCatalog,
    clone_service: XttsVoiceCloneService,
) -> dict[str, object]:
    return {
        "languages": catalog.as_template_data(),
        "custom_description_placeholder": CUSTOM_DESCRIPTION_PLACEHOLDER,
        "clone_supported_languages": list(clone_service.supported_languages),
    }


def render_index_page(
    request: Request,
    catalog: LanguageCatalog,
    clone_service: XttsVoiceCloneService,
) -> HTMLResponse:
    html = templates.get_template("index.html").render(
        request=request,
        **build_ui_context(catalog, clone_service),
    )
    return HTMLResponse(content=html)


def build_health_payload(
    service: IndicParlerTTSService,
    clone_service: XttsVoiceCloneService,
    lip_sync: dict[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    ready = service.is_ready
    load_error = service.load_error
    if ready:
        status = "ok"
        status_code = 200
    elif load_error:
        status = "error"
        status_code = 503
    else:
        status = "loading"
        status_code = 503

    payload = {
        "status": status,
        "ready": ready,
        "model_loaded": ready,
        "model_name": service.model_name,
        "device": service.device,
        "model_error": load_error,
        "generation_in_progress": service.is_generating,
        "voice_clone": {
            "ready": clone_service.is_ready,
            "model_name": clone_service.model_name,
            "device": clone_service.device,
            "model_error": clone_service.load_error,
            "supported_languages": list(clone_service.supported_languages),
        },
    }
    if lip_sync is not None:
        payload["lip_sync"] = lip_sync
    return payload, status_code


def register_tts_routes(
    app: FastAPI,
    service: IndicParlerTTSService,
    clone_service: XttsVoiceCloneService,
    catalog: LanguageCatalog,
) -> None:
    @app.post("/tts/")
    async def synthesize(request: Request, request_body: TTSRequest) -> Response:
        voice_label = request_body.speaker or "custom-description"
        started_at = time.perf_counter()
        cancellation = RequestCancellation()
        disconnect_task = asyncio.create_task(_watch_for_disconnect(request, cancellation))
        try:
            wav_bytes = await service.synthesize(
                language_code=request_body.language,
                text=request_body.text,
                speaker_name=request_body.speaker,
                voice_description=request_body.voice_description,
                cancellation=cancellation,
            )
        except UnsupportedLanguageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UnsupportedSpeakerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RequestCancelledError as exc:
            logger.info("Client disconnected during TTS synthesis")
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error during TTS synthesis")
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during speech generation.",
            ) from exc
        finally:
            cancellation.cancel()
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task

        generation_time_ms = (time.perf_counter() - started_at) * 1000

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="{_build_filename(request_body.language, voice_label)}"',
                "X-Generation-Time-Ms": f"{generation_time_ms:.0f}",
                "X-Voice-Mode": request_body.voice_mode,
            },
        )

    @app.post("/clone-voice")
    async def clone_voice(
        request: Request,
        text: str = Form(...),
        language: str = Form(...),
        reference_audio: UploadFile = File(...),
    ) -> Response:
        try:
            request_body = VoiceCloneRequest(text=text, language=language)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        started_at = time.perf_counter()
        reference_audio_bytes = await reference_audio.read()
        cancellation = RequestCancellation()
        disconnect_task = asyncio.create_task(_watch_for_disconnect(request, cancellation))
        try:
            wav_bytes = await clone_service.synthesize(
                language_code=request_body.language,
                text=request_body.text,
                reference_audio=reference_audio_bytes,
                reference_filename=reference_audio.filename,
                cancellation=cancellation,
            )
        except UnsupportedLanguageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InvalidReferenceAudioError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VoiceCloneLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RequestCancelledError as exc:
            logger.info("Client disconnected during voice clone synthesis")
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error during voice clone synthesis")
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during voice clone generation.",
            ) from exc
        finally:
            cancellation.cancel()
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task

        generation_time_ms = (time.perf_counter() - started_at) * 1000
        voice_label = Path(reference_audio.filename or "reference").stem or "reference"

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    f'inline; filename="{_build_filename(request_body.language, f"{voice_label}-clone")}"'
                ),
                "X-Generation-Time-Ms": f"{generation_time_ms:.0f}",
                "X-Voice-Mode": "clone",
            },
        )


def create_app(
    service: IndicParlerTTSService | None = None,
    catalog: LanguageCatalog | None = None,
    clone_service: XttsVoiceCloneService | None = None,
) -> FastAPI:
    language_catalog = catalog or build_default_catalog()
    tts_service = service or IndicParlerTTSService(catalog=language_catalog)
    voice_clone_service = clone_service or XttsVoiceCloneService(catalog=language_catalog)

    app = FastAPI(title="AI4Bharat Indic Parler TTS API", version="1.0.0")

    @app.on_event("startup")
    async def startup() -> None:
        await tts_service.load()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return render_index_page(request, language_catalog, voice_clone_service)

    @app.get("/health")
    async def health() -> JSONResponse:
        payload, status_code = build_health_payload(tts_service, voice_clone_service)
        return JSONResponse(payload, status_code=status_code)

    register_tts_routes(app, tts_service, voice_clone_service, language_catalog)
    return app
