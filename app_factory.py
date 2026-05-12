from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from tts_api.catalog import LanguageCatalog, build_default_catalog
from tts_api.services import (
    IndicParlerTTSService,
    SynthesisConcurrencyGate,
    XttsVoiceCloneService,
)
from tts_api.web import build_health_payload, build_ui_context, register_tts_routes

if TYPE_CHECKING:
    from lipsync_app import LipSyncService


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def render_index_page(
    request: Request,
    language_catalog: LanguageCatalog,
    voice_clone_service: XttsVoiceCloneService,
) -> HTMLResponse:
    html = templates.get_template("index.html").render(
        request=request,
        **build_ui_context(language_catalog, voice_clone_service),
    )
    return HTMLResponse(content=html)


def create_app(
    service: IndicParlerTTSService | None = None,
    clone_service: XttsVoiceCloneService | None = None,
    catalog: LanguageCatalog | None = None,
    lip_sync_service: "LipSyncService | None" = None,
) -> FastAPI:
    language_catalog = catalog or build_default_catalog()
    generation_gate = SynthesisConcurrencyGate(max_concurrent_requests=1)
    tts_service = service or IndicParlerTTSService(
        catalog=language_catalog,
        generation_gate=generation_gate,
    )
    voice_clone_service = clone_service or XttsVoiceCloneService(
        catalog=language_catalog,
        generation_gate=generation_gate,
    )
    if lip_sync_service is None:
        from lipsync_app import LipSyncService

        lip_sync = LipSyncService()
    else:
        lip_sync = lip_sync_service

    app = FastAPI(title="Lip-Sync + TTS Studio", version="1.0.0")
    app.state.tts_service = tts_service
    app.state.voice_clone_service = voice_clone_service
    app.state.language_catalog = language_catalog
    app.state.lip_sync_service = lip_sync

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "X-Processing-Time", "X-Generation-Time-Ms"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        await tts_service.load()

    @app.get("/")
    async def root(request: Request):
        return render_index_page(request, language_catalog, voice_clone_service)

    @app.get("/ui")
    async def ui_page(request: Request):
        return render_index_page(request, language_catalog, voice_clone_service)

    @app.get("/health")
    async def health() -> JSONResponse:
        payload, status_code = build_health_payload(
            tts_service,
            voice_clone_service,
            lip_sync=lip_sync.get_health_payload(),
        )
        return JSONResponse(payload, status_code=status_code)

    register_tts_routes(app, tts_service, voice_clone_service, language_catalog)
    app.include_router(lip_sync.build_router())
    return app
