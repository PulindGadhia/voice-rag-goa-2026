"""FastAPI application factory and lifecycle."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from .config import get_settings
from .orchestrator import Orchestrator
from .api.routes_health import router as health_router
from .api.routes_metrics import router as metrics_router
from .api.routes_query import router as query_router
from .api.routes_voice import router as voice_router


def create_app(*, orchestrator: Orchestrator | None = None):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    settings = get_settings()
    service = orchestrator or Orchestrator(settings=settings)

    @asynccontextmanager
    async def lifespan(app):
        await service.startup()
        app.state.orchestrator = service
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="Voice RAG Goa", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.orchestrator = service
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(voice_router)
    app.include_router(metrics_router)

    # --- Serve frontend static files ---
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.is_dir():
        # Mount src/assets for images/icons referenced by the frontend
        assets_dir = frontend_dir / "src" / "assets"
        if assets_dir.is_dir():
            app.mount("/src/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()

