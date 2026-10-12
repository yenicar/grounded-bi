from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .auth import get_auth_settings, load_user_from_request
from .dependencies import build_services
from .routers import (
    auth as auth_router,
    corpora,
    documents,
    eval as eval_router,
    export as export_router,
    query,
    search,
)


def _cors_origins() -> list[str]:
    default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    configured = os.getenv("TRBI_WEB_ORIGINS", "")
    if not configured.strip():
        return default_origins

    extra_origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    # Preserve local development defaults while allowing a production origin to be added via env var.
    return list(dict.fromkeys([*default_origins, *extra_origins]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services()
    try:
        yield
    finally:
        app.state.services.scheduler.shutdown()


app = FastAPI(
    title="GroundedBI API",
    version="0.1.0",
    summary="API wrapper for the local-first Business Intelligence RAG platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=get_auth_settings().secret_key,
    same_site="lax",
    https_only=get_auth_settings().session_cookie_secure,
)


@app.middleware("http")
async def attach_current_user(request, call_next):
    load_user_from_request(request)
    return await call_next(request)

app.include_router(auth_router.router)
app.include_router(corpora.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(export_router.router)
app.include_router(search.router)
app.include_router(eval_router.router)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
