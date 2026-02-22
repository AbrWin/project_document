from __future__ import annotations

"""
Main — FastAPI Application Entry Point
Bootstraps the app: middleware, routes, lifecycle, error handlers.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.auth import APIKeyMiddleware
from src.api.v1.router import v1_router
from src.domain.exceptions.ai_exceptions import (
    AIProviderError,
    ConversationNotFoundError,
    DomainError,
    RateLimitError,
)
from src.infrastructure.config.settings import get_settings
from src.infrastructure.db.database import Base, engine

logger = structlog.get_logger()
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app.startup",
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.APP_ENV.value,
        provider=settings.DEFAULT_AI_PROVIDER.value,
    )

    # Auto-create tables in development (use Alembic migrations in production)
    if settings.APP_ENV.value == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("app.db.tables_created")

    yield

    logger.info("app.shutdown")
    await engine.dispose()


# ──────────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI Backend with Clean Architecture. "
            "Supports Azure OpenAI, Azure AI Inference, and OpenAI with RAG."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost first) ──────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)

    # ── Root endpoint ────────────────────────────────────────────────────
    @app.get("/", tags=["Status"], summary="Root")
    async def root():
        return {
            "status": "🟢 activo",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV.value,
            "docs": "/docs",
            "provider": settings.DEFAULT_AI_PROVIDER.value,
        }

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(v1_router)

    # ── Global Exception Handlers ─────────────────────────────────────────
    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(request: Request, exc: ConversationNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"code": exc.code, "message": exc.message},
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"code": exc.code, "message": exc.message},
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(AIProviderError)
    async def ai_provider_handler(request: Request, exc: AIProviderError):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"code": exc.code, "message": exc.message, "provider": exc.provider},
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"code": exc.code, "message": exc.message},
        )

    return app


app = create_app()
