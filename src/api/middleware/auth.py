"""
API → Middleware → API Key Auth
Simple header-based API key authentication middleware.
Skips auth for /health and /docs endpoints.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.infrastructure.config.settings import get_settings

_OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # Skip auth for open endpoints and non-production
        if not settings.is_production or request.url.path in _OPEN_PATHS:
            return await call_next(request)

        if not settings.API_KEYS:
            return await call_next(request)

        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key or api_key not in settings.API_KEYS:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"code": "UNAUTHORIZED", "message": "Invalid or missing API key."},
            )

        return await call_next(request)
