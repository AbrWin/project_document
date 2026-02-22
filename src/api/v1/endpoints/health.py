"""
API → V1 → Endpoints → Health
Liveness and readiness probes.
"""

from fastapi import APIRouter

from src.api.v1.schemas import HealthResponse
from src.infrastructure.config.settings import get_settings
from src.infrastructure.container import SettingsDep

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Health check")
async def health(settings: SettingsDep):
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV.value,
        provider=settings.DEFAULT_AI_PROVIDER.value,
    )
