"""
API → V1 → Router
Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from src.api.v1.endpoints.chat import router as chat_router
from src.api.v1.endpoints.health import router as health_router
from src.api.v1.endpoints.rag import router as rag_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(health_router)
v1_router.include_router(chat_router)
v1_router.include_router(rag_router)
