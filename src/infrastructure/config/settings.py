"""
Infrastructure → Config → Settings
Centralized configuration using Pydantic Settings v2.
All env vars are validated and typed at startup.
"""

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AIProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    AZURE_INFERENCE = "azure_inference"
    OPENAI = "openai"


class VectorStoreProvider(str, Enum):
    PGVECTOR = "pgvector"
    FAISS = "faiss"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ App
    APP_NAME: str = "ProyectIA Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: AppEnv = AppEnv.DEVELOPMENT
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ------------------------------------------------------------------ CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ------------------------------------------------------------------ Security
    SECRET_KEY: str = Field(min_length=32)
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = []

    # ------------------------------------------------------------------ Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"

    # ------------------------------------------------------------------ Azure AI Inference
    AZURE_AI_INFERENCE_ENDPOINT: Optional[str] = None
    AZURE_AI_INFERENCE_KEY: Optional[str] = None
    AZURE_AI_INFERENCE_MODEL_NAME: str = "phi-3-medium-128k-instruct"

    # ------------------------------------------------------------------ OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    # ------------------------------------------------------------------ Default Provider
    DEFAULT_AI_PROVIDER: AIProvider = AIProvider.AZURE_OPENAI

    # ------------------------------------------------------------------ LLM Defaults
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=4096, ge=1)
    LLM_STREAMING: bool = True

    # ------------------------------------------------------------------ Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/proyectia"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ------------------------------------------------------------------ Vector Store
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.PGVECTOR
    VECTOR_COLLECTION_NAME: str = "documents"
    EMBEDDING_DIMENSION: int = 3072

    # ------------------------------------------------------------------ Validators
    @field_validator("AZURE_OPENAI_ENDPOINT", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: Optional[str]) -> Optional[str]:
        return v.rstrip("/") if v else v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == AppEnv.PRODUCTION

    @property
    def database_url_sync(self) -> str:
        """Sync URL for Alembic migrations."""
        return self.DATABASE_URL.replace("+asyncpg", "+psycopg2")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — import this everywhere."""
    return Settings()
