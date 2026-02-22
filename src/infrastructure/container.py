from __future__ import annotations

"""
Infrastructure → Container
Dependency Injection container — wires all adapters to use cases.
FastAPI endpoints import from here via Depends().
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.use_cases.chat_use_case import ChatUseCase
from src.core.use_cases.rag_use_case import RAGUseCase
from src.infrastructure.ai.provider_factory import create_ai_provider
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.db.database import get_db_session
from src.infrastructure.db.repositories.conversation_repository import SQLConversationRepository
from src.infrastructure.db.repositories.pgvector_repository import PGVectorRepository

# --------------------------------------------------------------------------
# Settings dependency
# --------------------------------------------------------------------------
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --------------------------------------------------------------------------
# AI Provider dependency
# --------------------------------------------------------------------------
def get_ai_provider(settings: SettingsDep):
    return create_ai_provider(settings)


AIProviderDep = Annotated[object, Depends(get_ai_provider)]


# --------------------------------------------------------------------------
# Repository dependencies
# --------------------------------------------------------------------------
def get_conversation_repo(
    session: Annotated[AsyncSession, Depends(get_db_session)]
) -> SQLConversationRepository:
    return SQLConversationRepository(session)


def get_vector_store(settings: SettingsDep) -> PGVectorRepository:
    from src.infrastructure.ai.provider_factory import create_ai_provider
    embedding_provider = create_ai_provider(settings)
    return PGVectorRepository(settings, embedding_provider)


# --------------------------------------------------------------------------
# Use Case dependencies
# --------------------------------------------------------------------------
def get_chat_use_case(
    settings: SettingsDep,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ChatUseCase:
    ai_provider = create_ai_provider(settings)
    conversation_repo = SQLConversationRepository(session)
    vector_store = PGVectorRepository(settings, ai_provider)
    return ChatUseCase(
        ai_provider=ai_provider,
        conversation_repo=conversation_repo,
        vector_store=vector_store,
    )


ChatUseCaseDep = Annotated[ChatUseCase, Depends(get_chat_use_case)]


def get_rag_use_case(settings: SettingsDep) -> RAGUseCase:
    ai_provider = create_ai_provider(settings)
    vector_store = PGVectorRepository(settings, ai_provider)
    return RAGUseCase(vector_store=vector_store)


RAGUseCaseDep = Annotated[RAGUseCase, Depends(get_rag_use_case)]
