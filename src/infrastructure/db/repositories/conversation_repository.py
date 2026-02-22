from __future__ import annotations

"""
Infrastructure → DB → Repositories → Conversation Repository
Implements ConversationRepositoryPort using SQLAlchemy + PostgreSQL.
"""

from typing import Optional
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.interfaces.repositories import ConversationRepositoryPort
from src.domain.entities.message import Conversation, Message, MessageRole
from src.domain.exceptions.ai_exceptions import ConversationNotFoundError
from src.infrastructure.db.models.conversation_model import ConversationModel, MessageModel

logger = structlog.get_logger()


def _model_to_message(m: MessageModel) -> Message:
    return Message(
        id=UUID(m.id),
        role=MessageRole(m.role),
        content=m.content,
        conversation_id=UUID(m.conversation_id),
        metadata=m.metadata_ or {},
    )


def _model_to_conversation(c: ConversationModel) -> Conversation:
    return Conversation(
        id=UUID(c.id),
        title=c.title,
        created_at=c.created_at,
        updated_at=c.updated_at,
        messages=[_model_to_message(m) for m in c.messages],
        metadata=c.metadata_ or {},
    )


class SQLConversationRepository(ConversationRepositoryPort):
    """PostgreSQL-backed conversation repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, conversation: Conversation) -> Conversation:
        model = ConversationModel(
            id=str(conversation.id),
            title=conversation.title,
            metadata_=conversation.metadata,
        )
        self._session.add(model)
        await self._session.flush()
        logger.info("conversation.saved", id=str(conversation.id))
        return conversation

    async def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]:
        stmt = (
            select(ConversationModel)
            .where(ConversationModel.id == str(conversation_id))
            .options(selectinload(ConversationModel.messages))
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return _model_to_conversation(model) if model else None

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Conversation]:
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .order_by(ConversationModel.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return [_model_to_conversation(row) for row in result.scalars().all()]

    async def delete(self, conversation_id: UUID) -> bool:
        stmt = select(ConversationModel).where(ConversationModel.id == str(conversation_id))
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if not model:
            return False
        await self._session.delete(model)
        return True

    async def add_message(self, message: Message) -> Message:
        model = MessageModel(
            id=str(message.id),
            conversation_id=str(message.conversation_id),
            role=message.role.value,
            content=message.content,
            metadata_=message.metadata,
        )
        self._session.add(model)
        await self._session.flush()
        return message
