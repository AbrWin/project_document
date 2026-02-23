from __future__ import annotations

"""
Core → Interfaces → Repository Ports
Abstract interfaces for data persistence.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from src.domain.entities.document import DocumentChunk
from src.domain.entities.message import Conversation, Message


class ConversationRepositoryPort(ABC):
    """Port for conversation persistence."""

    @abstractmethod
    async def save(self, conversation: Conversation) -> Conversation:
        ...

    @abstractmethod
    async def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]:
        ...

    @abstractmethod
    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Conversation]:
        ...

    @abstractmethod
    async def delete(self, conversation_id: UUID) -> bool:
        ...

    @abstractmethod
    async def add_message(self, message: Message) -> Message:
        ...


class VectorStorePort(ABC):
    """Port for vector store (RAG) operations."""

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        collection: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Return most relevant chunks for query."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        ...
