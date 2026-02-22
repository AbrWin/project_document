from __future__ import annotations

"""
Core → Interfaces → AI Provider Port
Abstract interface that ALL AI providers must implement.
Follows the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional

from src.domain.entities.message import Message
from src.domain.value_objects.llm_config import LLMConfig


class AIProviderPort(ABC):
    """
    Port (interface) for LLM providers.
    Infrastructure adapters (Azure OpenAI, Azure Inference, OpenAI)
    implement this contract.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        config: Optional[LLMConfig] = None,
    ) -> Message:
        """Send messages and get a single complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as they arrive (SSE)."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the identifier name for this provider."""
        ...
