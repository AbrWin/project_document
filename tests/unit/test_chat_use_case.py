"""
Unit Tests → Chat Use Case
Tests business logic in isolation using mocks (no real LLM/DB needed).
"""

from collections.abc import AsyncGenerator
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.interfaces.ai_provider import AIProviderPort
from src.core.interfaces.repositories import ConversationRepositoryPort
from src.core.use_cases.chat_use_case import ChatUseCase
from src.domain.entities.message import Conversation, Message, MessageRole
from src.domain.exceptions.ai_exceptions import ConversationNotFoundError
from src.domain.value_objects.llm_config import LLMConfig


# ──────────────────────────────────────────────────────────────────────────
# Fakes / Mocks
# ──────────────────────────────────────────────────────────────────────────

class FakeAIProvider(AIProviderPort):
    def __init__(self, response: str = "Fake AI response"):
        self._response = response

    @property
    def provider_name(self) -> str:
        return "fake"

    async def chat(self, messages, config=None) -> Message:
        return Message(role=MessageRole.ASSISTANT, content=self._response)

    async def chat_stream(self, messages, config=None) -> AsyncGenerator[str, None]:
        for word in self._response.split():
            yield word + " "

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeConversationRepo(ConversationRepositoryPort):
    def __init__(self):
        self._store: dict = {}

    async def save(self, conversation: Conversation) -> Conversation:
        self._store[conversation.id] = conversation
        return conversation

    async def get_by_id(self, conversation_id) -> Optional[Conversation]:
        return self._store.get(conversation_id)

    async def list_all(self, limit=20, offset=0) -> list[Conversation]:
        return list(self._store.values())

    async def delete(self, conversation_id) -> bool:
        return self._store.pop(conversation_id, None) is not None

    async def add_message(self, message: Message) -> Message:
        return message


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def chat_use_case():
    return ChatUseCase(
        ai_provider=FakeAIProvider(),
        conversation_repo=FakeConversationRepo(),
    )


@pytest.mark.asyncio
async def test_start_conversation_creates_with_title(chat_use_case):
    conv = await chat_use_case.start_conversation(title="Test Chat")
    assert conv.id is not None
    assert conv.title == "Test Chat"


@pytest.mark.asyncio
async def test_send_message_returns_assistant_response(chat_use_case):
    conv = await chat_use_case.start_conversation()
    msg = await chat_use_case.send_message(conv.id, "Hello!")
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content == "Fake AI response"


@pytest.mark.asyncio
async def test_send_message_unknown_conversation_raises(chat_use_case):
    with pytest.raises(ConversationNotFoundError):
        await chat_use_case.send_message(uuid4(), "Hello!")


@pytest.mark.asyncio
async def test_send_message_stream_yields_tokens(chat_use_case):
    conv = await chat_use_case.start_conversation()
    tokens = []
    async for token in chat_use_case.send_message_stream(conv.id, "Hello!"):
        tokens.append(token)
    assert len(tokens) > 0
    assert "".join(tokens).strip() == "Fake AI response"
