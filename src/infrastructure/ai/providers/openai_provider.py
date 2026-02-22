from __future__ import annotations

"""
Infrastructure → AI → Providers → OpenAI Direct Adapter
Uses LangChain's ChatOpenAI for direct OpenAI API access.
"""

from collections.abc import AsyncGenerator
from typing import Optional

import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces.ai_provider import AIProviderPort
from src.domain.entities.message import Message, MessageRole
from src.domain.exceptions.ai_exceptions import AIProviderError, AIProviderNotConfiguredError
from src.domain.value_objects.llm_config import LLMConfig
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger()


class OpenAIProvider(AIProviderPort):
    """Adapter for direct OpenAI API (gpt-4o, gpt-4-turbo, etc.)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.OPENAI_API_KEY:
            raise AIProviderNotConfiguredError("openai")
        self._settings = settings

    def _get_llm(self, config: LLMConfig, streaming: bool = False) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.OPENAI_MODEL,
            api_key=self._settings.OPENAI_API_KEY,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=streaming,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        config: Optional[LLMConfig] = None,
    ) -> Message:
        cfg = config or LLMConfig()
        llm = self._get_llm(cfg)
        lc_messages = [m.to_langchain_dict() for m in messages]

        try:
            response = await llm.ainvoke(lc_messages)
            return Message(role=MessageRole.ASSISTANT, content=response.content)
        except Exception as exc:
            logger.error("openai.chat.error", error=str(exc))
            raise AIProviderError(str(exc), provider="openai") from exc

    async def chat_stream(
        self,
        messages: list[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LLMConfig()
        llm = self._get_llm(cfg, streaming=True)
        lc_messages = [m.to_langchain_dict() for m in messages]

        try:
            async for chunk in llm.astream(lc_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error("openai.stream.error", error=str(exc))
            raise AIProviderError(str(exc), provider="openai") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embed_model = OpenAIEmbeddings(
            model=self._settings.OPENAI_EMBEDDING_MODEL,
            api_key=self._settings.OPENAI_API_KEY,
        )
        try:
            return await embed_model.aembed_documents(texts)
        except Exception as exc:
            logger.error("openai.embed.error", error=str(exc))
            raise AIProviderError(str(exc), provider="openai") from exc
