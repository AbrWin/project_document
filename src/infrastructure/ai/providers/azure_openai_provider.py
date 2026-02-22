from __future__ import annotations

"""
Infrastructure → AI → Providers → Azure OpenAI Adapter
Implements AIProviderPort using LangChain's AzureChatOpenAI.
"""

from collections.abc import AsyncGenerator
from typing import Optional

import structlog
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces.ai_provider import AIProviderPort
from src.domain.entities.message import Message, MessageRole
from src.domain.exceptions.ai_exceptions import AIProviderError, AIProviderNotConfiguredError
from src.domain.value_objects.llm_config import LLMConfig
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger()


class AzureOpenAIProvider(AIProviderPort):
    """
    Adapter for Azure OpenAI Service.
    Supports chat, streaming, and Ada/text-embedding-3 embeddings.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise AIProviderNotConfiguredError("azure_openai")
        self._settings = settings
        self._llm_cache: dict[str, AzureChatOpenAI] = {}

    def _get_llm(self, config: LLMConfig, streaming: bool = False) -> AzureChatOpenAI:
        cache_key = f"{config.temperature}_{config.max_tokens}_{streaming}"
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = AzureChatOpenAI(
                azure_deployment=self._settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                azure_endpoint=self._settings.AZURE_OPENAI_ENDPOINT,
                api_key=self._settings.AZURE_OPENAI_API_KEY,
                api_version=self._settings.AZURE_OPENAI_API_VERSION,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                streaming=streaming,
            )
        return self._llm_cache[cache_key]

    @property
    def provider_name(self) -> str:
        return "azure_openai"

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
        llm = self._get_llm(cfg, streaming=False)
        lc_messages = [m.to_langchain_dict() for m in messages]

        try:
            response = await llm.ainvoke(lc_messages)
            return Message(role=MessageRole.ASSISTANT, content=response.content)
        except Exception as exc:
            logger.error("azure_openai.chat.error", error=str(exc))
            raise AIProviderError(str(exc), provider="azure_openai") from exc

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
            logger.error("azure_openai.stream.error", error=str(exc))
            raise AIProviderError(str(exc), provider="azure_openai") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=self._settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=self._settings.AZURE_OPENAI_ENDPOINT,
            api_key=self._settings.AZURE_OPENAI_API_KEY,
            api_version=self._settings.AZURE_OPENAI_API_VERSION,
        )
        try:
            return await embeddings_model.aembed_documents(texts)
        except Exception as exc:
            logger.error("azure_openai.embed.error", error=str(exc))
            raise AIProviderError(str(exc), provider="azure_openai") from exc
