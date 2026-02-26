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

    # Reasoning models (o-series, gpt-5-*) do not support custom temperature/max_tokens params.
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def _is_reasoning_model(self) -> bool:
        name = self._settings.AZURE_OPENAI_DEPLOYMENT_NAME.lower()
        return any(name.startswith(p) for p in self._REASONING_MODEL_PREFIXES)

    def _get_llm(self, config: LLMConfig, streaming: bool = False) -> AzureChatOpenAI:
        cache_key = f"{config.temperature}_{config.max_tokens}_{streaming}"
        if cache_key not in self._llm_cache:
            kwargs: dict = dict(
                azure_deployment=self._settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                azure_endpoint=self._settings.AZURE_OPENAI_ENDPOINT,
                api_key=self._settings.AZURE_OPENAI_API_KEY,
                api_version=self._settings.AZURE_OPENAI_API_VERSION,
                streaming=streaming,
            )
            if self._is_reasoning_model():
                kwargs["temperature"] = 1
                # Reasoning models spend tokens on internal thinking before producing output.
                # Enforce a minimum budget so the model has tokens left for the visible response.
                reasoning_budget = max(config.max_tokens, 16000)
                kwargs["max_completion_tokens"] = reasoning_budget
            else:
                kwargs["temperature"] = config.temperature
                kwargs["max_tokens"] = config.max_tokens
            self._llm_cache[cache_key] = AzureChatOpenAI(**kwargs)
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
            logger.debug(
                "azure_openai.chat.raw_response",
                content=response.content,
                content_type=type(response.content).__name__,
                response_metadata=response.response_metadata,
                additional_kwargs=response.additional_kwargs,
            )
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            return Message(role=MessageRole.ASSISTANT, content=content)
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
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if content:
                    yield content
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
