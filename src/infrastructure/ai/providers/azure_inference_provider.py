from __future__ import annotations

"""
Infrastructure → AI → Providers → Azure AI Inference Adapter
For models hosted in Azure AI Studio: Phi-3, Mistral, Llama-3, etc.
Uses the azure-ai-inference SDK directly (not LangChain, as langchain support is partial).
"""

from collections.abc import AsyncGenerator
from typing import Optional

import structlog
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.interfaces.ai_provider import AIProviderPort
from src.domain.entities.message import Message, MessageRole
from src.domain.exceptions.ai_exceptions import AIProviderError, AIProviderNotConfiguredError
from src.domain.value_objects.llm_config import LLMConfig
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger()

_ROLE_MAP = {
    MessageRole.SYSTEM: SystemMessage,
    MessageRole.USER: UserMessage,
    MessageRole.ASSISTANT: AssistantMessage,
}


def _to_azure_message(message: Message):
    cls = _ROLE_MAP.get(message.role, UserMessage)
    return cls(content=message.content)


class AzureInferenceProvider(AIProviderPort):
    """
    Adapter for Azure AI Inference endpoint.
    Works with serverless + managed compute models in Azure AI Studio.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.AZURE_AI_INFERENCE_ENDPOINT or not settings.AZURE_AI_INFERENCE_KEY:
            raise AIProviderNotConfiguredError("azure_inference")
        self._settings = settings
        self._client = ChatCompletionsClient(
            endpoint=settings.AZURE_AI_INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_AI_INFERENCE_KEY),
        )

    @property
    def provider_name(self) -> str:
        return "azure_inference"

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
        azure_messages = [_to_azure_message(m) for m in messages]

        try:
            response = await self._client.complete(
                messages=azure_messages,
                model=self._settings.AZURE_AI_INFERENCE_MODEL_NAME,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            content = response.choices[0].message.content
            return Message(role=MessageRole.ASSISTANT, content=content)
        except Exception as exc:
            logger.error("azure_inference.chat.error", error=str(exc))
            raise AIProviderError(str(exc), provider="azure_inference") from exc

    async def chat_stream(
        self,
        messages: list[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        cfg = config or LLMConfig()
        azure_messages = [_to_azure_message(m) for m in messages]

        try:
            stream = await self._client.complete(
                messages=azure_messages,
                model=self._settings.AZURE_AI_INFERENCE_MODEL_NAME,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
            )
            async for update in stream:
                if update.choices and update.choices[0].delta.content:
                    yield update.choices[0].delta.content
        except Exception as exc:
            logger.error("azure_inference.stream.error", error=str(exc))
            raise AIProviderError(str(exc), provider="azure_inference") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Azure AI Inference does not expose a universal embeddings API.
        Fall back to raising — use AzureOpenAI or OpenAI for embeddings.
        """
        raise NotImplementedError(
            "Azure AI Inference does not support embeddings. "
            "Use AzureOpenAIProvider or OpenAIProvider for embeddings."
        )
