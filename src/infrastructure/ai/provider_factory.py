from __future__ import annotations

"""
Infrastructure → AI → Provider Factory
Creates the correct AI provider adapter based on configuration.
Follows the Factory pattern to keep provider selection in one place.
"""

import structlog

from src.core.interfaces.ai_provider import AIProviderPort
from src.infrastructure.config.settings import AIProvider, Settings
from src.infrastructure.ai.providers.azure_openai_provider import AzureOpenAIProvider
from src.infrastructure.ai.providers.azure_inference_provider import AzureInferenceProvider
from src.infrastructure.ai.providers.openai_provider import OpenAIProvider

logger = structlog.get_logger()

_PROVIDER_MAP = {
    AIProvider.AZURE_OPENAI: AzureOpenAIProvider,
    AIProvider.AZURE_INFERENCE: AzureInferenceProvider,
    AIProvider.OPENAI: OpenAIProvider,
}


def create_ai_provider(
    settings: Settings,
    provider: AIProvider | None = None,
) -> AIProviderPort:
    """
    Instantiate the appropriate AI provider adapter.
    Falls back to DEFAULT_AI_PROVIDER from settings if none specified.
    """
    target = provider or settings.DEFAULT_AI_PROVIDER
    provider_class = _PROVIDER_MAP.get(target)

    if provider_class is None:
        raise ValueError(f"Unknown AI provider: {target}")

    logger.info("ai_provider.create", provider=target.value)
    return provider_class(settings)
