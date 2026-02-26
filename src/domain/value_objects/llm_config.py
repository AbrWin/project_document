"""Domain → Value Objects
Immutable value objects used across the domain.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM call configuration."""
    temperature: float = 0.1
    max_tokens: int = 4096
    streaming: bool = True
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAG retrieval."""
    top_k: int = 5
    similarity_threshold: float = 0.7
    collection_name: Optional[str] = None  # None = no filter, searches all documents
    rerank: bool = False


@dataclass(frozen=True)
class AIProvider:
    """Identifies which AI provider to use."""
    name: str       # azure_openai | azure_inference | openai
    model: str      # deployment name or model id
