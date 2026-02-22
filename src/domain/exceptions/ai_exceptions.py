"""Domain → Exceptions
Domain-specific exceptions — independent of framework/infrastructure.
"""


class DomainError(Exception):
    """Base domain error."""
    def __init__(self, message: str, code: str = "DOMAIN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class AIProviderError(DomainError):
    """Raised when an AI provider call fails."""
    def __init__(self, message: str, provider: str, code: str = "AI_PROVIDER_ERROR"):
        self.provider = provider
        super().__init__(message, code)


class AIProviderNotConfiguredError(AIProviderError):
    """Raised when an AI provider is not configured."""
    def __init__(self, provider: str):
        super().__init__(
            message=f"Provider '{provider}' is not configured. Check your env variables.",
            provider=provider,
            code="AI_PROVIDER_NOT_CONFIGURED",
        )


class RateLimitError(AIProviderError):
    """Raised when the provider rate-limits the request."""
    def __init__(self, provider: str, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(
            message=f"Rate limit exceeded for provider '{provider}'. Retry after {retry_after}s.",
            provider=provider,
            code="RATE_LIMIT_EXCEEDED",
        )


class ConversationNotFoundError(DomainError):
    """Raised when a conversation ID does not exist."""
    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation '{conversation_id}' not found.",
            code="CONVERSATION_NOT_FOUND",
        )


class DocumentNotFoundError(DomainError):
    """Raised when a document ID does not exist."""
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document '{document_id}' not found.",
            code="DOCUMENT_NOT_FOUND",
        )


class EmbeddingError(DomainError):
    """Raised when document embedding fails."""
    def __init__(self, message: str):
        super().__init__(message, code="EMBEDDING_ERROR")
