from __future__ import annotations

"""
Core → Use Cases → Chat Use Case
Orchestrates: conversation history → (optional RAG) → LLM → save + return.
"""

from collections.abc import AsyncGenerator
from typing import Optional
from uuid import UUID

import structlog

from src.core.interfaces.ai_provider import AIProviderPort
from src.core.interfaces.repositories import ConversationRepositoryPort, VectorStorePort
from src.domain.entities.message import Conversation, Message, MessageRole
from src.domain.exceptions.ai_exceptions import ConversationNotFoundError
from src.domain.value_objects.llm_config import LLMConfig, RAGConfig

logger = structlog.get_logger()


class ChatUseCase:
    """
    Application use case: manages chat flow with optional RAG context injection.
    Depends only on abstract ports — no direct infrastructure imports.
    """

    def __init__(
        self,
        ai_provider: AIProviderPort,
        conversation_repo: ConversationRepositoryPort,
        vector_store: Optional[VectorStorePort] = None,
    ) -> None:
        self._ai = ai_provider
        self._repo = conversation_repo
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    async def start_conversation(self, title: Optional[str] = None) -> Conversation:
        """Create and persist a new conversation."""
        conversation = Conversation(title=title)
        return await self._repo.save(conversation)

    # ------------------------------------------------------------------
    async def send_message(
        self,
        conversation_id: UUID,
        user_content: str,
        llm_config: Optional[LLMConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        use_rag: bool = False,
    ) -> Message:
        """
        Non-streaming: send a user message, get full assistant response.
        1. Load conversation history
        2. (Optional) Inject RAG context
        3. Call LLM
        4. Persist and return assistant message
        """
        conversation = await self._get_conversation(conversation_id)
        user_message = Message(role=MessageRole.USER, content=user_content)
        conversation.add_message(user_message)
        await self._repo.add_message(user_message)

        messages = await self._build_messages(conversation, user_content, use_rag, rag_config)

        logger.info(
            "chat.send_message",
            conversation_id=str(conversation_id),
            provider=self._ai.provider_name,
            use_rag=use_rag,
        )

        assistant_message = await self._ai.chat(messages, config=llm_config)
        assistant_message.conversation_id = conversation_id
        conversation.add_message(assistant_message)
        await self._repo.add_message(assistant_message)

        return assistant_message

    # ------------------------------------------------------------------
    async def send_message_stream(
        self,
        conversation_id: UUID,
        user_content: str,
        llm_config: Optional[LLMConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        use_rag: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming: yields token chunks as they arrive via SSE.
        Persists the full assembled response after stream completes.
        """
        conversation = await self._get_conversation(conversation_id)
        user_message = Message(role=MessageRole.USER, content=user_content)
        conversation.add_message(user_message)
        await self._repo.add_message(user_message)

        messages = await self._build_messages(conversation, user_content, use_rag, rag_config)

        logger.info(
            "chat.stream_message",
            conversation_id=str(conversation_id),
            provider=self._ai.provider_name,
        )

        full_response = ""
        async for token in self._ai.chat_stream(messages, config=llm_config):
            full_response += token
            yield token

        # Persist assistant response after stream finishes
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=full_response,
            conversation_id=conversation_id,
        )
        conversation.add_message(assistant_message)
        await self._repo.add_message(assistant_message)

    # ------------------------------------------------------------------
    async def get_conversation(self, conversation_id: UUID) -> Conversation:
        return await self._get_conversation(conversation_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    async def _get_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = await self._repo.get_by_id(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(str(conversation_id))
        return conversation

    async def _build_messages(
        self,
        conversation: Conversation,
        user_content: str,
        use_rag: bool,
        rag_config: Optional[RAGConfig],
    ) -> list[Message]:
        """Build the final message list, optionally injecting RAG context as system message."""
        history = conversation.get_last_n_messages(20)  # Keep last 20 for context window

        if use_rag and self._vector_store:
            cfg = rag_config or RAGConfig()
            chunks = await self._vector_store.similarity_search(
                query=user_content,
                top_k=cfg.top_k,
                score_threshold=cfg.similarity_threshold,
                collection=cfg.collection_name,
            )
            logger.info(
                "chat.rag_context",
                chunks_found=len(chunks),
                top_k=cfg.top_k,
                score_threshold=cfg.similarity_threshold,
            )
            if chunks:
                context = "\n\n---\n\n".join(c.content for c in chunks)
                rag_system = Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Use the following context to answer the user's question. "
                        "If the answer is not in the context, say so.\n\n"
                        f"CONTEXT:\n{context}"
                    ),
                )
                return [rag_system] + history

        return history
