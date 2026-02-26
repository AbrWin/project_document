from __future__ import annotations

"""
API → V1 → Endpoints → Chat
Handles conversation management and LLM chat with streaming support.
"""

import json
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.v1.schemas import (
    ConversationDetailResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    QuickChatRequest,
    QuickChatResponse,
    SendMessageRequest,
)
from src.domain.exceptions.ai_exceptions import (
    AIProviderError,
    ConversationNotFoundError,
    RateLimitError,
)
from src.domain.value_objects.llm_config import LLMConfig, RAGConfig
from src.infrastructure.config.settings import AIProvider
from src.infrastructure.container import ChatUseCaseDep

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────────

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    body: CreateConversationRequest,
    use_case: ChatUseCaseDep,
):
    conversation = await use_case.start_conversation(title=body.title)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation with full message history",
)
async def get_conversation(
    conversation_id: UUID,
    use_case: ChatUseCaseDep,
):
    try:
        conv = await use_case.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message)

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role.value,
                content=m.content,
                conversation_id=m.conversation_id,
                created_at=m.created_at,
            )
            for m in conv.messages
        ],
    )


# ──────────────────────────────────────────────────────────────────────────
# Quick chat — creates conversation + sends message in one call
# ──────────────────────────────────────────────────────────────────────────

@router.post(
    "/messages",
    response_model=QuickChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create conversation and send first message in one call",
)
async def quick_chat(
    body: QuickChatRequest,
    use_case: ChatUseCaseDep,
):
    conversation = await use_case.start_conversation(title=body.title)
    llm_config = _build_llm_config(body)
    rag_config = RAGConfig(top_k=body.rag_top_k, similarity_threshold=body.rag_score_threshold)

    try:
        message = await use_case.send_message(
            conversation_id=conversation.id,
            user_content=body.content,
            llm_config=llm_config,
            rag_config=rag_config,
            use_rag=body.use_rag,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        )
    except AIProviderError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    return QuickChatResponse(
        conversation_id=conversation.id,
        message=MessageResponse(
            id=message.id,
            role=message.role.value,
            content=message.content,
            conversation_id=message.conversation_id,
            created_at=message.created_at,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# Non-streaming chat
# ──────────────────────────────────────────────────────────────────────────

@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    summary="Send a message (blocking, full response)",
)
async def send_message(
    conversation_id: UUID,
    body: SendMessageRequest,
    use_case: ChatUseCaseDep,
):
    llm_config = _build_llm_config(body)
    rag_config = RAGConfig(top_k=body.rag_top_k, similarity_threshold=body.rag_score_threshold)

    try:
        message = await use_case.send_message(
            conversation_id=conversation_id,
            user_content=body.content,
            llm_config=llm_config,
            rag_config=rag_config,
            use_rag=body.use_rag,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message)
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        )
    except AIProviderError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    return MessageResponse(
        id=message.id,
        role=message.role.value,
        content=message.content,
        conversation_id=message.conversation_id,
        created_at=message.created_at,
    )


# ──────────────────────────────────────────────────────────────────────────
# Streaming chat via SSE
# ──────────────────────────────────────────────────────────────────────────

@router.post(
    "/conversations/{conversation_id}/messages/stream",
    summary="Send a message and receive streaming response (SSE)",
    response_class=EventSourceResponse,
)
async def send_message_stream(
    conversation_id: UUID,
    body: SendMessageRequest,
    use_case: ChatUseCaseDep,
):
    llm_config = _build_llm_config(body)
    rag_config = RAGConfig(top_k=body.rag_top_k, similarity_threshold=body.rag_score_threshold)

    async def event_generator():
        try:
            async for token in use_case.send_message_stream(
                conversation_id=conversation_id,
                user_content=body.content,
                llm_config=llm_config,
                rag_config=rag_config,
                use_rag=body.use_rag,
            ):
                yield {"event": "token", "data": token}

            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except ConversationNotFoundError as exc:
            yield {"event": "error", "data": json.dumps({"code": exc.code, "message": exc.message})}
        except AIProviderError as exc:
            yield {"event": "error", "data": json.dumps({"code": exc.code, "message": exc.message})}

    return EventSourceResponse(event_generator())


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _build_llm_config(body: SendMessageRequest) -> LLMConfig:
    from src.infrastructure.config.settings import get_settings
    settings = get_settings()
    return LLMConfig(
        temperature=body.temperature if body.temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=body.max_tokens if body.max_tokens is not None else settings.LLM_MAX_TOKENS,
        streaming=True,
    )
