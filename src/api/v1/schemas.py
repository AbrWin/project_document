from __future__ import annotations

"""
API → V1 → Schemas
Pydantic v2 schemas for request/response validation.
Separate from domain entities — presentation layer concern.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# Chat Schemas
# ============================================================

class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, examples=["Mi primera conversación"])


class ConversationResponse(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32_000, examples=["¿Qué es LangChain?"])
    provider: Optional[str] = Field(
        None,
        description="Override provider: azure_openai | azure_inference | openai",
        examples=["azure_openai"],
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=128_000)
    use_rag: bool = Field(False, description="Inject relevant documents from vector store")
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_score_threshold: float = Field(0.7, ge=0.0, le=1.0)


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    conversation_id: Optional[UUID]
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []


# ============================================================
# RAG Schemas
# ============================================================

class IngestDocumentRequest(BaseModel):
    source: str = Field(..., examples=["https://docs.example.com/page"])
    content: str = Field(..., min_length=10)
    title: Optional[str] = None
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=1000)
    metadata: dict = Field(default_factory=dict)


class IngestDocumentResponse(BaseModel):
    document_id: UUID
    source: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)
    collection: Optional[str] = None


class ChunkResponse(BaseModel):
    content: str
    document_id: UUID
    chunk_index: int
    similarity_score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResponse]
    total: int


# ============================================================
# Health Schemas
# ============================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    provider: str


# ============================================================
# Error Schema
# ============================================================

class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None
