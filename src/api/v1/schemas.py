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
# Excel Schemas
# ============================================================

class ExcelSheetResponse(BaseModel):
    sheet_name: str
    columns: list[str]
    rows: list[dict]          # Array of objects — one per row
    row_count: int


class ExcelParseResponse(BaseModel):
    filename: str
    sheets: list[ExcelSheetResponse]
    total_rows: int


# ============================================================
# Integrated Vectorization (Mode 2) Schemas
# ============================================================

class ProvisionComponent(BaseModel):
    index: str
    datasource: str
    skillset: str
    indexer: str


class ProvisionResponse(BaseModel):
    message: str
    index_name: str
    components: ProvisionComponent


class IndexerStatusResponse(BaseModel):
    indexer: str
    status: str
    last_run_status: Optional[str] = None
    last_run_start: Optional[str] = None
    last_run_end: Optional[str] = None
    items_processed: int = 0
    items_failed: int = 0
    errors: list[str] = []


class BlobIngestResponse(BaseModel):
    filename: str
    blob_name: str
    blob_url: str
    sheets: list[ExcelSheetResponse]
    total_rows: int
    indexer_triggered: bool


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
