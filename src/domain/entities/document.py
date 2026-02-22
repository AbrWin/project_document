from __future__ import annotations

"""Domain → Entities → Document
Entity for RAG - ingested documents and chunked segments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class DocumentChunk:
    content: str
    document_id: UUID
    id: UUID = field(default_factory=uuid4)
    chunk_index: int = 0
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    source: str                               # file path, URL, etc.
    content: str
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None
    mime_type: str = "text/plain"
    created_at: datetime = field(default_factory=datetime.utcnow)
    chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
