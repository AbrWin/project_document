from __future__ import annotations

"""
Core → Use Cases → RAG Use Case
Handles document ingestion, chunking strategy, and retrieval.
"""

from typing import Optional
from uuid import UUID

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces.repositories import VectorStorePort
from src.domain.entities.document import Document, DocumentChunk
from src.domain.exceptions.ai_exceptions import EmbeddingError

logger = structlog.get_logger()

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


class RAGUseCase:
    """
    Manages the RAG ingestion pipeline:
    Load → Split → Embed → Store
    And retrieval for QA.
    """

    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    async def ingest_document(
        self,
        document: Document,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> Document:
        """
        Chunk a document and store embeddings in the vector store.
        Returns the document with populated chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(document.content)

        document.chunks = [
            DocumentChunk(
                content=chunk,
                document_id=document.id,
                chunk_index=i,
                metadata={
                    "source": document.source,
                    "title": document.title,
                    **document.metadata,
                },
            )
            for i, chunk in enumerate(raw_chunks)
        ]

        logger.info(
            "rag.ingest_document",
            document_id=str(document.id),
            source=document.source,
            chunk_count=len(document.chunks),
        )

        try:
            await self._vector_store.add_documents([document])
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed document '{document.source}': {exc}") from exc

        return document

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        collection: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Retrieve relevant document chunks for a query."""
        logger.info("rag.search", query=query[:80], top_k=top_k)
        return await self._vector_store.similarity_search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            collection=collection,
        )

    async def delete_document(self, document_id: UUID) -> bool:
        return await self._vector_store.delete_document(document_id)
