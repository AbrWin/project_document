from __future__ import annotations

"""
Core → Use Cases → RAG Use Case
Search and deletion operations for Mode 2 (Integrated Vectorization).
Ingestion is handled externally by the Azure AI Search Indexer.
"""

from typing import Optional
from uuid import UUID

import structlog

from src.core.interfaces.repositories import VectorStorePort
from src.domain.entities.document import DocumentChunk

logger = structlog.get_logger()


class RAGUseCase:
    """Handles search and deletion against the Azure AI Search vector store."""

    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

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
