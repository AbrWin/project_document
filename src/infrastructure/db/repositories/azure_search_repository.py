from __future__ import annotations

"""
Infrastructure → DB → Repositories → Azure AI Search (Mode 2 - Search Only)
Implements VectorStorePort for QUERY operations only.
Documents are indexed by the Azure AI Search Indexer (Integrated Vectorization).
This adapter handles search queries and deletions — NOT document ingestion.
"""

import json
import uuid
from typing import List, Optional
from uuid import UUID

import structlog

from src.core.interfaces.ai_provider import AIProviderPort
from src.core.interfaces.repositories import VectorStorePort
from src.domain.entities.document import DocumentChunk
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger()

FIELD_ID = "id"
FIELD_CONTENT = "content"
FIELD_VECTOR = "content_vector"
FIELD_SOURCE = "source"
FIELD_SHEET = "sheet_name"
FIELD_ROW_DATA = "row_data"
FIELD_STORAGE_NAME = "metadata_storage_name"


class AzureSearchRepository(VectorStorePort):
    """
    Search adapter for Azure AI Search (Mode 2 - Integrated Vectorization).

    Documents are indexed by the Azure Indexer — this class only handles:
      - similarity_search(): embed query locally → HNSW vector search
      - text_search():        full-text search (no local embedding)
      - delete_document():    remove entries by source filter
    """

    def __init__(self, settings: Settings, embedding_provider: AIProviderPort) -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self._settings = settings
        self._embedding_provider = embedding_provider
        self._index_name = settings.AZURE_SEARCH_INDEX_NAME

        credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
        self._search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=self._index_name,
            credential=credential,
        )

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        collection: Optional[str] = None,
    ) -> List[DocumentChunk]:
        from azure.search.documents.models import VectorizedQuery

        vectors = await self._embedding_provider.embed([query])
        query_vector = vectors[0]

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=FIELD_VECTOR,
        )

        filter_expr = f"{FIELD_SOURCE} eq '{collection}'" if collection else None

        results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=[FIELD_ID, FIELD_CONTENT, FIELD_SOURCE, FIELD_SHEET, FIELD_ROW_DATA, FIELD_STORAGE_NAME],
            top=top_k,
        )

        return self._to_chunks(results, score_threshold)

    async def text_search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Full-text search — no local embedding needed."""
        filter_expr = f"{FIELD_SOURCE} eq '{source_filter}'" if source_filter else None

        results = self._search_client.search(
            search_text=query,
            filter=filter_expr,
            select=[FIELD_ID, FIELD_CONTENT, FIELD_SOURCE, FIELD_SHEET, FIELD_ROW_DATA, FIELD_STORAGE_NAME],
            top=top_k,
        )

        return self._to_chunks(results, score_threshold=0.0)

    async def delete_document(self, document_id: UUID) -> bool:
        results = self._search_client.search(
            search_text="*",
            filter=f"{FIELD_SOURCE} eq '{document_id}'",
            select=[FIELD_ID],
        )
        doc_ids = [{"id": r[FIELD_ID]} for r in results]
        if not doc_ids:
            logger.warning("azure_search.delete.not_found", document_id=str(document_id))
            return False

        self._search_client.delete_documents(documents=doc_ids)
        logger.info("azure_search.delete", document_id=str(document_id), chunks=len(doc_ids))
        return True

    def _to_chunks(self, results, score_threshold: float) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for result in results:
            score: float = result.get("@search.score", 0.0)
            if score < score_threshold:
                continue

            raw_row = result.get(FIELD_ROW_DATA, "{}")
            try:
                row_meta: dict = json.loads(raw_row)
            except (json.JSONDecodeError, TypeError):
                row_meta = {}

            chunk = DocumentChunk(
                content=result.get(FIELD_CONTENT, ""),
                document_id=uuid.uuid4(),
                chunk_index=0,
                metadata={
                    **row_meta,
                    "similarity_score": score,
                    "source": result.get(FIELD_SOURCE, ""),
                    "sheet_name": result.get(FIELD_SHEET, ""),
                    "storage_name": result.get(FIELD_STORAGE_NAME, ""),
                },
            )
            chunks.append(chunk)

        logger.info("azure_search.search", results=len(chunks), index=self._index_name)
        return chunks
