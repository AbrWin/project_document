from __future__ import annotations

"""
Infrastructure → DB → Repositories → PGVector Store
Implements VectorStorePort using LangChain + pgvector extension.
Handles embedding + storage + similarity search.
"""

from typing import Optional
from uuid import UUID

import structlog
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore

from src.core.interfaces.ai_provider import AIProviderPort
from src.core.interfaces.repositories import VectorStorePort
from src.domain.entities.document import Document, DocumentChunk
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger()


class _LCEmbeddingsAdapter:
    """Thin adapter so our AIProviderPort works as a LangChain Embeddings object."""

    def __init__(self, provider: AIProviderPort) -> None:
        self._provider = provider

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        results = await self._provider.embed([text])
        return results[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.aembed_query(text))


class PGVectorRepository(VectorStorePort):
    """
    Vector store backed by PostgreSQL + pgvector.
    Requires the pgvector extension installed in your database:
        CREATE EXTENSION IF NOT EXISTS vector;
    """

    def __init__(self, settings: Settings, embedding_provider: AIProviderPort) -> None:
        self._settings = settings
        self._embeddings = _LCEmbeddingsAdapter(embedding_provider)
        self._sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")

    def _get_store(self, collection: Optional[str] = None) -> PGVectorStore:
        return PGVector(
            embeddings=self._embeddings,
            collection_name=collection or self._settings.VECTOR_COLLECTION_NAME,
            connection=self._sync_db_url,
            use_jsonb=True,
        )

    async def add_documents(self, documents: list[Document]) -> list[str]:
        from langchain_core.documents import Document as LCDoc

        all_ids: list[str] = []
        for doc in documents:
            lc_docs = [
                LCDoc(
                    page_content=chunk.content,
                    metadata={
                        "document_id": str(doc.id),
                        "chunk_index": chunk.chunk_index,
                        "source": doc.source,
                        **chunk.metadata,
                    },
                )
                for chunk in doc.chunks
            ]
            store = self._get_store()
            ids = store.add_documents(lc_docs)
            all_ids.extend(ids)
            logger.info("pgvector.add_documents", document_id=str(doc.id), chunks=len(lc_docs))

        return all_ids

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        collection: Optional[str] = None,
    ) -> list[DocumentChunk]:
        from uuid import uuid4

        store = self._get_store(collection)
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            score_threshold=score_threshold,
        )

        chunks = []
        for lc_doc, score in results:
            meta = lc_doc.metadata or {}
            doc_id_str = meta.get("document_id", str(uuid4()))
            chunk = DocumentChunk(
                content=lc_doc.page_content,
                document_id=UUID(doc_id_str) if doc_id_str else uuid4(),
                chunk_index=meta.get("chunk_index", 0),
                metadata={**meta, "similarity_score": score},
            )
            chunks.append(chunk)

        return chunks

    async def delete_document(self, document_id: UUID) -> bool:
        store = self._get_store()
        # PGVector supports filtering by metadata
        store.delete(filter={"document_id": str(document_id)})
        logger.info("pgvector.delete_document", document_id=str(document_id))
        return True
