from __future__ import annotations

"""
API → V1 → Endpoints → RAG
Document ingestion and semantic search endpoints.
"""

from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, File, status

from src.api.v1.schemas import (
    ChunkResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    SearchRequest,
    SearchResponse,
)
from src.domain.entities.document import Document
from src.domain.exceptions.ai_exceptions import EmbeddingError
from src.infrastructure.container import RAGUseCaseDep

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = structlog.get_logger()


@router.post(
    "/documents",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a text document into the vector store",
)
async def ingest_document(
    body: IngestDocumentRequest,
    use_case: RAGUseCaseDep,
):
    doc = Document(
        id=uuid4(),
        source=body.source,
        content=body.content,
        title=body.title,
        metadata=body.metadata,
    )

    try:
        doc = await use_case.ingest_document(
            document=doc,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    return IngestDocumentResponse(
        document_id=doc.id,
        source=doc.source,
        chunk_count=len(doc.chunks),
    )


@router.post(
    "/documents/upload",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file and ingest it into the vector store",
)
async def upload_and_ingest(
    file: UploadFile = File(...),
    use_case: RAGUseCaseDep = None,
):
    if file.content_type not in ("text/plain", "text/markdown", "application/json"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use text/plain or text/markdown.",
        )

    content = (await file.read()).decode("utf-8")
    doc = Document(
        id=uuid4(),
        source=file.filename or "uploaded_file",
        content=content,
        title=file.filename,
        mime_type=file.content_type or "text/plain",
    )

    try:
        doc = await use_case.ingest_document(document=doc)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    return IngestDocumentResponse(
        document_id=doc.id,
        source=doc.source,
        chunk_count=len(doc.chunks),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic similarity search across ingested documents",
)
async def search_documents(
    body: SearchRequest,
    use_case: RAGUseCaseDep,
):
    chunks = await use_case.search(
        query=body.query,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        collection=body.collection,
    )

    return SearchResponse(
        query=body.query,
        results=[
            ChunkResponse(
                content=chunk.content,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.metadata.get("similarity_score", 0.0),
                metadata={k: v for k, v in chunk.metadata.items() if k != "similarity_score"},
            )
            for chunk in chunks
        ],
        total=len(chunks),
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all its embeddings",
)
async def delete_document(
    document_id: UUID,
    use_case: RAGUseCaseDep,
):
    await use_case.delete_document(document_id)
