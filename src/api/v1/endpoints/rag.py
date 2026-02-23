from __future__ import annotations

"""
API → V1 → Endpoints → RAG
Document ingestion and semantic search endpoints.
"""

from typing import Annotated, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, status

from src.api.v1.schemas import (
    BlobIngestResponse,
    ChunkResponse,
    ExcelParseResponse,
    ExcelSheetResponse,
    IndexerStatusResponse,
    ProvisionComponent,
    ProvisionResponse,
    SearchRequest,
    SearchResponse,
)
from src.infrastructure.container import RAGUseCaseDep
from src.infrastructure.config.settings import get_settings

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = structlog.get_logger()


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


@router.post(
    "/documents/excel",
    response_model=ExcelParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload an Excel file (.xlsx / .xls) → array of row-objects per sheet",
    description=(
        "Parses every sheet into an array of objects where each key is a column name."
    ),
)
async def upload_excel(
    file: Annotated[UploadFile, File(description=".xlsx or .xls file")],
    sheet_name: Annotated[
        Optional[str],
        Form(description="Parse only this sheet. Leave empty to parse all sheets."),
    ] = None,
):
    filename = file.filename or "upload.xlsx"
    allowed_types = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel",                                           # .xls
        "application/octet-stream",                                           # generic binary
    )
    if file.content_type not in allowed_types and not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a .xlsx or .xls file.",
        )

    raw_bytes = await file.read()

    from src.infrastructure.parsers.excel_parser import parse_excel

    try:
        sheet_results = parse_excel(raw_bytes, filename=filename, sheet_name=sheet_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    total_rows = sum(s.row_count for s in sheet_results)

    return ExcelParseResponse(
        filename=filename,
        sheets=[
            ExcelSheetResponse(
                sheet_name=s.sheet_name,
                columns=s.columns,
                rows=s.rows,
                row_count=s.row_count,
            )
            for s in sheet_results
        ],
        total_rows=total_rows,
    )


@router.post(
    "/provision",
    response_model=ProvisionResponse,
    status_code=status.HTTP_200_OK,
    summary="[Mode 2] Create Azure AI Search index, data source, skillset and indexer",
    description=(
        "One-time setup for Integrated Vectorization. "
        "Requires AZURE_STORAGE_CONNECTION_STRING and AZURE_OPENAI credentials in .env. "
        "Safe to call multiple times — already-existing components are skipped."
    ),
)
async def provision_azure_search():
    settings = get_settings()
    if not settings.use_integrated_vectorization:
        raise HTTPException(
            status_code=503,
            detail=(
                "Integrated Vectorization is not configured. "
                "Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, "
                "AZURE_STORAGE_CONNECTION_STRING, AZURE_OPENAI_ENDPOINT, "
                "and AZURE_OPENAI_API_KEY in your .env file."
            ),
        )

    from src.infrastructure.azure.search_provisioner import SearchProvisioner

    provisioner = SearchProvisioner(
        search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
        search_api_key=settings.AZURE_SEARCH_API_KEY,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        storage_connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        storage_container=settings.AZURE_STORAGE_CONTAINER_NAME,
        openai_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_embedding_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        semantic_config_name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
        datasource_name=settings.AZURE_SEARCH_DATASOURCE_NAME,
        indexer_name=settings.AZURE_SEARCH_INDEXER_NAME,
        skillset_name=settings.AZURE_SEARCH_SKILLSET_NAME,
    )

    results = provisioner.provision_all()

    return ProvisionResponse(
        message="Azure AI Search pipeline provisioned successfully.",
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        components=ProvisionComponent(
            index=results["index"],
            datasource=results["datasource"],
            skillset=results["skillset"],
            indexer=results["indexer"],
        ),
    )


@router.post(
    "/documents",
    response_model=BlobIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="[Mode 2] Upload Excel → Blob Storage → Azure AI Search auto-indexes via Indexer",
    description=(
        "Converts every row of the Excel to a JSON document, uploads to Azure Blob Storage, "
        "then triggers the Azure AI Search Indexer. "
        "Azure OpenAI generates the embeddings automatically via the Skillset (no local embedding call). "
        "Call GET /rag/indexer/status to check progress."
    ),
)
async def upload_excel_to_blob(
    file: Annotated[UploadFile, File(description=".xlsx or .xls file")],
    sheet_name: Annotated[
        Optional[str],
        Form(description="Parse only this sheet. Leave empty to parse all sheets."),
    ] = None,
    trigger: Annotated[
        bool,
        Form(description="Trigger the indexer immediately after upload. Default: true."),
    ] = True,
):
    settings = get_settings()
    if not settings.use_integrated_vectorization:
        raise HTTPException(
            status_code=503,
            detail=(
                "Integrated Vectorization is not configured. "
                "Run POST /api/v1/rag/provision first and set the required env vars."
            ),
        )

    filename = file.filename or "upload.xlsx"
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=415, detail="Upload a .xlsx or .xls file.")

    raw_bytes = await file.read()

    from src.infrastructure.parsers.excel_parser import parse_excel
    from src.infrastructure.azure.blob_storage import BlobStorageClient
    from src.infrastructure.azure.search_provisioner import SearchProvisioner
    import json, uuid as _uuid

    try:
        sheet_results = parse_excel(raw_bytes, filename=filename, sheet_name=sheet_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Build JSON array — one object per row, with pre-formatted text content
    all_json_docs = []
    for sheet in sheet_results:
        text_rows = sheet.to_text_rows()
        for i, (row, text) in enumerate(zip(sheet.rows, text_rows)):
            all_json_docs.append({
                "id": str(_uuid.uuid4()),
                "content": text,                          # text the skillset will embed
                "source": filename,
                "sheet_name": sheet.sheet_name,
                "row_data": json.dumps(row, ensure_ascii=False),
            })

    blob_name = f"{filename.rsplit('.', 1)[0]}_{_uuid.uuid4().hex[:8]}.json"
    json_bytes = json.dumps(all_json_docs, ensure_ascii=False, indent=2).encode("utf-8")

    blob_client = BlobStorageClient(
        connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        container_name=settings.AZURE_STORAGE_CONTAINER_NAME,
    )
    blob_url = blob_client.upload(blob_name=blob_name, data=json_bytes)

    if trigger:
        provisioner = SearchProvisioner(
            search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
            search_api_key=settings.AZURE_SEARCH_API_KEY,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            storage_connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
            storage_container=settings.AZURE_STORAGE_CONTAINER_NAME,
            openai_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            openai_embedding_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            datasource_name=settings.AZURE_SEARCH_DATASOURCE_NAME,
            indexer_name=settings.AZURE_SEARCH_INDEXER_NAME,
            skillset_name=settings.AZURE_SEARCH_SKILLSET_NAME,
        )
        provisioner.trigger_indexer()

    total_rows = sum(s.row_count for s in sheet_results)

    logger.info(
        "excel.blob_uploaded",
        filename=filename,
        blob=blob_name,
        rows=total_rows,
        indexer_triggered=trigger,
    )

    return BlobIngestResponse(
        filename=filename,
        blob_name=blob_name,
        blob_url=blob_url,
        sheets=[
            ExcelSheetResponse(
                sheet_name=s.sheet_name,
                columns=s.columns,
                rows=s.rows,
                row_count=s.row_count,
            )
            for s in sheet_results
        ],
        total_rows=total_rows,
        indexer_triggered=trigger,
    )


@router.get(
    "/indexer/status",
    response_model=IndexerStatusResponse,
    summary="[Mode 2] Get the current status of the Azure AI Search Indexer",
)
async def get_indexer_status():
    settings = get_settings()
    if not settings.use_integrated_vectorization:
        raise HTTPException(
            status_code=503,
            detail="Integrated Vectorization is not configured.",
        )

    from src.infrastructure.azure.search_provisioner import SearchProvisioner

    provisioner = SearchProvisioner(
        search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
        search_api_key=settings.AZURE_SEARCH_API_KEY,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        storage_connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        storage_container=settings.AZURE_STORAGE_CONTAINER_NAME,
        openai_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_embedding_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        datasource_name=settings.AZURE_SEARCH_DATASOURCE_NAME,
        indexer_name=settings.AZURE_SEARCH_INDEXER_NAME,
        skillset_name=settings.AZURE_SEARCH_SKILLSET_NAME,
    )
    status_data = provisioner.get_indexer_status()
    return IndexerStatusResponse(**status_data)


@router.post(
    "/indexer/reset",
    status_code=status.HTTP_200_OK,
    summary="[Mode 2] Reset indexer state and re-trigger — forces reprocessing of all blobs",
)
async def reset_and_run_indexer():
    settings = get_settings()
    if not settings.use_integrated_vectorization:
        raise HTTPException(status_code=503, detail="Integrated Vectorization is not configured.")

    from src.infrastructure.azure.search_provisioner import SearchProvisioner

    provisioner = SearchProvisioner(
        search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
        search_api_key=settings.AZURE_SEARCH_API_KEY,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        storage_connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        storage_container=settings.AZURE_STORAGE_CONTAINER_NAME,
        openai_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_embedding_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        datasource_name=settings.AZURE_SEARCH_DATASOURCE_NAME,
        indexer_name=settings.AZURE_SEARCH_INDEXER_NAME,
        skillset_name=settings.AZURE_SEARCH_SKILLSET_NAME,
    )
    provisioner.reset_indexer()
    provisioner.trigger_indexer()
    return {"message": "Indexer reset and triggered. Call GET /indexer/status to check progress."}


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
