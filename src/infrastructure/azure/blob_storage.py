from __future__ import annotations

"""
Infrastructure → Azure → Blob Storage Client
Uploads files to Azure Blob Storage for Integrated Vectorization (Mode 2).
Azure AI Search Indexer reads from here automatically.
"""

import structlog
from azure.storage.blob import BlobServiceClient, ContentSettings

logger = structlog.get_logger()


class BlobStorageClient:
    """
    Thin wrapper around azure-storage-blob.
    Creates the container automatically if it does not exist.
    """

    def __init__(self, connection_string: str, container_name: str) -> None:
        self._container_name = container_name
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._service.get_container_client(container_name)
        self._ensure_container()

    def _ensure_container(self) -> None:
        try:
            self._container.create_container()
            logger.info("blob.container_created", container=self._container_name)
        except Exception:
            # Already exists — ignore
            pass

    def upload(
        self,
        blob_name: str,
        data: bytes,
        content_type: str = "application/json",
        overwrite: bool = True,
    ) -> str:
        """
        Upload bytes to a blob. Returns the blob URL (without SAS token).
        """
        blob_client = self._container.get_blob_client(blob_name)
        blob_client.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=ContentSettings(content_type=content_type),
        )
        url = blob_client.url
        logger.info("blob.uploaded", blob=blob_name, size_bytes=len(data), url=url)
        return url

    def delete(self, blob_name: str) -> None:
        blob_client = self._container.get_blob_client(blob_name)
        blob_client.delete_blob(delete_snapshots="include")
        logger.info("blob.deleted", blob=blob_name)

    def list_blobs(self) -> list[str]:
        return [b.name for b in self._container.list_blobs()]
