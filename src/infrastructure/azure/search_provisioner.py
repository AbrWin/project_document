from __future__ import annotations

"""
Infrastructure → Azure → Search Provisioner
Creates / updates the full Azure AI Search pipeline for Integrated Vectorization (Mode 2):

  Blob Storage  →  Data Source  →  Indexer  →  Skillset (Azure OpenAI embed)  →  Index

Call provision_all() once to bootstrap everything.
Then upload JSON blobs and call trigger_indexer() to index new content.
"""

import structlog
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    FieldMapping,
    HnswAlgorithmConfiguration,
    IndexingParameters,
    IndexingParametersConfiguration,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerSkillset,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Field names — must match what we upload in the JSON blob
# ---------------------------------------------------------------------------
F_ID = "id"
F_CONTENT = "content"
F_VECTOR = "content_vector"
F_SOURCE = "source"
F_SHEET = "sheet_name"
F_ROW_DATA = "row_data"          # original JSON of the row (string)
F_STORAGE_PATH = "metadata_storage_path"
F_STORAGE_NAME = "metadata_storage_name"


class SearchProvisioner:
    """
    Provisions and manages the Azure AI Search Integrated Vectorization pipeline.

    Pipeline:
        Azure Blob Storage
            └─ Data Source Connection
                └─ Indexer  (parsingMode: jsonArray)
                    └─ Skillset
                        └─ AzureOpenAIEmbeddingSkill → content → content_vector
                            └─ Index  (HNSW + semantic search)
    """

    def __init__(
        self,
        search_endpoint: str,
        search_api_key: str,
        index_name: str,
        storage_connection_string: str,
        storage_container: str,
        openai_endpoint: str,
        openai_api_key: str,
        openai_embedding_deployment: str,
        embedding_dimension: int = 3072,
        semantic_config_name: str = "default",
        datasource_name: str = None,
        indexer_name: str = None,
        skillset_name: str = None,
    ) -> None:
        credential = AzureKeyCredential(search_api_key)
        self._index_client = SearchIndexClient(search_endpoint, credential)
        self._indexer_client = SearchIndexerClient(search_endpoint, credential)

        self._index_name = index_name
        self._datasource_name = datasource_name or f"{index_name}-datasource"
        self._skillset_name = skillset_name or f"{index_name}-skillset"
        self._indexer_name = indexer_name or f"{index_name}-indexer"
        self._semantic_config = semantic_config_name

        self._storage_connection_string = storage_connection_string
        self._storage_container = storage_container

        self._openai_endpoint = openai_endpoint
        self._openai_api_key = openai_api_key
        self._openai_embedding_deployment = openai_embedding_deployment
        self._embedding_dimension = embedding_dimension

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def provision_all(self) -> dict:
        """
        Idempotent — creates everything that doesn't already exist.
        Returns a dict with the status of each component.
        """
        results = {}
        results["index"] = self._create_index()
        results["datasource"] = self._create_datasource()
        results["skillset"] = self._create_skillset()
        results["indexer"] = self._create_indexer()
        logger.info("search_provisioner.provisioned", **results)
        return results

    def trigger_indexer(self) -> None:
        """Manually run the indexer immediately (picks up new/changed blobs)."""
        self._indexer_client.run_indexer(self._indexer_name)
        logger.info("search_provisioner.indexer_triggered", indexer=self._indexer_name)

    def get_indexer_status(self) -> dict:
        """Return the last run status of the indexer."""
        status = self._indexer_client.get_indexer_status(self._indexer_name)
        last_run = status.last_result

        def _str(val) -> str:
            """Return string from either an enum or a plain string."""
            if val is None:
                return "unknown"
            return val.value if hasattr(val, "value") else str(val)

        return {
            "indexer": self._indexer_name,
            "status": _str(status.status),
            "last_run_status": _str(last_run.status) if last_run else None,
            "last_run_start": last_run.start_time.isoformat() if last_run and last_run.start_time else None,
            "last_run_end": last_run.end_time.isoformat() if last_run and last_run.end_time else None,
            "items_processed": last_run.item_count if last_run else 0,
            "items_failed": last_run.failed_item_count if last_run else 0,
            "errors": [str(e.error_message) for e in (last_run.errors or [])] if last_run else [],
        }

    def reset_indexer(self) -> None:
        """Reset indexer state — next run will re-process all blobs."""
        self._indexer_client.reset_indexer(self._indexer_name)
        logger.info("search_provisioner.indexer_reset", indexer=self._indexer_name)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _create_index(self) -> str:
        fields = [
            SimpleField(name=F_ID, type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name=F_CONTENT, type=SearchFieldDataType.String, analyzer_name="es.microsoft"),
            SimpleField(name=F_SOURCE, type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name=F_SHEET, type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name=F_ROW_DATA, type=SearchFieldDataType.String),
            SimpleField(name=F_STORAGE_NAME, type=SearchFieldDataType.String, filterable=True),
            SimpleField(name=F_STORAGE_PATH, type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name=F_VECTOR,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._embedding_dimension,
                vector_search_profile_name="hnsw-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
            profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-algo")],
        )

        semantic_config = SemanticConfiguration(
            name=self._semantic_config,
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name=F_CONTENT)],
            ),
        )

        index = SearchIndex(
            name=self._index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config]),
        )

        try:
            self._index_client.create_index(index)
            logger.info("search_provisioner.index_created", index=self._index_name)
            return "created"
        except ResourceExistsError:
            return "already_exists"
        except Exception as exc:
            # Index exists with different schema — update it
            if "already exists" in str(exc).lower():
                return "already_exists"
            raise

    # ------------------------------------------------------------------
    # Data Source
    # ------------------------------------------------------------------

    def _create_datasource(self) -> str:
        datasource = SearchIndexerDataSourceConnection(
            name=self._datasource_name,
            type="azureblob",
            connection_string=self._storage_connection_string,
            container=SearchIndexerDataContainer(name=self._storage_container),
        )
        try:
            self._indexer_client.create_or_update_data_source_connection(datasource)
            logger.info("search_provisioner.datasource_upserted", name=self._datasource_name)
            return "created"
        except Exception as exc:
            raise

    # ------------------------------------------------------------------
    # Skillset — AzureOpenAI embedding skill
    # ------------------------------------------------------------------

    def _create_skillset(self) -> str:
        embedding_skill = AzureOpenAIEmbeddingSkill(
            name="embed-content",
            description="Generate embeddings for the content field using Azure OpenAI",
            context="/document",
            resource_url=self._openai_endpoint,
            api_key=self._openai_api_key,
            deployment_name=self._openai_embedding_deployment,
            model_name=self._openai_embedding_deployment,
            inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
            outputs=[OutputFieldMappingEntry(name="embedding", target_name="content_vector")],
        )

        skillset = SearchIndexerSkillset(
            name=self._skillset_name,
            description="Integrated vectorization skillset for structured data",
            skills=[embedding_skill],
        )
        try:
            self._indexer_client.create_or_update_skillset(skillset)
            logger.info("search_provisioner.skillset_upserted", name=self._skillset_name)
            return "created"
        except Exception as exc:
            raise

    # ------------------------------------------------------------------
    # Indexer
    # ------------------------------------------------------------------

    def _create_indexer(self) -> str:
        indexer = SearchIndexer(
            name=self._indexer_name,
            description="Indexes JSON array blobs from the Excel upload pipeline",
            data_source_name=self._datasource_name,
            target_index_name=self._index_name,
            skillset_name=self._skillset_name,
            parameters=IndexingParameters(
                configuration=IndexingParametersConfiguration(
                    parsing_mode="jsonArray",          # each JSON array item → one document
                    query_timeout=None,
                ),
                batch_size=100,
                max_failed_items=10,
                max_failed_items_per_batch=5,
            ),
            field_mappings=[
                FieldMapping(source_field_name=F_ID, target_field_name=F_ID),
                FieldMapping(source_field_name=F_CONTENT, target_field_name=F_CONTENT),
                FieldMapping(source_field_name=F_SOURCE, target_field_name=F_SOURCE),
                FieldMapping(source_field_name=F_SHEET, target_field_name=F_SHEET),
                FieldMapping(source_field_name=F_ROW_DATA, target_field_name=F_ROW_DATA),
                FieldMapping(
                    source_field_name="metadata_storage_name",
                    target_field_name=F_STORAGE_NAME,
                ),
                FieldMapping(
                    source_field_name="metadata_storage_path",
                    target_field_name=F_STORAGE_PATH,
                ),
            ],
            output_field_mappings=[
                FieldMapping(
                    source_field_name="/document/content_vector",
                    target_field_name=F_VECTOR,
                )
            ],
        )
        try:
            # Reset change-tracking state so we can switch datasources
            try:
                self._indexer_client.reset_indexer(self._indexer_name)
                logger.info("search_provisioner.indexer_reset_before_upsert", name=self._indexer_name)
            except Exception:
                pass  # indexer doesn't exist yet — that's fine
            self._indexer_client.create_or_update_indexer(indexer)
            logger.info("search_provisioner.indexer_upserted", name=self._indexer_name)
            return "created"
        except Exception as exc:
            raise
