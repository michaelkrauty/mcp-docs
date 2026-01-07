"""Document indexer for Qdrant with hybrid search support."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue, PayloadSchemaType, PointStruct
from vector_core import (
    EmbeddingClient,
    GlobalVocabulary,
    QdrantStorage,
    create_hybrid_point_with_key,
    generate_point_id,
)

from mcp_docs.extraction.extractor import extract_content
from mcp_docs.indexing.chunker import DocumentChunker, chunk_document
from mcp_docs.models import Document, DocumentType, ExtractionStatus, ExtractionError
from mcp_docs.settings import settings
from mcp_docs.storage.database import DocumentStore

logger = logging.getLogger(__name__)

# Codebase ID for GlobalVocabulary registration
DOCS_CODEBASE_ID = "docs"


class DocumentIndexer:
    """
    Indexes documents into Qdrant for hybrid search.

    Uses vector-core components:
    - EmbeddingClient for dense vectors
    - GlobalVocabulary for cross-codebase sparse vectors (two-pass indexing)
    - QdrantStorage for vector storage

    Documents are indexed as:
    - type="document" for document-level summary
    - type="doc_chunk" for individual chunks
    """

    def __init__(
        self,
        document_store: DocumentStore,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize indexer.

        Args:
            document_store: DocumentStore for accessing documents
            storage: QdrantStorage instance (created if not provided)
            embedder: EmbeddingClient instance (created if not provided)
            global_vocab: GlobalVocabulary instance (created if not provided)
            collection_name: Qdrant collection name (from settings if not provided)
        """
        self.document_store = document_store
        self.storage = storage
        self.embedder = embedder
        self.global_vocab = global_vocab
        self._collection_name = collection_name
        self._chunker = DocumentChunker()

    async def _ensure_components(self) -> None:
        """Ensure async components are initialized."""
        if self.storage is None:
            self.storage = QdrantStorage()
        if self.embedder is None:
            self.embedder = EmbeddingClient()
        if self.global_vocab is None:
            self.global_vocab = GlobalVocabulary.get_instance()

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        if self._collection_name is None:
            self._collection_name = settings.collection_name
        return self._collection_name

    async def ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with required indexes."""
        await self._ensure_components()

        if not await self.storage.collection_exists(self.collection_name):
            await self.storage.create_collection(self.collection_name)
            logger.info(f"Created collection: {self.collection_name}")

        # Ensure payload indexes for efficient filtering (idempotent)
        await self.storage.ensure_payload_indexes(
            self.collection_name,
            [
                ("type", PayloadSchemaType.KEYWORD),
                ("document_id", PayloadSchemaType.KEYWORD),
                ("content_hash", PayloadSchemaType.KEYWORD),
                ("doc_type", PayloadSchemaType.KEYWORD),
                ("tags", PayloadSchemaType.KEYWORD),
            ],
        )

    async def index_document(
        self,
        document_id: UUID,
        content: str,
    ) -> int:
        """
        Index a single document.

        For single document indexing, GlobalVocabulary should already be trained.
        If not, a warning is logged and sparse vectors may be suboptimal.

        Args:
            document_id: Document UUID
            content: Extracted document content

        Returns:
            Number of points indexed
        """
        await self._ensure_components()
        await self.ensure_collection()

        # Get document metadata
        document = self.document_store.read(document_id)
        if document is None:
            logger.error(f"Document not found: {document_id}")
            return 0

        # Ensure GlobalVocabulary has vocabulary
        if self.global_vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 0:
            logger.warning(
                "GlobalVocabulary not trained for docs codebase. "
                "Sparse vectors will be trained on this document only."
            )
            # Train on just this document
            tokens = set(self.global_vocab.tokenize(content))
            self.global_vocab.register_codebase(DOCS_CODEBASE_ID, [tokens])

        # Delete existing points
        await self._delete_document_points(document_id)

        # Index document
        points = await self._create_document_points(document, content)

        # Upsert to Qdrant
        if points:
            await self.storage.upsert_batch(self.collection_name, points)
            # Update status to indexed
            self.document_store.update(
                document_id,
                extraction_status=ExtractionStatus.INDEXED,
            )
            logger.info(f"Indexed document {document_id}: {len(points)} points")

        return len(points)

    async def index_all(self, force: bool = False) -> dict:
        """
        Index all extracted documents using two-pass GlobalVocabulary pattern.

        Pass 1: Extract content and collect tokens for GlobalVocabulary training
        Pass 2: Generate embeddings and sparse vectors, create summary + chunk points

        Args:
            force: If True, reindex everything. If False, incremental update.

        Returns:
            Status dict with indexed count, points, errors
        """
        await self._ensure_components()
        await self.ensure_collection()

        # Get all extracted documents
        docs_to_index = self.document_store.query(
            extraction_status=ExtractionStatus.EXTRACTED,
        )

        if not docs_to_index:
            logger.info("No documents to index")
            return {"indexed": 0, "total": 0}

        # Filter for incremental updates
        if not force:
            indexed_hashes = await self._get_indexed_hashes()
            docs_to_index = [
                doc
                for doc in docs_to_index
                if self._doc_hash(doc) not in indexed_hashes
            ]

        if not docs_to_index:
            logger.info("All documents already indexed")
            return {"indexed": 0, "total": len(indexed_hashes), "skipped": True}

        logger.info(f"Indexing {len(docs_to_index)} documents")

        # Pass 1: Extract content and collect tokens for GlobalVocabulary
        # We re-extract content here since it's not stored in the database
        doc_contents: dict[UUID, str] = {}
        tokens_per_doc: list[set[str]] = []
        extraction_errors: list[str] = []

        for doc in docs_to_index:
            try:
                path = Path(doc.path)
                if not path.exists():
                    logger.warning(f"Document file not found: {doc.path}")
                    extraction_errors.append(f"{doc.filename}: file not found")
                    continue

                # Re-extract content for indexing
                extracted = extract_content(path, DocumentType(doc.doc_type))
                content = extracted.text
                doc_contents[doc.id] = content

                # Tokenize actual content (not just metadata)
                tokens = set(self.global_vocab.tokenize(content))
                tokens_per_doc.append(tokens)
                logger.debug(f"Tokenized {doc.filename}: {len(tokens)} unique tokens")

            except ExtractionError as e:
                logger.warning(f"Failed to extract {doc.filename}: {e}")
                extraction_errors.append(f"{doc.filename}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error extracting {doc.filename}: {e}")
                extraction_errors.append(f"{doc.filename}: {e}")

        if not doc_contents:
            logger.error("No documents could be extracted for indexing")
            return {
                "indexed": 0,
                "total": self.document_store.count(),
                "errors": extraction_errors,
            }

        # Register vocabulary with GlobalVocabulary
        logger.info(f"Registering vocabulary from {len(tokens_per_doc)} documents")
        self.global_vocab.register_codebase(DOCS_CODEBASE_ID, tokens_per_doc)

        # Pass 2: Generate embeddings and create points (summary + chunks)
        total_points = 0
        indexed_count = 0

        for doc in docs_to_index:
            if doc.id not in doc_contents:
                continue  # Extraction failed for this doc

            try:
                content = doc_contents[doc.id]

                # Delete any existing points for this document
                await self._delete_document_points(doc.id)

                # Create both summary AND chunk points (like index_document does)
                points = await self._create_document_points(doc, content)

                if points:
                    await self.storage.upsert_batch(self.collection_name, points)
                    total_points += len(points)
                    indexed_count += 1
                    logger.debug(f"Indexed {doc.filename}: {len(points)} points")

                # Update status to indexed
                self.document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)

            except Exception as e:
                logger.error(f"Failed to index document {doc.id}: {e}")
                extraction_errors.append(f"{doc.filename}: indexing failed - {e}")

        logger.info(f"Indexed {indexed_count} documents with {total_points} points")

        return {
            "indexed": indexed_count,
            "points": total_points,
            "total": self.document_store.count(),
            "errors": extraction_errors if extraction_errors else None,
        }

    async def delete_document_index(self, document_id: UUID) -> None:
        """
        Remove a document from the index.

        Args:
            document_id: Document UUID to remove
        """
        await self._delete_document_points(document_id)

    async def _create_document_points(
        self,
        document: Document,
        content: str,
    ) -> list[PointStruct]:
        """Create Qdrant points for a document and its chunks."""
        points: list[PointStruct] = []

        # Chunk the document
        chunks = chunk_document(document.id, content, document.page_count)

        # Prepare texts for batch embedding
        texts = [chunk.content for chunk in chunks]

        # Add document summary
        summary = self._generate_doc_summary(document)
        texts.insert(0, summary)

        # Batch embed
        embeddings = await self.embedder.embed_batch(texts)

        # Create summary point
        points.append(
            self._create_point(
                point_type="document",
                document=document,
                content=summary,
                embedding=embeddings[0],
            )
        )

        # Create chunk points
        for i, chunk in enumerate(chunks):
            points.append(
                self._create_point(
                    point_type="doc_chunk",
                    document=document,
                    content=chunk.content,
                    embedding=embeddings[i + 1],
                    chunk_index=chunk.chunk_index,
                    section_title=chunk.section_title,
                )
            )

        return points

    async def _create_summary_point(
        self,
        document: Document,
        summary: str,
    ) -> list[PointStruct]:
        """Create just a summary point for incremental indexing."""
        embedding = await self.embedder.embed(summary)
        return [
            self._create_point(
                point_type="document",
                document=document,
                content=summary,
                embedding=embedding,
            )
        ]

    def _create_point(
        self,
        point_type: str,
        document: Document,
        content: str,
        embedding: list[float],
        chunk_index: int | None = None,
        section_title: str | None = None,
    ) -> PointStruct:
        """Create a Qdrant point with dense + sparse vectors."""
        # Generate deterministic key for point ID
        if chunk_index is not None:
            key = f"{point_type}:{document.id}:{chunk_index}"
        else:
            key = f"{point_type}:{document.id}"

        # Generate sparse vector
        sparse = self.global_vocab.vectorize_document(content)

        # Build payload
        payload = {
            "type": point_type,
            "document_id": str(document.id),
            "content": content,
            "content_hash": document.content_hash,
            "filename": document.filename,
            "path": document.path,
            "doc_type": document.doc_type,  # Already string due to use_enum_values=True
            "title": document.title,
            "tags": document.tags,
            "doc_hash": self._doc_hash(document),
            "indexed_at": datetime.now(UTC).isoformat(),
        }

        if chunk_index is not None:
            payload["chunk_index"] = chunk_index
        if section_title:
            payload["section_title"] = section_title

        return create_hybrid_point_with_key(key, embedding, sparse, payload)

    def _generate_doc_summary(self, document: Document) -> str:
        """Generate summary text for a document."""
        parts = [document.filename]
        if document.title and document.title != document.filename:
            parts.append(document.title)
        if document.tags:
            parts.append(f"Tags: {', '.join(document.tags)}")
        return " | ".join(parts)

    def _doc_hash(self, document: Document) -> str:
        """
        Generate truncated hash for incremental indexing cache key.

        This combines document content_hash with mutable metadata (title, tags)
        to detect when a document needs reindexing. The 16-char truncation is
        acceptable here because:
        1. This is only for cache invalidation, not document identity
        2. Full content_hash is preserved in the document record
        3. 64 bits provides sufficient collision resistance for ~10K documents
           (birthday paradox: sqrt(2^64) = 4B collisions before ~50% probability)
        """
        content = f"{document.content_hash}:{document.title or ''}:{','.join(document.tags)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_indexed_hashes(self) -> set[str]:
        """Get hashes of already indexed documents."""
        try:
            results = await self.storage.scroll_points(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="type", match=MatchValue(value="document")),
                ],
                payload_fields=["doc_hash"],
                limit=10000,
            )
            # scroll_points returns list of payload dicts, not ScoredPoint objects
            return {p.get("doc_hash", "") for p in results if p}
        except Exception as e:
            logger.debug(f"Could not retrieve indexed hashes (collection may not exist): {e}")
            return set()

    async def _delete_document_points(self, document_id: UUID) -> None:
        """Delete all points for a document."""
        try:
            await self.storage.delete_by_filter(
                self.collection_name,
                field="document_id",
                value=str(document_id),
            )
        except Exception as e:
            logger.warning(f"Failed to delete document points: {e}")

    async def update_document_path_in_index(self, document_id: UUID, new_path: str) -> None:
        """
        Update path in vector index payloads for a document's chunks.

        Args:
            document_id: Document UUID
            new_path: New file path
        """
        await self._ensure_components()
        
        try:
            await self.storage.update_payload(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ],
                payload={"path": new_path}
            )
            logger.debug(f"Updated path in index for document {document_id}: {new_path}")
        except Exception as e:
            logger.warning(f"Failed to update path in index for document {document_id}: {e}")

    async def update_paths_batch_in_index(self, old_prefix: str, new_prefix: str) -> int:
        """
        Batch update paths in vector index for directory moves.

        Args:
            old_prefix: Old path prefix to replace
            new_prefix: New path prefix

        Returns:
            Number of points updated
        """
        await self._ensure_components()
        
        try:
            # Get all points with paths starting with old_prefix
            results = await self.storage.scroll_points(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="path", match=MatchValue(value=old_prefix + "*")),
                ],
                payload_fields=["path"],
                limit=10000,
            )
            
            if not results:
                logger.debug(f"No points found with path prefix: {old_prefix}")
                return 0
            
            # Update each point's path payload
            updated_count = 0
            for point_data in results:
                if isinstance(point_data, dict) and "path" in point_data:
                    current_path = point_data["path"]
                    if current_path.startswith(old_prefix):
                        new_path = new_prefix + current_path[len(old_prefix):]
                        
                        # Find the point ID for this path
                        point_results = await self.storage.scroll_points(
                            self.collection_name,
                            filter_conditions=[
                                FieldCondition(key="path", match=MatchValue(value=current_path)),
                            ],
                            payload_fields=["path"],
                            limit=1000,
                        )
                        
                        # Update all matching points
                        for point in point_results:
                            await self.storage.update_payload(
                                self.collection_name,
                                filter_conditions=[
                                    FieldCondition(key="path", match=MatchValue(value=current_path)),
                                ],
                                payload={"path": new_path}
                            )
                            updated_count += 1
            
            logger.info(f"Updated {updated_count} points in index: {old_prefix} -> {new_prefix}")
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to batch update paths in index: {e}")
            return 0

    async def close(self) -> None:
        """Close async resources."""
        if self.storage is not None:
            await self.storage.close()
        if self.embedder is not None:
            await self.embedder.close()
