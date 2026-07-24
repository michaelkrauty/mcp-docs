"""Document indexer for Qdrant with hybrid search support."""

import hashlib
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue, PayloadSchemaType, PointStruct
from vector_core import (
    EmbeddingClient,
    GlobalVocabulary,
    QdrantStorage,
    create_hybrid_point_with_key,
)

from mcp_docs.extraction.extractor import extract_content
from mcp_docs.indexing.chunker import DocumentChunker, chunk_document
from mcp_docs.models import (
    Document,
    DocumentChunk,
    DocumentType,
    ExtractionError,
    ExtractionStatus,
)
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

    # ------------------------------------------------------------------
    # Sparse vocabulary accounting
    #
    # The docs corpus contributes to a GlobalVocabulary shared with every other
    # indexed codebase. Two invariants keep sparse retrieval and ranking honest:
    #
    # 1. Every term in an indexed document is in the vocabulary before that
    #    document is vectorized. ``vectorize_document`` silently drops unknown
    #    tokens, so a term registered late never appears in the sparse vector of
    #    the document that introduced it.
    # 2. The contribution describes exactly the currently indexed corpus. Its
    #    document count and document frequencies feed the IDF weights used to
    #    rank results for *every* codebase sharing the database, so a document
    #    counted twice, or never retired, skews ranking well beyond mcp-docs.
    #
    # A document's token set is derived from the text of its points (summary
    # plus chunks) rather than from the raw extracted content. That is the same
    # text the removal side reads back from Qdrant, and symmetry is what keeps
    # repeated edits from leaking document frequency.
    # ------------------------------------------------------------------

    def _token_set(self, texts: Iterable[str]) -> set[str]:
        """Vocabulary token set for one document, given its point texts."""
        tokens: set[str] = set()
        for text in texts:
            tokens.update(self.global_vocab.tokenize(text))
        return tokens

    async def _indexed_token_set(self, document_id: UUID) -> tuple[bool, set[str]]:
        """Tokens currently registered for a document, read back from its points.

        Returns ``(is_indexed, tokens)``. An unindexed document has no points
        and contributes nothing, which is distinct from an indexed document
        whose points happen to hold no usable tokens.
        """
        assert self.storage is not None  # set by _ensure_components
        payloads = await self.storage.scroll_points(
            self.collection_name,
            filter_conditions=[
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            ],
            payload_fields=["content"],
            max_results=0,
        )
        return bool(payloads), self._token_set(p.get("content") or "" for p in payloads)

    def _update_vocabulary(
        self,
        added_tokens: list[set[str]],
        removed_tokens: list[set[str]],
        net_doc_change: int,
    ) -> None:
        """Apply a vocabulary delta, logging rather than raising on failure.

        A delta can only adjust a contribution that already exists:
        ``update_codebase_incremental`` moves the document count with a bare
        ``UPDATE``, which does nothing while the docs corpus has no row yet. The
        first registration therefore goes through ``register_codebase``, which
        establishes the contribution and its document count in one transaction.
        Anything already recorded is replaced, which is what an empty
        contribution wants anyway.

        Indexing must not fail because the shared vocabulary database was busy:
        a missed delta degrades ranking, while a raised error would lose the
        document. ``index_all(force=True)`` rebuilds the contribution from
        scratch and repairs any drift.
        """
        if not added_tokens and not removed_tokens and not net_doc_change:
            return
        try:
            if added_tokens and self.global_vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) <= 0:
                self.global_vocab.register_codebase(DOCS_CODEBASE_ID, added_tokens)
                return
            self.global_vocab.update_codebase_incremental(
                DOCS_CODEBASE_ID,
                added_tokens=added_tokens,
                removed_tokens=removed_tokens,
                net_doc_change=net_doc_change,
            )
        except Exception as e:
            logger.warning(f"Failed to update the docs vocabulary contribution: {e}")

    async def _revert_vocabulary_update(self, document_id: UUID, registered: set[str]) -> None:
        """Restore the contribution for a document whose indexing failed.

        Which half of the replace failed decides what is actually stored: an
        embedding failure leaves the previous points intact, while a failed
        upsert after a successful delete leaves none. Rather than guess, read
        the points back and make the contribution describe them.

        Best-effort: a failure here is logged so the original indexing error is
        the one that propagates.
        """
        try:
            still_indexed, stored = await self._indexed_token_set(document_id)
        except Exception as e:
            logger.warning(
                f"Could not read back document {document_id} to repair its "
                f"vocabulary contribution: {e}"
            )
            return
        actual = [stored] if still_indexed else []
        self._update_vocabulary(
            added_tokens=actual,
            removed_tokens=[registered],
            net_doc_change=len(actual) - 1,
        )

    async def index_document(
        self,
        document_id: UUID,
        content: str,
    ) -> int:
        """
        Index a single document.

        The document's vocabulary contribution is updated before its vectors are
        built, so terms this document introduces are known to the vocabulary in
        time to appear in its own sparse vector.

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

        summary, chunks = self._split_document(document, content)
        added = self._token_set([summary, *(chunk.content for chunk in chunks)])
        was_indexed, previous = await self._indexed_token_set(document_id)

        self._update_vocabulary(
            added_tokens=[added],
            removed_tokens=[previous] if was_indexed else [],
            net_doc_change=0 if was_indexed else 1,
        )

        # Embedding can fail, so preserve the existing index until replacements are ready.
        try:
            points = await self._build_points(document, summary, chunks)
            await self._replace_document_points(document_id, points)
        except Exception:
            await self._revert_vocabulary_update(document_id, added)
            raise

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

        # Index the entire corpus. iter_all() streams every matching document;
        # query()/list_summaries() default to a 50-row limit and would silently
        # truncate a larger backlog.
        #
        # A force rebuild must reindex everything, including documents already in
        # the INDEXED state. That is the production steady state: the worker sets
        # EXTRACTED then auto-indexes to INDEXED, so in steady state no document
        # is left EXTRACTED. iter_all() filters strictly by status, so force must
        # widen the selection to both EXTRACTED and INDEXED; otherwise the work
        # set is empty and the documented bulk-repair path silently no-ops.
        # Incremental (force=False) stays EXTRACTED-only and leans on the per-doc
        # hash filter below to skip unchanged docs without re-extracting the
        # already-indexed corpus.
        selection: ExtractionStatus | set[ExtractionStatus] = (
            {ExtractionStatus.EXTRACTED, ExtractionStatus.INDEXED}
            if force
            else ExtractionStatus.EXTRACTED
        )
        docs_to_index = list(self.document_store.iter_all(extraction_status=selection))

        if not docs_to_index:
            logger.info("No documents to index")
            return {"indexed": 0, "total": self.document_store.count()}

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
            return {"indexed": 0, "total": self.document_store.count(), "skipped": True}

        logger.info(f"Indexing {len(docs_to_index)} documents")

        # Pass 1: Extract content and collect tokens for GlobalVocabulary
        # We re-extract content here since it's not stored in the database
        units: dict[UUID, tuple[str, list[DocumentChunk]]] = {}
        tokens_by_doc: dict[UUID, set[str]] = {}
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
                summary, chunks = self._split_document(doc, extracted.text)
                units[doc.id] = (summary, chunks)

                # Tokenize the text that will actually be stored as points, so
                # a later reindex or delete subtracts exactly what was added.
                tokens = self._token_set([summary, *(chunk.content for chunk in chunks)])
                tokens_by_doc[doc.id] = tokens
                logger.debug(f"Tokenized {doc.filename}: {len(tokens)} unique tokens")

            except ExtractionError as e:
                logger.warning(f"Failed to extract {doc.filename}: {e}")
                extraction_errors.append(f"{doc.filename}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error extracting {doc.filename}: {e}")
                extraction_errors.append(f"{doc.filename}: {e}")

        if not units:
            logger.error("No documents could be extracted for indexing")
            return {
                "indexed": 0,
                "total": self.document_store.count(),
                "errors": extraction_errors,
            }

        # Register the vocabulary before Pass 2 vectorizes anything, so every
        # term in this batch is available to every document in it.
        logger.info(f"Registering vocabulary from {len(tokens_by_doc)} documents")
        await self._register_batch_vocabulary(docs_to_index, tokens_by_doc, force=force)

        # Pass 2: Generate embeddings and create points (summary + chunks)
        total_points = 0
        indexed_count = 0

        for doc in docs_to_index:
            if doc.id not in units:
                continue  # Extraction failed for this doc

            try:
                summary, chunks = units[doc.id]

                # Create both summary AND chunk points (like index_document does)
                points = await self._build_points(doc, summary, chunks)
                await self._replace_document_points(doc.id, points)

                total_points += len(points)
                indexed_count += 1
                logger.debug(f"Indexed {doc.filename}: {len(points)} points")

                # Update status to indexed
                self.document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)

            except Exception as e:
                logger.error(f"Failed to index document {doc.id}: {e}")
                extraction_errors.append(f"{doc.filename}: indexing failed - {e}")
                await self._revert_vocabulary_update(doc.id, tokens_by_doc[doc.id])

        logger.info(f"Indexed {indexed_count} documents with {total_points} points")

        return {
            "indexed": indexed_count,
            "points": total_points,
            "total": self.document_store.count(),
            "errors": extraction_errors if extraction_errors else None,
        }

    async def _register_batch_vocabulary(
        self,
        docs_to_index: list[Document],
        tokens_by_doc: dict[UUID, set[str]],
        force: bool,
    ) -> None:
        """Fold one ``index_all`` batch into the docs vocabulary contribution.

        A force rebuild reindexes the whole corpus, so it replaces the
        contribution outright. Documents whose extraction failed keep their
        existing points, so their stored tokens are carried over rather than
        dropped: silently retiring the contribution of content that is still
        searchable would understate every one of its terms' document frequency.

        An incremental run only sees the documents that changed, so it applies a
        delta instead. Replacing the contribution there would subtract the
        entire unchanged corpus and reset the document count to the size of the
        batch.
        """
        await self._ensure_components()

        if not force:
            added: list[set[str]] = []
            removed: list[set[str]] = []
            new_documents = 0
            for doc in docs_to_index:
                tokens = tokens_by_doc.get(doc.id)
                if tokens is None:
                    continue  # Extraction failed: its points and tokens both stand
                added.append(tokens)
                was_indexed, previous = await self._indexed_token_set(doc.id)
                if was_indexed:
                    removed.append(previous)
                else:
                    new_documents += 1
            self._update_vocabulary(
                added_tokens=added,
                removed_tokens=removed,
                net_doc_change=new_documents,
            )
            return

        tokens_per_doc = [tokens_by_doc[doc.id] for doc in docs_to_index if doc.id in tokens_by_doc]
        for doc in docs_to_index:
            if doc.id in tokens_by_doc:
                continue
            was_indexed, stored = await self._indexed_token_set(doc.id)
            if was_indexed:
                tokens_per_doc.append(stored)

        try:
            self.global_vocab.register_codebase(DOCS_CODEBASE_ID, tokens_per_doc)
        except Exception as e:
            logger.warning(f"Failed to rebuild the docs vocabulary contribution: {e}")

    async def delete_document_index(self, document_id: UUID) -> None:
        """
        Remove a document from the index.

        Args:
            document_id: Document UUID to remove
        """
        was_indexed, tokens = await self._indexed_token_set_safe(document_id)
        await self._delete_document_points(document_id)
        if was_indexed:
            self._update_vocabulary(
                added_tokens=[],
                removed_tokens=[tokens],
                net_doc_change=-1,
            )

    async def _indexed_token_set_safe(self, document_id: UUID) -> tuple[bool, set[str]]:
        """``_indexed_token_set`` for best-effort callers.

        Deletion must proceed even if the tokens cannot be read back; the cost
        is a stale contribution that the next force rebuild repairs, which beats
        leaving the points in place.
        """
        await self._ensure_components()
        try:
            return await self._indexed_token_set(document_id)
        except Exception as e:
            logger.warning(
                f"Could not read document {document_id} back before deleting it "
                f"from the index; its vocabulary contribution is left in place: {e}"
            )
            return False, set()

    def _split_document(
        self,
        document: Document,
        content: str,
    ) -> tuple[str, list[DocumentChunk]]:
        """Split a document into the units that become points.

        Returns the summary text and the chunks that will be stored, so callers
        can derive the document's vocabulary token set without embedding
        anything.
        """
        chunks = chunk_document(document.id, content, document.page_count)

        # Drop empty or whitespace-only chunks so an empty extraction is not
        # embedded as an empty string or stored as a meaningless chunk point;
        # the summary point still keeps the document searchable.
        chunks = [chunk for chunk in chunks if chunk.content.strip()]

        return self._generate_doc_summary(document), chunks

    async def _build_points(
        self,
        document: Document,
        summary: str,
        chunks: list[DocumentChunk],
    ) -> list[PointStruct]:
        """Embed a document's summary and chunks and turn them into points."""
        points: list[PointStruct] = []

        # Prepare texts for batch embedding, summary first
        texts = [summary, *(chunk.content for chunk in chunks)]

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
        # Ensure storage is initialized: this path may run on a cold indexer
        # (e.g. delete_document / remove_document_root right after startup),
        # and without this the missing storage would raise and be swallowed
        # below, silently orphaning the points.
        await self._ensure_components()
        try:
            await self.storage.delete_by_filter(
                self.collection_name,
                field="document_id",
                value=str(document_id),
            )
        except Exception as e:
            logger.warning(f"Failed to delete document points: {e}")

    async def _replace_document_points(
        self,
        document_id: UUID,
        points: list[PointStruct],
    ) -> None:
        """Replace a document's points once the replacements are built.

        Every successfully built document index contains a summary point, so an
        empty result means construction failed and the old index is left alone.

        A failed delete is logged and the upsert still runs. A remote delete has
        an ambiguous outcome: Qdrant may have applied it and only lost the
        response, in which case skipping the upsert would leave the document with
        no points at all. Writing the replacements keeps it searchable either
        way, at the cost of possibly stranding surplus chunk points from a
        shrunken document, which is the lesser failure.
        """
        if not points:
            raise RuntimeError(f"Point construction produced no points for document {document_id}")

        await self._delete_document_points(document_id)
        await self.storage.upsert_batch(self.collection_name, points)

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

    async def update_document_tags_in_index(self, document: Document) -> None:
        """
        Propagate a document's current tags into the vector index.

        Every point carries a ``tags`` payload used to filter searches, and the
        document summary point additionally embeds the tags in its searchable
        content (see ``_generate_doc_summary``). A tag change must update both,
        or the index keeps matching and serving the document's previous tags
        until a full reindex.

        Updates the tags payload on every existing point and, when the document
        is indexed, regenerates the summary point so its content and vectors
        reflect the new tags. This is metadata-only and does not need the
        source file. Best-effort: failures are logged, not raised.

        Args:
            document: The document with its updated tags.
        """
        await self._ensure_components()
        assert self.storage is not None  # set by _ensure_components
        assert self.embedder is not None
        document_id = document.id

        # Tag filter payload on every point (no-op if the document is unindexed).
        try:
            await self.storage.update_payload(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ],
                payload={"tags": document.tags},
            )
        except Exception as e:
            logger.warning(f"Failed to update tags payload for document {document_id}: {e}")

        # The summary point embeds the tags in its content and vectors, so it
        # must be rebuilt. Skip when the document is not indexed, to avoid
        # creating a point for a document that is absent from the index.
        if document.extraction_status != ExtractionStatus.INDEXED:
            return

        try:
            summary = self._generate_doc_summary(document)
            embedding = (await self.embedder.embed_batch([summary]))[0]
            point = self._create_point(
                point_type="document",
                document=document,
                content=summary,
                embedding=embedding,
            )
            await self.storage.upsert_batch(self.collection_name, [point])
            logger.debug(f"Refreshed summary point for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to refresh summary point for document {document_id}: {e}")

    async def update_document_filename_in_index(self, document: Document) -> None:
        """
        Propagate a document's current basename into the vector index.

        Every point carries a ``filename`` payload returned with search results,
        and the document summary point additionally embeds the filename in its
        searchable content (see ``_generate_doc_summary``). A rename must update
        both, or the index keeps reporting and matching the old basename until a
        full reindex.

        Updates the filename payload on every existing point and, when the
        document is indexed, regenerates the summary point so its content and
        vectors reflect the new name. Metadata-only; the source file is not
        needed. Best-effort: failures are logged, not raised.

        Args:
            document: The document with its updated filename.
        """
        await self._ensure_components()
        assert self.storage is not None  # set by _ensure_components
        assert self.embedder is not None
        document_id = document.id

        # filename payload on every point (no-op if the document is unindexed).
        try:
            await self.storage.update_payload(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ],
                payload={"filename": document.filename},
            )
        except Exception as e:
            logger.warning(f"Failed to update filename payload for document {document_id}: {e}")

        # The summary point embeds the filename in its content and vectors, so it
        # must be rebuilt. Skip when the document is not indexed, to avoid
        # creating a point for a document that is absent from the index.
        if document.extraction_status != ExtractionStatus.INDEXED:
            return

        try:
            summary = self._generate_doc_summary(document)
            embedding = (await self.embedder.embed_batch([summary]))[0]
            point = self._create_point(
                point_type="document",
                document=document,
                content=summary,
                embedding=embedding,
            )
            await self.storage.upsert_batch(self.collection_name, [point])
            logger.debug(f"Refreshed summary point for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to refresh summary point for document {document_id}: {e}")

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
            # Scroll all points to find those with matching path prefix.
            # Client-side filtering is required because Qdrant's MatchValue
            # is exact-match only (not wildcard/prefix).
            results = await self.storage.scroll_points(
                self.collection_name,
                payload_fields=["path"],
                limit=10000,
            )

            if not results:
                logger.debug("No points found in collection")
                return 0

            # Collect unique old paths strictly under the directory. The
            # boundary slash keeps a rename of ".../docs" from rewriting
            # points under a sibling like ".../docs2".
            old_dir = old_prefix.rstrip("/") + "/"
            new_dir = new_prefix.rstrip("/") + "/"
            old_paths: set[str] = set()
            for point_data in results:
                if isinstance(point_data, dict):
                    current_path = point_data.get("path", "")
                    if current_path.startswith(old_dir):
                        old_paths.add(current_path)

            if not old_paths:
                logger.debug(f"No points found with path prefix: {old_prefix}")
                return 0

            # Update each unique path with a single bulk update call
            updated_count = 0
            for old_path in old_paths:
                new_path = new_dir + old_path[len(old_dir):]
                await self.storage.update_payload(
                    self.collection_name,
                    filter_conditions=[
                        FieldCondition(key="path", match=MatchValue(value=old_path)),
                    ],
                    payload={"path": new_path}
                )
                updated_count += 1

            logger.info(
                f"Updated {updated_count} unique paths in index: "
                f"{old_prefix} -> {new_prefix}"
            )
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
