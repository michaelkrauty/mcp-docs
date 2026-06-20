"""Hybrid document search engine using Qdrant."""

import logging
from dataclasses import dataclass
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_core import (
    EmbeddingClient,
    HybridSearcher,
    QdrantStorage,
)
from vector_core.embeddings.global_vocab import GlobalVocabulary

from mcp_docs.models import DocumentNotFoundError
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


def _normalize_tag_filters(tags: list[str]) -> list[str]:
    """Normalize tag filters to match how tags are stored.

    DocumentStore lowercases and strips every tag on write, so a filter
    must do the same or a wrong-case tag (e.g. "Finance") silently
    matches nothing. Blank entries are dropped rather than matched.
    """
    normalized = (tag.lower().strip() for tag in tags)
    return [tag for tag in normalized if tag]


@dataclass
class SearchResult:
    """A single search result."""

    document_id: UUID
    score: float
    content: str
    point_type: str  # "document" or "doc_chunk"
    filename: str
    path: str
    title: str | None
    doc_type: str
    tags: list[str]
    chunk_index: int | None = None
    section_title: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "document_id": str(self.document_id),
            "score": round(self.score, 4),
            "content": self.content,
            "point_type": self.point_type,
            "filename": self.filename,
            "path": self.path,
            "title": self.title,
            "doc_type": self.doc_type,
            "tags": self.tags,
        }
        if self.chunk_index is not None:
            result["chunk_index"] = self.chunk_index
        if self.section_title:
            result["section_title"] = self.section_title
        return result


class DocumentSearchEngine:
    """
    Hybrid search engine for documents.

    Uses dense+sparse vectors for semantic search with keyword boosting.
    """

    def __init__(
        self,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize search engine.

        Args:
            storage: QdrantStorage instance (created if not provided)
            embedder: EmbeddingClient instance (created if not provided)
            global_vocab: GlobalVocabulary instance (created if not provided)
            collection_name: Qdrant collection name (from settings if not provided)
        """
        self.storage = storage
        self.embedder = embedder
        self._global_vocab = global_vocab
        self._collection_name = collection_name
        self._searcher: HybridSearcher | None = None

    @property
    def global_vocab(self) -> GlobalVocabulary:
        """Get GlobalVocabulary instance.

        Raises:
            RuntimeError: If accessed before async initialization.
        """
        if self._global_vocab is None:
            raise RuntimeError(
                "GlobalVocabulary not initialized. Call await _ensure_components() first."
            )
        return self._global_vocab

    async def _ensure_components(self) -> None:
        """Ensure async components are initialized."""
        if self.storage is None:
            self.storage = QdrantStorage()
        if self.embedder is None:
            self.embedder = EmbeddingClient()
        if self._global_vocab is None:
            self._global_vocab = GlobalVocabulary.get_instance()
        if self._searcher is None:
            self._searcher = HybridSearcher(storage=self.storage)

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        if self._collection_name is None:
            self._collection_name = settings.collection_name
        return self._collection_name

    async def search(
        self,
        query: str,
        limit: int = 10,
        doc_type: str | None = None,
        tags: list[str] | None = None,
        include_chunks: bool = True,
    ) -> list[SearchResult]:
        """
        Search documents with hybrid search.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            doc_type: Filter by document type
            tags: Filter by tags (document must have ALL tags)
            include_chunks: If True, search chunks too. If False, only document summaries.

        Returns:
            List of SearchResult ordered by relevance
        """
        await self._ensure_components()

        # Build filter conditions
        filter_conditions: list[FieldCondition] = []

        if not include_chunks:
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="document"))
            )

        if doc_type:
            filter_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )

        if tags:
            for tag in _normalize_tag_filters(tags):
                filter_conditions.append(
                    FieldCondition(key="tags", match=MatchValue(value=tag))
                )

        # Generate query vectors
        dense_vector = await self.embedder.embed_single_cached(query)
        sparse_vector = self.global_vocab.vectorize_query(query)

        # Perform hybrid search using HybridSearcher
        if self._searcher is None:
            raise RuntimeError("Search engine not initialized. Call _ensure_components() first.")
        results = await self._searcher.search(
            collection=self.collection_name,
            dense_query=dense_vector,
            sparse_query=sparse_vector,
            filter_conditions=filter_conditions if filter_conditions else None,
            limit=limit,
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = result.payload
            try:
                doc_id = UUID(payload.get("document_id", ""))
            except (ValueError, TypeError):
                logger.warning(f"Invalid document_id in payload: {payload.get('document_id')}")
                continue

            search_results.append(
                SearchResult(
                    document_id=doc_id,
                    score=result.score,
                    content=payload.get("content", ""),
                    point_type=payload.get("type", "unknown"),
                    filename=payload.get("filename", ""),
                    path=payload.get("path", ""),
                    title=payload.get("title"),
                    doc_type=payload.get("doc_type", ""),
                    tags=payload.get("tags", []),
                    chunk_index=payload.get("chunk_index"),
                    section_title=payload.get("section_title"),
                )
            )

        return search_results

    async def find_similar(
        self,
        document_id: UUID,
        limit: int = 5,
        exclude_same_document: bool = True,
    ) -> list[SearchResult]:
        """
        Find documents similar to a given document.

        Args:
            document_id: Document to find similar content for
            limit: Maximum results to return
            exclude_same_document: If True, exclude chunks from same document

        Returns:
            List of similar SearchResults. An empty list means the document is
            indexed but has no similar neighbors (for example a single-document
            collection); it does not mean the document is missing.

        Raises:
            DocumentNotFoundError: If the source document is not in the index.
        """
        await self._ensure_components()
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call _ensure_components() first.")

        # First, get the document's summary point using scroll_points
        results = await self.storage.scroll_points(
            self.collection_name,
            filter_conditions=[
                FieldCondition(key="type", match=MatchValue(value="document")),
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            ],
            limit=1,
        )

        if not results:
            raise DocumentNotFoundError(f"Document not found in index: {document_id}")

        # Get the dense vector - we need to retrieve the point with vectors
        # Use the Qdrant client directly for vector retrieval
        client = await self.storage.get_client()

        # Get the point ID from scroll - we need to look up by filter
        scroll_result = await client.scroll(
            self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="document")),
                    FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
                ]
            ),
            limit=1,
            with_vectors=True,
        )

        points, _ = scroll_result
        if not points:
            raise DocumentNotFoundError(f"Document not found in index: {document_id}")

        source_point = points[0]

        # Get the dense vector from the source point
        vectors = source_point.vector
        if isinstance(vectors, dict):
            dense_vector = vectors.get("dense", [])
        else:
            dense_vector = vectors or []

        if not dense_vector:
            logger.warning(f"No dense vector for document: {document_id}")
            return []

        # Build filter conditions for similar search
        filter_conditions: list[FieldCondition] = [
            FieldCondition(key="type", match=MatchValue(value="document"))
        ]

        # Perform vector search using query_dense
        similar = await self.storage.query_dense(
            collection=self.collection_name,
            query_vector=dense_vector,
            filter_conditions=filter_conditions,
            limit=limit + (1 if exclude_same_document else 0),
        )

        # Convert to SearchResult objects
        search_results = []
        for point in similar:
            payload = point.payload or {}
            result_doc_id = payload.get("document_id", "")

            # Skip same document if requested
            if exclude_same_document and result_doc_id == str(document_id):
                continue

            try:
                doc_uuid = UUID(result_doc_id)
            except (ValueError, TypeError):
                continue

            search_results.append(
                SearchResult(
                    document_id=doc_uuid,
                    score=point.score or 0.0,
                    content=payload.get("content", ""),
                    point_type=payload.get("type", "unknown"),
                    filename=payload.get("filename", ""),
                    path=payload.get("path", ""),
                    title=payload.get("title"),
                    doc_type=payload.get("doc_type", ""),
                    tags=payload.get("tags", []),
                    chunk_index=payload.get("chunk_index"),
                    section_title=payload.get("section_title"),
                )
            )

            if len(search_results) >= limit:
                break

        return search_results

    async def get_document_chunks(
        self,
        document_id: UUID,
    ) -> list[SearchResult]:
        """
        Get all indexed chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks ordered by chunk_index
        """
        await self._ensure_components()
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call _ensure_components() first.")

        results = await self.storage.scroll_points(
            self.collection_name,
            filter_conditions=[
                FieldCondition(key="type", match=MatchValue(value="doc_chunk")),
                FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            ],
            limit=1000,
        )

        chunks = []
        for payload in results:
            try:
                doc_uuid = UUID(payload.get("document_id", ""))
            except (ValueError, TypeError):
                continue

            chunks.append(
                SearchResult(
                    document_id=doc_uuid,
                    score=1.0,  # No score for scroll
                    content=payload.get("content", ""),
                    point_type=payload.get("type", "unknown"),
                    filename=payload.get("filename", ""),
                    path=payload.get("path", ""),
                    title=payload.get("title"),
                    doc_type=payload.get("doc_type", ""),
                    tags=payload.get("tags", []),
                    chunk_index=payload.get("chunk_index"),
                    section_title=payload.get("section_title"),
                )
            )

        # Sort by chunk index
        chunks.sort(key=lambda c: c.chunk_index or 0)
        return chunks

    async def close(self) -> None:
        """Close async resources."""
        if self.storage is not None:
            await self.storage.close()
        if self.embedder is not None:
            await self.embedder.close()
