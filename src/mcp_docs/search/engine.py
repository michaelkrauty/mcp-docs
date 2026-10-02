"""Hybrid document search engine using Qdrant."""

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from vector_core import (
    EmbeddingClient,
    HybridSearcher,
    QdrantStorage,
)
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_docs.embedding import EmbeddingCollection, embedding_operation
from mcp_docs.models import DocumentNotFoundError
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


def _matched_payload(payload: dict, query: str | None = None) -> dict:
    """Return bounded evidence without changing the retained canonical source."""
    marker = payload.get("embedding_fragment")
    if not marker:
        return payload
    result = dict(payload)
    result["embedding_span"] = {"start": marker["start"], "end": marker["end"]}
    result["evidence_span"] = dict(result["embedding_span"])
    result["evidence_kind"] = "embedding_span"
    if is_derived_fragment(payload):
        return result
    text = stored_embedding_text(payload)
    if text is None:
        raise ValueError("Indexed fragment has no retained source text")
    start, end = marker["start"], marker["end"]
    if query and marker["count"] > 1:
        # Canonical sparse vectors describe the complete retained source, whereas
        # their dense vector describes only the first span. Show tail keyword
        # evidence truthfully without relabeling it as the dense-vector span.
        terms = list(dict.fromkeys(re.findall(r"\w+", query)))
        if terms:
            pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE)
            match = pattern.search(text, end)
            if match:
                start, end = max(0, match.start() - 200), min(len(text), match.end() + 600)
                result["evidence_kind"] = "keyword_excerpt"
                result["evidence_span"] = {"start": start, "end": end}
    result["content"] = text[start:end]
    if isinstance(payload.get("char_start"), int):
        result["char_start"] = payload["char_start"] + start
        result["char_end"] = payload["char_start"] + end
    return result


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
    char_start: int | None = None
    char_end: int | None = None
    embedding_span: dict[str, int] | None = None
    evidence_span: dict[str, int] | None = None
    evidence_kind: str | None = None

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
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.embedding_span is not None:
            result["embedding_span"] = self.embedding_span
            result["evidence_span"] = self.evidence_span
            result["evidence_kind"] = self.evidence_kind
        return result


class DocumentSearchEngine(EmbeddingCollection):
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
        super().__init__()
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
        if active := self.active_collection_name():
            return active
        if self._collection_name is None:
            self._collection_name = settings.collection_name
        return self._collection_name

    @embedding_operation()
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
            include_chunks: If True, return matching passages and auxiliary metadata.
                If False, group content matches into one result per document.

        Returns:
            List of SearchResult ordered by relevance
        """
        await self._ensure_components()

        if not query.strip():
            raise ValueError("Search query cannot be blank")

        # Build filter conditions
        filter_conditions: list[FieldCondition] = [
            FieldCondition(key="type", match=MatchAny(any=["document", "doc_chunk"]))
        ]

        if doc_type:
            filter_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )

        if tags:
            for tag in _normalize_tag_filters(tags):
                filter_conditions.append(FieldCondition(key="tags", match=MatchValue(value=tag)))

        # Generate query vectors
        dense_vector = await self.embedder.embed_single_cached(query, role="query")
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
            group_by=None if include_chunks else "document_id",
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = _matched_payload(result.payload, query)
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
                    char_start=payload.get("char_start"),
                    char_end=payload.get("char_end"),
                    embedding_span=payload.get("embedding_span"),
                    evidence_span=payload.get("evidence_span"),
                    evidence_kind=payload.get("evidence_kind"),
                )
            )

        return search_results

    @embedding_operation()
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

        client = await self.storage.get_client()
        document_filter = FieldCondition(
            key="document_id", match=MatchValue(value=str(document_id))
        )
        chunk_filter = FieldCondition(key="type", match=MatchValue(value="doc_chunk"))
        target_filter = Filter(
            must=[chunk_filter],
            must_not=[document_filter] if exclude_same_document else None,
        )
        best: dict[UUID, tuple[float, int | str]] = {}
        offset = None
        found_source = False
        while True:
            points, offset = await client.scroll(
                self.collection_name,
                scroll_filter=Filter(must=[chunk_filter, document_filter]),
                limit=128,
                offset=offset,
                with_vectors=["dense"],
                with_payload=False,
            )
            for source_point in points:
                found_source = True
                vectors = source_point.vector
                dense = vectors.get("dense") if isinstance(vectors, dict) else vectors
                if not isinstance(dense, list) or not dense:
                    raise RuntimeError(f"Source passage {source_point.id} has no dense vector")
                # Each query requests distinct documents before applying its limit.
                # A document's final score is its strongest passage-pair match.
                grouped = await client.query_points_groups(
                    self.collection_name,
                    query=dense,
                    using="dense",
                    group_by="document_id",
                    group_size=1,
                    limit=limit,
                    query_filter=target_filter,
                    with_payload=False,
                )
                for group in grouped.groups:
                    try:
                        doc_uuid = UUID(str(group.id))
                    except (ValueError, TypeError):
                        continue
                    for point in group.hits:
                        score = point.score or 0.0
                        if doc_uuid in best and best[doc_uuid][0] >= score:
                            continue
                        point_id = str(point.id) if isinstance(point.id, UUID) else point.id
                        best[doc_uuid] = (score, point_id)
                # Only the best requested documents need retaining between queries.
                best = dict(sorted(best.items(), key=lambda item: item[1][0], reverse=True)[:limit])
            if offset is None:
                break
        if not found_source:
            existing, _ = await client.scroll(
                self.collection_name,
                scroll_filter=Filter(
                    must=[
                        document_filter,
                        FieldCondition(key="type", match=MatchAny(any=["document", "doc_chunk"])),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if not existing:
                raise DocumentNotFoundError(f"Document not found in index: {document_id}")
        if not best:
            return []
        # A canonical target may retain a very large body. Hydrate winners once,
        # rather than transferring the same body for every source-vector query.
        retained = await client.retrieve(
            self.collection_name,
            ids=[point_id for _, point_id in best.values()],
            with_payload=True,
            with_vectors=False,
        )
        payloads = {
            str(point.id) if isinstance(point.id, UUID) else point.id: point.payload or {}
            for point in retained
        }
        results = []
        for doc_uuid, (score, point_id) in best.items():
            if point_id not in payloads:
                raise RuntimeError("Similarity result changed during retrieval; retry the search")
            payload = _matched_payload(payloads[point_id])
            results.append(
                SearchResult(
                    document_id=doc_uuid,
                    score=score,
                    content=payload.get("content", ""),
                    point_type=payload.get("type", "unknown"),
                    filename=payload.get("filename", ""),
                    path=payload.get("path", ""),
                    title=payload.get("title"),
                    doc_type=payload.get("doc_type", ""),
                    tags=payload.get("tags", []),
                    chunk_index=payload.get("chunk_index"),
                    section_title=payload.get("section_title"),
                    char_start=payload.get("char_start"),
                    char_end=payload.get("char_end"),
                    embedding_span=payload.get("embedding_span"),
                    evidence_span=payload.get("evidence_span"),
                    evidence_kind=payload.get("evidence_kind"),
                )
            )
        return results

    @embedding_operation()
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
            max_results=0,
        )

        chunks = []
        for payload in results:
            if is_derived_fragment(payload):
                continue
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
                    char_start=payload.get("char_start"),
                    char_end=payload.get("char_end"),
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
