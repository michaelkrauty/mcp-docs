"""Tests for search engine functionality."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from mcp_docs.search.engine import DocumentSearchEngine, SearchResult


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_to_dict_full(self) -> None:
        """SearchResult.to_dict includes all fields."""
        result = SearchResult(
            document_id=uuid4(),
            score=0.95,
            content="Test content",
            point_type="document",
            filename="test.pdf",
            path="/docs/test.pdf",
            title="Test Document",
            doc_type="pdf",
            tags=["tag1", "tag2"],
            chunk_index=0,
            section_title="Introduction",
        )

        d = result.to_dict()

        assert "document_id" in d
        assert d["score"] == 0.95
        assert d["content"] == "Test content"
        assert d["point_type"] == "document"
        assert d["filename"] == "test.pdf"
        assert d["path"] == "/docs/test.pdf"
        assert d["title"] == "Test Document"
        assert d["doc_type"] == "pdf"
        assert d["tags"] == ["tag1", "tag2"]
        assert d["chunk_index"] == 0
        assert d["section_title"] == "Introduction"

    def test_to_dict_optional_fields(self) -> None:
        """SearchResult.to_dict excludes None optional fields."""
        result = SearchResult(
            document_id=uuid4(),
            score=0.8,
            content="Content",
            point_type="document",
            filename="doc.txt",
            path="/doc.txt",
            title=None,
            doc_type="text",
            tags=[],
            chunk_index=None,
            section_title=None,
        )

        d = result.to_dict()

        assert "chunk_index" not in d
        assert "section_title" not in d
        assert d["title"] is None


class TestDocumentSearchEngineInit:
    """Tests for DocumentSearchEngine initialization."""

    def test_init_defaults(self) -> None:
        """Engine initializes with all None defaults."""
        engine = DocumentSearchEngine()

        assert engine.storage is None
        assert engine.embedder is None
        assert engine._global_vocab is None
        assert engine._collection_name is None
        assert engine._searcher is None

    def test_init_with_collection_name(self) -> None:
        """Engine accepts explicit collection name."""
        engine = DocumentSearchEngine(collection_name="test_collection")

        assert engine.collection_name == "test_collection"

    def test_global_vocab_property_raises_before_init(self) -> None:
        """Accessing global_vocab before init raises RuntimeError."""
        engine = DocumentSearchEngine()

        with pytest.raises(RuntimeError, match="GlobalVocabulary not initialized"):
            _ = engine.global_vocab


class TestDocumentSearchEngineSearch:
    """Tests for DocumentSearchEngine.search method."""

    @pytest.mark.asyncio
    async def test_search_builds_filter_conditions(self) -> None:
        """Search builds correct filter conditions from parameters."""
        engine = DocumentSearchEngine(collection_name="test")

        # Mock all dependencies
        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_single_cached.return_value = [0.1] * 1024

        mock_vocab = MagicMock()
        mock_vocab.vectorize_query.return_value = MagicMock(indices=[1], values=[0.5])

        mock_searcher = AsyncMock()
        mock_searcher.search.return_value = []

        engine.storage = mock_storage
        engine.embedder = mock_embedder
        engine._global_vocab = mock_vocab
        engine._searcher = mock_searcher

        # Call search with filters
        await engine.search(
            query="test query",
            doc_type="pdf",
            tags=["tag1"],
            include_chunks=False,
        )

        # Verify searcher was called
        mock_searcher.search.assert_called_once()
        call_kwargs = mock_searcher.search.call_args.kwargs

        # Check filter conditions
        conditions = call_kwargs.get("filter_conditions", [])
        assert len(conditions) == 3  # type=document, doc_type, tag

    @pytest.mark.asyncio
    async def test_search_uses_vectorize_query(self) -> None:
        """Search uses vectorize_query not vectorize_document for queries."""
        engine = DocumentSearchEngine(collection_name="test")

        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_single_cached.return_value = [0.1] * 1024

        mock_vocab = MagicMock()
        mock_vocab.vectorize_query.return_value = MagicMock(indices=[1], values=[0.5])
        mock_vocab.vectorize_document.return_value = MagicMock(indices=[2], values=[0.3])

        mock_searcher = AsyncMock()
        mock_searcher.search.return_value = []

        engine.storage = mock_storage
        engine.embedder = mock_embedder
        engine._global_vocab = mock_vocab
        engine._searcher = mock_searcher

        await engine.search(query="test query")

        # Should use vectorize_query, not vectorize_document
        mock_vocab.vectorize_query.assert_called_once_with("test query")
        mock_vocab.vectorize_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_converts_results(self) -> None:
        """Search correctly converts HybridSearcher results."""
        engine = DocumentSearchEngine(collection_name="test")

        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_single_cached.return_value = [0.1] * 1024

        mock_vocab = MagicMock()
        mock_vocab.vectorize_query.return_value = MagicMock(indices=[1], values=[0.5])

        # Create mock result
        doc_id = uuid4()
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.payload = {
            "document_id": str(doc_id),
            "content": "Test content",
            "type": "document",
            "filename": "test.pdf",
            "path": "/test.pdf",
            "title": "Test",
            "doc_type": "pdf",
            "tags": ["test"],
        }

        mock_searcher = AsyncMock()
        mock_searcher.search.return_value = [mock_result]

        engine.storage = mock_storage
        engine.embedder = mock_embedder
        engine._global_vocab = mock_vocab
        engine._searcher = mock_searcher

        results = await engine.search(query="test")

        assert len(results) == 1
        assert results[0].document_id == doc_id
        assert results[0].score == 0.85
        assert results[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_search_skips_invalid_document_ids(self) -> None:
        """Search skips results with invalid document IDs."""
        engine = DocumentSearchEngine(collection_name="test")

        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_single_cached.return_value = [0.1] * 1024

        mock_vocab = MagicMock()
        mock_vocab.vectorize_query.return_value = MagicMock(indices=[1], values=[0.5])

        # Create mock result with invalid document_id
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.payload = {
            "document_id": "not-a-valid-uuid",
            "content": "Test",
            "type": "document",
        }

        mock_searcher = AsyncMock()
        mock_searcher.search.return_value = [mock_result]

        engine.storage = mock_storage
        engine.embedder = mock_embedder
        engine._global_vocab = mock_vocab
        engine._searcher = mock_searcher

        results = await engine.search(query="test")

        # Invalid UUID should be skipped
        assert len(results) == 0


class TestDocumentSearchEngineGetChunks:
    """Tests for get_document_chunks method."""

    @pytest.mark.asyncio
    async def test_get_chunks_sorts_by_index(self) -> None:
        """Chunks are returned sorted by chunk_index."""
        engine = DocumentSearchEngine(collection_name="test")

        mock_storage = AsyncMock()
        doc_id = uuid4()

        # Return chunks out of order
        mock_storage.scroll_points.return_value = [
            {"document_id": str(doc_id), "chunk_index": 2, "type": "doc_chunk", "content": "C", "filename": "f", "path": "p", "doc_type": "t", "tags": []},
            {"document_id": str(doc_id), "chunk_index": 0, "type": "doc_chunk", "content": "A", "filename": "f", "path": "p", "doc_type": "t", "tags": []},
            {"document_id": str(doc_id), "chunk_index": 1, "type": "doc_chunk", "content": "B", "filename": "f", "path": "p", "doc_type": "t", "tags": []},
        ]

        engine.storage = mock_storage
        engine.embedder = AsyncMock()
        engine._global_vocab = MagicMock()

        chunks = await engine.get_document_chunks(doc_id)

        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2
        assert chunks[0].content == "A"
        assert chunks[1].content == "B"
        assert chunks[2].content == "C"


class TestTagFilterNormalization:
    """Tag filters must be normalized like stored tags (lowercase+strip)."""

    def test_mixed_case_and_whitespace_normalized(self):
        from mcp_docs.search.engine import _normalize_tag_filters

        assert _normalize_tag_filters(["Finance", "  Tax  ", "ok"]) == [
            "finance",
            "tax",
            "ok",
        ]

    def test_blank_entries_dropped(self):
        from mcp_docs.search.engine import _normalize_tag_filters

        assert _normalize_tag_filters(["", "   ", "real"]) == ["real"]


class TestDocumentSearchEngineFindSimilar:
    """find_similar must distinguish a source document that is absent from the
    index (raise DocumentNotFoundError) from one that is indexed but simply has
    no similar neighbors (return an empty list)."""

    @pytest.mark.asyncio
    async def test_raises_when_source_not_indexed(self) -> None:
        from mcp_docs.models import DocumentNotFoundError

        engine = DocumentSearchEngine(collection_name="test")
        mock_storage = AsyncMock()
        mock_storage.scroll_points.return_value = []  # summary point not in index
        engine.storage = mock_storage

        with pytest.raises(DocumentNotFoundError):
            await engine.find_similar(document_id=uuid4())

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_neighbors(self) -> None:
        """An indexed document with no other similar documents returns []."""
        doc_id = uuid4()
        engine = DocumentSearchEngine(collection_name="test")

        mock_storage = AsyncMock()
        # source summary point exists
        mock_storage.scroll_points.return_value = [MagicMock()]
        # the dense-vector lookup returns the source point with a usable vector
        source_point = MagicMock()
        source_point.vector = {"dense": [0.1, 0.2, 0.3]}
        mock_client = AsyncMock()
        mock_client.scroll.return_value = ([source_point], None)
        mock_storage.get_client.return_value = mock_client
        # the only candidate is the same document, excluded -> no neighbors
        same = MagicMock()
        same.payload = {"document_id": str(doc_id), "type": "document"}
        same.score = 1.0
        mock_storage.query_dense.return_value = [same]
        engine.storage = mock_storage

        results = await engine.find_similar(document_id=doc_id, exclude_same_document=True)
        assert results == []


class TestFindSimilarDocumentsTool:
    """find_similar_documents must return an empty list for an indexed document
    with no neighbors, and only return NOT_FOUND when the document is genuinely
    absent from the index."""

    @pytest.mark.asyncio
    async def test_no_neighbors_returns_empty_list(self) -> None:
        from mcp_docs.tools.search import find_similar_documents

        engine = AsyncMock()
        engine.find_similar.return_value = []
        with patch(
            "mcp_docs.tools.search.get_search_engine", new=AsyncMock(return_value=engine)
        ):
            result = await find_similar_documents(str(uuid4()))
        assert result == []

    @pytest.mark.asyncio
    async def test_not_indexed_returns_not_found(self) -> None:
        from vector_core.errors import ErrorCode

        from mcp_docs.models import DocumentNotFoundError
        from mcp_docs.tools.search import find_similar_documents

        engine = AsyncMock()
        engine.find_similar.side_effect = DocumentNotFoundError("nope")
        with patch(
            "mcp_docs.tools.search.get_search_engine", new=AsyncMock(return_value=engine)
        ):
            result = await find_similar_documents(str(uuid4()))
        assert isinstance(result, dict)
        assert result["error_code"] == ErrorCode.NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_returns_neighbor_dicts(self) -> None:
        from mcp_docs.tools.search import find_similar_documents

        neighbor = SearchResult(
            document_id=uuid4(), score=0.9, content="c", point_type="document",
            filename="n.txt", path="/n.txt", title=None, doc_type="txt", tags=[],
        )
        engine = AsyncMock()
        engine.find_similar.return_value = [neighbor]
        with patch(
            "mcp_docs.tools.search.get_search_engine", new=AsyncMock(return_value=engine)
        ):
            result = await find_similar_documents(str(uuid4()))
        assert isinstance(result, list)
        assert result[0]["filename"] == "n.txt"
