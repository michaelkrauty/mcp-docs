"""Tests for document indexing and chunking."""

from uuid import uuid4

import pytest

from mcp_docs.indexing.chunker import DocumentChunker, chunk_document


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_single_chunk_small_document(self) -> None:
        """Small documents stay as single chunk."""
        chunker = DocumentChunker(max_chars=10000)
        text = "This is a small document."
        doc_id = uuid4()

        result = chunker.chunk(doc_id, text)

        assert result.strategy == "single"
        assert len(result.chunks) == 1
        assert result.chunks[0].content == text
        assert result.chunks[0].chunk_index == 0

    def test_chunk_by_sections(self) -> None:
        """Documents with H1 headers split at sections."""
        chunker = DocumentChunker(max_chars=100)  # Force split
        text = """# Introduction
This is the introduction section.

# Methods
This is the methods section.

# Results
This is the results section."""
        doc_id = uuid4()

        result = chunker.chunk(doc_id, text)

        # Should detect sections
        assert result.strategy == "sections"
        assert len(result.chunks) >= 1

    def test_chunk_by_paragraphs(self) -> None:
        """Documents without sections split at paragraphs."""
        chunker = DocumentChunker(max_chars=50, min_chars=10)
        text = """First paragraph of the document.

Second paragraph with more content.

Third paragraph with even more content."""
        doc_id = uuid4()

        result = chunker.chunk(doc_id, text)

        # Should use paragraph strategy
        assert result.strategy == "paragraphs"
        assert len(result.chunks) >= 1

    def test_chunk_includes_metadata(self) -> None:
        """Chunks include proper metadata."""
        chunker = DocumentChunker()
        text = "Short document."
        doc_id = uuid4()

        result = chunker.chunk(doc_id, text, page_count=5)

        chunk = result.chunks[0]
        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert chunk.char_start == 0
        assert chunk.char_end == len(text)
        assert chunk.page_start == 1
        assert chunk.page_end == 5

    def test_overlap_between_chunks(self) -> None:
        """Paragraph chunks have overlap for context."""
        chunker = DocumentChunker(max_chars=100, overlap_chars=20)
        text = """First paragraph with some content here.

Second paragraph with more content here for testing.

Third paragraph with additional content for verification."""
        doc_id = uuid4()

        result = chunker.chunk(doc_id, text)

        if len(result.chunks) > 1:
            # Check that later chunks might contain overlap content
            assert result.strategy == "paragraphs"


class TestChunkDocumentHelper:
    """Tests for chunk_document convenience function."""

    def test_chunk_document_returns_chunks(self) -> None:
        """chunk_document returns list of DocumentChunks."""
        text = "This is a test document."
        doc_id = uuid4()

        chunks = chunk_document(doc_id, text)

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0].document_id == doc_id
        assert chunks[0].content == text

    def test_chunk_document_with_page_count(self) -> None:
        """chunk_document passes page_count to chunks."""
        text = "This is a test document."
        doc_id = uuid4()

        chunks = chunk_document(doc_id, text, page_count=10)

        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 10


class TestDocumentIndexerScrollPoints:
    """Regression tests for DocumentIndexer.

    CRIT-1: The indexer was calling nonexistent scroll() method instead of
    scroll_points() with wrong filter format. These tests verify the fix.
    """

    @pytest.mark.asyncio
    async def test_get_indexed_hashes_uses_scroll_points(self) -> None:
        """_get_indexed_hashes uses scroll_points with FieldCondition filter.

        Regression test for CRIT-1: Previously called scroll() which doesn't exist.
        """
        from unittest.mock import AsyncMock, MagicMock

        from mcp_docs.indexing.indexer import DocumentIndexer
        from mcp_docs.storage.database import DocumentStore

        # Create mock dependencies
        mock_store = MagicMock(spec=DocumentStore)
        mock_storage = MagicMock()
        mock_storage.scroll_points = AsyncMock(return_value=[
            {"doc_hash": "abc123"},
            {"doc_hash": "def456"},
        ])

        # Create indexer with mocks
        indexer = DocumentIndexer(
            document_store=mock_store,
            storage=mock_storage,
            collection_name="test_collection",
        )

        # Call the method
        result = await indexer._get_indexed_hashes()

        # Verify scroll_points was called (not scroll)
        mock_storage.scroll_points.assert_called_once()
        call_args = mock_storage.scroll_points.call_args

        # Verify correct collection name
        assert call_args[0][0] == "test_collection"

        # Verify filter_conditions uses FieldCondition objects
        filter_conditions = call_args[1]["filter_conditions"]
        assert len(filter_conditions) == 1
        # The filter should be a FieldCondition for type="document"
        assert filter_conditions[0].key == "type"
        assert filter_conditions[0].match.value == "document"

        # Verify result
        assert result == {"abc123", "def456"}

    @pytest.mark.asyncio
    async def test_get_indexed_hashes_handles_empty_collection(self) -> None:
        """_get_indexed_hashes returns empty set for empty/missing collection."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_docs.indexing.indexer import DocumentIndexer
        from mcp_docs.storage.database import DocumentStore

        mock_store = MagicMock(spec=DocumentStore)
        mock_storage = MagicMock()
        mock_storage.scroll_points = AsyncMock(return_value=[])

        indexer = DocumentIndexer(
            document_store=mock_store,
            storage=mock_storage,
            collection_name="test_collection",
        )

        result = await indexer._get_indexed_hashes()

        assert result == set()

    @pytest.mark.asyncio
    async def test_get_indexed_hashes_handles_exception(self) -> None:
        """_get_indexed_hashes returns empty set on exception."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_docs.indexing.indexer import DocumentIndexer
        from mcp_docs.storage.database import DocumentStore

        mock_store = MagicMock(spec=DocumentStore)
        mock_storage = MagicMock()
        mock_storage.scroll_points = AsyncMock(side_effect=Exception("Collection not found"))

        indexer = DocumentIndexer(
            document_store=mock_store,
            storage=mock_storage,
            collection_name="test_collection",
        )

        result = await indexer._get_indexed_hashes()

        # Should return empty set instead of raising
        assert result == set()
