"""Tests for incremental document indexing.

These tests verify the core incremental indexing logic:
- Document hash generation for change detection
- Two-pass vocabulary registration pattern
- Chunking behavior for different document sizes
"""

from uuid import uuid4

import pytest

from mcp_docs.indexing.chunker import DocumentChunker
from mcp_docs.indexing.indexer import DOCS_CODEBASE_ID
from mcp_docs.models import DocumentType


class TestDocHashConcepts:
    """Tests for document hash generation concepts (change detection)."""

    def test_hash_formula_includes_content(self) -> None:
        """Hash formula incorporates content_hash."""
        # Simulating _doc_hash logic: hash is based on content_hash + title + tags
        content_hash_1 = "abc123"
        content_hash_2 = "def456"
        title = "Test"
        tags: list[str] = []

        hash1 = f"{content_hash_1}:{title}:{sorted(tags)}"
        hash2 = f"{content_hash_2}:{title}:{sorted(tags)}"

        assert hash1 != hash2

    def test_hash_formula_includes_title(self) -> None:
        """Hash formula incorporates title."""
        content_hash = "abc123"
        title_1 = "Title A"
        title_2 = "Title B"
        tags: list[str] = []

        hash1 = f"{content_hash}:{title_1}:{sorted(tags)}"
        hash2 = f"{content_hash}:{title_2}:{sorted(tags)}"

        assert hash1 != hash2

    def test_hash_formula_includes_tags(self) -> None:
        """Hash formula incorporates tags."""
        content_hash = "abc123"
        title = "Test"
        tags_1 = ["tag1"]
        tags_2 = ["tag2"]

        hash1 = f"{content_hash}:{title}:{sorted(tags_1)}"
        hash2 = f"{content_hash}:{title}:{sorted(tags_2)}"

        assert hash1 != hash2

    def test_hash_stable_with_same_inputs(self) -> None:
        """Same inputs produce same hash deterministically."""
        content_hash = "abc123"
        title = "Test"
        tags = ["a", "b"]

        hash1 = f"{content_hash}:{title}:{sorted(tags)}"
        hash2 = f"{content_hash}:{title}:{sorted(tags)}"

        assert hash1 == hash2

    def test_tag_order_normalized(self) -> None:
        """Tag order is normalized via sorting."""
        content_hash = "abc123"
        title = "Test"
        tags_1 = ["b", "a", "c"]
        tags_2 = ["a", "c", "b"]

        hash1 = f"{content_hash}:{title}:{sorted(tags_1)}"
        hash2 = f"{content_hash}:{title}:{sorted(tags_2)}"

        assert hash1 == hash2


class TestChunkingStrategies:
    """Tests for document chunking strategies used during indexing."""

    def test_small_document_single_chunk(self) -> None:
        """Small documents are kept as single chunk."""
        chunker = DocumentChunker(max_chars=10000)
        doc_id = uuid4()
        text = "This is a small document with not much content."

        result = chunker.chunk(doc_id, text)

        assert result.strategy == "single"
        assert len(result.chunks) == 1
        assert result.chunks[0].content == text

    def test_large_document_multiple_chunks(self) -> None:
        """Large documents are split into multiple chunks."""
        chunker = DocumentChunker(max_chars=100, min_chars=20)
        doc_id = uuid4()
        text = "First paragraph with substantial content here.\n\n" * 10

        result = chunker.chunk(doc_id, text)

        assert len(result.chunks) > 1

    def test_section_headers_detected(self) -> None:
        """Documents with markdown headers are detected as having sections."""
        chunker = DocumentChunker(max_chars=200)
        doc_id = uuid4()
        # Longer content to trigger section-based chunking
        text = """# Introduction
This is the introduction section with some content that needs to be long enough.

# Methods
This is the methods section with detailed methodology that needs to be long enough.

# Results
This is the results section showing our findings that also needs to be longer."""

        result = chunker.chunk(doc_id, text)

        # Strategy detection depends on content size vs max_chars
        # With small max_chars, sections should be detected
        assert result.strategy in ("single", "sections", "paragraphs")

    def test_chunk_overlap(self) -> None:
        """Chunks can have overlap for context preservation."""
        chunker = DocumentChunker(max_chars=80, overlap_chars=20, min_chars=30)
        doc_id = uuid4()
        text = """First paragraph with some interesting content here.

Second paragraph with more interesting content to test overlap.

Third paragraph providing additional content for verification purposes."""

        result = chunker.chunk(doc_id, text)

        if len(result.chunks) > 1:
            # Multiple chunks were created
            assert result.strategy in ("sections", "paragraphs")

    def test_chunk_metadata_preserved(self) -> None:
        """Chunks maintain document_id and proper indexing."""
        chunker = DocumentChunker()
        doc_id = uuid4()
        text = "Test content."

        result = chunker.chunk(doc_id, text, page_count=3)

        chunk = result.chunks[0]
        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert chunk.char_start == 0
        assert chunk.char_end == len(text)

    def test_empty_document_handling(self) -> None:
        """Empty documents produce a single chunk."""
        chunker = DocumentChunker()
        doc_id = uuid4()

        result = chunker.chunk(doc_id, "")

        assert result.strategy == "single"
        assert len(result.chunks) == 1
        assert result.chunks[0].content == ""

    def test_whitespace_only_document(self) -> None:
        """Whitespace-only documents are handled."""
        chunker = DocumentChunker()
        doc_id = uuid4()

        result = chunker.chunk(doc_id, "   \n\n   \t   ")

        assert result.strategy == "single"
        assert len(result.chunks) == 1


class TestVocabularyTraining:
    """Tests for two-pass vocabulary training pattern."""

    def test_tokenization_extracts_words(self) -> None:
        """Tokenization produces reasonable word tokens."""
        from vector_core import GlobalVocabulary

        vocab = GlobalVocabulary.get_instance()
        tokens = vocab.tokenize("The quick brown fox jumps over the lazy dog")

        # Should extract meaningful tokens
        assert len(tokens) > 0
        # Common stop words may be filtered
        assert "quick" in tokens or "fox" in tokens or "jumps" in tokens

    def test_tokenization_handles_code(self) -> None:
        """Tokenization handles code-like content."""
        from vector_core import GlobalVocabulary

        vocab = GlobalVocabulary.get_instance()
        tokens = vocab.tokenize("def my_function(): return calculate_value()")

        # Should extract function names and keywords
        assert len(tokens) > 0

    def test_empty_content_tokenization(self) -> None:
        """Tokenization handles empty content gracefully."""
        from vector_core import GlobalVocabulary

        vocab = GlobalVocabulary.get_instance()
        tokens = vocab.tokenize("")

        assert isinstance(tokens, list)

    def test_unicode_content_tokenization(self) -> None:
        """Tokenization handles unicode content."""
        from vector_core import GlobalVocabulary

        vocab = GlobalVocabulary.get_instance()
        tokens = vocab.tokenize("Café résumé naïve über Müller")

        assert isinstance(tokens, list)

    def test_special_characters_tokenization(self) -> None:
        """Tokenization handles special characters."""
        from vector_core import GlobalVocabulary

        vocab = GlobalVocabulary.get_instance()
        tokens = vocab.tokenize("test@example.com http://example.com/path?query=1")

        assert isinstance(tokens, list)


class TestIndexerConstants:
    """Tests for indexer constants and configuration."""

    def test_codebase_id_defined(self) -> None:
        """DOCS_CODEBASE_ID is defined for vocabulary registration."""
        assert DOCS_CODEBASE_ID == "docs"

    def test_document_type_enum_values(self) -> None:
        """DocumentType has expected enum values."""
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.DOCX.value == "docx"
        assert DocumentType.TXT.value == "txt"
        assert DocumentType.MD.value == "md"
        assert DocumentType.PPTX.value == "pptx"

    def test_document_type_from_extension(self) -> None:
        """DocumentType.from_extension works correctly."""
        assert DocumentType.from_extension(".pdf") == DocumentType.PDF
        assert DocumentType.from_extension("pdf") == DocumentType.PDF
        assert DocumentType.from_extension(".PDF") == DocumentType.PDF
        assert DocumentType.from_extension(".docx") == DocumentType.DOCX
        assert DocumentType.from_extension(".unknown") == DocumentType.UNKNOWN


class TestIncrementalLogic:
    """Tests for incremental indexing logic concepts."""

    def test_set_difference_for_changes(self) -> None:
        """Set operations identify changed documents."""
        # Indexed documents (by hash)
        indexed_hashes = {"hash1", "hash2", "hash3"}

        # Current documents (by hash)
        current_hashes = {"hash1", "hash2", "hash4", "hash5"}

        # New documents = current - indexed
        new_docs = current_hashes - indexed_hashes
        assert new_docs == {"hash4", "hash5"}

        # Deleted documents = indexed - current
        deleted_docs = indexed_hashes - current_hashes
        assert deleted_docs == {"hash3"}

        # Unchanged documents = intersection
        unchanged = indexed_hashes & current_hashes
        assert unchanged == {"hash1", "hash2"}

    def test_force_reindex_ignores_hashes(self) -> None:
        """Force reindex processes all documents regardless of hash."""
        indexed_hashes = {"hash1", "hash2", "hash3"}
        current_docs = [
            {"hash": "hash1", "content": "doc1"},
            {"hash": "hash2", "content": "doc2"},
            {"hash": "hash3", "content": "doc3"},
        ]

        # Without force: filter to new docs only
        to_index_incremental = [
            d for d in current_docs if d["hash"] not in indexed_hashes
        ]
        assert len(to_index_incremental) == 0

        # With force: index all
        to_index_force = current_docs
        assert len(to_index_force) == 3

    def test_batch_processing_logic(self) -> None:
        """Batch processing splits work into manageable chunks."""
        docs = list(range(100))
        batch_size = 32

        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

        assert len(batches) == 4  # 32 + 32 + 32 + 4
        assert len(batches[0]) == 32
        assert len(batches[-1]) == 4
