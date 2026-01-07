"""Tests for mcp-docs server module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from mcp_docs.models import Document, DocumentStatus, DocumentType, ExtractionStatus


class TestServerImports:
    """Test that server module imports correctly."""

    def test_server_imports(self) -> None:
        """Verify server module can be imported without errors."""
        from mcp_docs.server import mcp

        assert mcp is not None
        assert mcp.name == "mcp-docs"

    def test_validation_helpers_imported(self) -> None:
        """Verify validation helpers are available from vector_core."""
        from vector_core import parse_uuid_or_none, validate_limit

        # Test limit validation
        assert validate_limit(5) == 5
        assert validate_limit(0) == 10  # default
        assert validate_limit(200) == 100  # max

        # Test UUID validation (now from vector_core)
        valid_uuid = parse_uuid_or_none("12345678-1234-5678-1234-567812345678")
        assert isinstance(valid_uuid, UUID)

        invalid_uuid = parse_uuid_or_none("not-a-uuid")
        assert invalid_uuid is None

    def test_singleton_getters_exist(self) -> None:
        """Verify singleton getter functions exist."""
        from mcp_docs.server import (
            get_document_store,
            get_glossary_store,
            get_integrity_manager,
        )

        # These should be callable
        assert callable(get_document_store)
        assert callable(get_glossary_store)
        assert callable(get_integrity_manager)

    def test_settings_imported(self) -> None:
        """Verify settings are properly imported."""
        from mcp_docs.server import settings

        # Settings should have expected attributes
        assert hasattr(settings, "collection_name")
        assert hasattr(settings, "qdrant_url")
        assert hasattr(settings, "embedding_url")


class TestIntegrityManager:
    """Test integrity manager integration."""

    def test_integrity_manager_available(self) -> None:
        """Test that integrity manager is available via vector-core."""
        from mcp_docs.server import get_integrity_manager

        # Facts module is now always available via vector-core
        manager = get_integrity_manager()
        assert manager is not None


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_limit_default(self) -> None:
        """Test default limit value."""
        from vector_core import validate_limit

        # 0 and negative values fall back to default
        assert validate_limit(0) == 10
        assert validate_limit(-1) == 10

    def test_validate_limit_capped(self) -> None:
        """Test limit is capped at maximum."""
        from vector_core import validate_limit

        assert validate_limit(50) == 50
        assert validate_limit(100) == 100
        assert validate_limit(150) == 100
        assert validate_limit(1000) == 100

    def test_validate_limit_negative(self) -> None:
        """Test negative limit falls back to default."""
        from vector_core import validate_limit

        assert validate_limit(-100) == 10
        assert validate_limit(-999) == 10

    def test_validate_uuid_valid(self) -> None:
        """Test valid UUID parsing."""
        from vector_core import parse_uuid_or_none

        test_uuid = uuid4()
        result = parse_uuid_or_none(str(test_uuid))
        assert result == test_uuid

    def test_validate_uuid_valid_with_hyphens(self) -> None:
        """Test valid UUID with hyphens."""
        from vector_core import parse_uuid_or_none

        result = parse_uuid_or_none("12345678-1234-5678-1234-567812345678")
        assert result is not None
        assert str(result) == "12345678-1234-5678-1234-567812345678"

    def test_validate_uuid_invalid(self) -> None:
        """Test invalid UUID returns None."""
        from vector_core import parse_uuid_or_none

        assert parse_uuid_or_none("not-a-uuid") is None
        assert parse_uuid_or_none("12345") is None
        assert parse_uuid_or_none("") is None
        assert parse_uuid_or_none("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz") is None


class TestDocumentStore:
    """Tests for document store singleton."""

    def test_get_document_store_creates_instance(self) -> None:
        """Test that get_document_store creates and returns store."""
        import mcp_docs.server as server_module
        from mcp_docs.server import get_document_store

        # Save original using SyncSingleton API
        original = server_module._document_store.get_if_initialized()

        try:
            # Force new creation by resetting singleton
            server_module._document_store.reset()
            store = get_document_store()
            assert store is not None

            # Should return same instance
            store2 = get_document_store()
            assert store is store2
        finally:
            # Restore original using SyncSingleton API
            if store is not original:
                try:
                    store.close()
                except Exception:
                    pass
            server_module._document_store.set_instance(original)

    def test_get_glossary_store_creates_instance(self) -> None:
        """Test that get_glossary_store creates and returns store."""
        import mcp_docs.server as server_module
        from mcp_docs.server import get_glossary_store

        # Save original using SyncSingleton API
        original = server_module._glossary_store.get_if_initialized()

        try:
            # Force new creation by resetting singleton
            server_module._glossary_store.reset()
            store = get_glossary_store()
            assert store is not None

            # Should return same instance
            store2 = get_glossary_store()
            assert store is store2
        finally:
            # Restore original using SyncSingleton API
            if store is not original:
                try:
                    store.close()
                except Exception:
                    pass
            server_module._glossary_store.set_instance(original)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World! This is a test document.")
    return file_path


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    mock = MagicMock()
    return mock


class TestRegisterDocument:
    """Tests for document registration."""

    @pytest.mark.asyncio
    async def test_register_document_file_not_found(self) -> None:
        """Test registration fails for non-existent file."""
        from mcp_docs.server import register_document

        result = await register_document(path="/nonexistent/file.pdf")

        assert "error_code" in result
        assert "not found" in result["message"].lower() or "does not exist" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_register_document_unknown_type(self, temp_dir: Path) -> None:
        """Test registration handles unknown file type."""
        from mcp_docs.server import register_document

        # Create a file with unknown extension
        unknown_file = temp_dir / "test.xyz"
        unknown_file.write_text("test content")

        result = await register_document(path=str(unknown_file))

        # Unknown types are registered with doc_type='unknown'
        assert "error_code" not in result
        assert result["doc_type"] == "unknown"


class TestGetDocument:
    """Tests for document retrieval."""

    @pytest.mark.asyncio
    async def test_get_document_invalid_uuid(self) -> None:
        """Test get_document with invalid UUID."""
        from mcp_docs.server import get_document

        result = await get_document("not-a-valid-uuid")

        assert "error_code" in result
        assert "invalid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_document_not_found(self) -> None:
        """Test get_document with non-existent document."""
        from mcp_docs.server import get_document

        result = await get_document("12345678-1234-5678-1234-567812345678")

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestGetDocumentByHash:
    """Tests for hash-based document retrieval."""

    @pytest.mark.asyncio
    async def test_get_document_by_hash_not_found(self) -> None:
        """Test get_document_by_hash with non-existent hash."""
        from mcp_docs.server import get_document_by_hash

        result = await get_document_by_hash("nonexistent_hash_value")

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestUpdateDocumentTags:
    """Tests for tag updates."""

    @pytest.mark.asyncio
    async def test_update_tags_invalid_uuid(self) -> None:
        """Test update_document_tags with invalid UUID."""
        from mcp_docs.server import update_document_tags

        result = await update_document_tags("not-a-uuid", ["tag1", "tag2"])

        assert "error_code" in result
        assert "invalid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_update_tags_not_found(self) -> None:
        """Test update_document_tags with non-existent document."""
        from mcp_docs.server import update_document_tags

        result = await update_document_tags(
            "12345678-1234-5678-1234-567812345678",
            ["tag1", "tag2"],
        )

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestDeleteDocument:
    """Tests for document deletion."""

    @pytest.mark.asyncio
    async def test_delete_document_invalid_uuid(self) -> None:
        """Test delete_document with invalid UUID."""
        from mcp_docs.server import delete_document

        result = await delete_document("not-a-uuid")

        assert "error_code" in result
        assert "invalid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self) -> None:
        """Test delete_document with non-existent document."""
        from mcp_docs.server import delete_document

        result = await delete_document("12345678-1234-5678-1234-567812345678")

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestListDocuments:
    """Tests for document listing."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self) -> None:
        """Test list_documents returns empty list when no documents."""
        import mcp_docs.server as server_module
        from mcp_docs.server import list_documents
        from mcp_docs.storage.database import DocumentStore

        # Use a fresh temp database
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original using SyncSingleton API
            original = server_module._document_store.get_if_initialized()
            try:
                # Create fresh store with temp database
                db_path = Path(tmpdir) / "test.db"
                temp_store = DocumentStore(db_path=db_path)
                server_module._document_store.set_instance(temp_store)

                result = await list_documents()
                assert isinstance(result, list)
                # Fresh DB should be empty
                assert len(result) == 0
            finally:
                # Close temp store and restore original
                temp_instance = server_module._document_store.get_if_initialized()
                if temp_instance is not None and temp_instance is not original:
                    temp_instance.close()
                server_module._document_store.set_instance(original)


class TestProcessingStatus:
    """Tests for processing status."""

    @pytest.mark.asyncio
    async def test_get_processing_status_invalid_uuid(self) -> None:
        """Test get_processing_status with invalid UUID."""
        from mcp_docs.server import get_processing_status

        result = await get_processing_status("not-a-uuid")

        assert "error_code" in result
        assert "invalid" in result["message"].lower()


class TestDocumentRoots:
    """Tests for document root management."""

    @pytest.mark.asyncio
    async def test_add_document_root_nonexistent(self) -> None:
        """Test adding non-existent path as root."""
        from mcp_docs.server import add_document_root

        result = await add_document_root(
            path="/nonexistent/directory",
            name="Test Root",
        )

        assert "error_code" in result
        assert "does not exist" in result["message"].lower() or "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_list_document_roots(self) -> None:
        """Test listing document roots."""
        from mcp_docs.server import list_document_roots

        result = await list_document_roots()

        assert isinstance(result, list)
        # Each root should have expected fields
        for root in result:
            assert "path" in root
            assert "name" in root

    @pytest.mark.asyncio
    async def test_get_document_root_not_found(self) -> None:
        """Test getting non-existent document root."""
        from mcp_docs.server import get_document_root

        result = await get_document_root("/nonexistent/path")

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestGlossary:
    """Tests for glossary operations."""

    @pytest.mark.asyncio
    async def test_lookup_term_not_found(self) -> None:
        """Test looking up non-existent term."""
        from mcp_docs.server import lookup_term

        result = await lookup_term("nonexistent_term_xyz_123")

        assert "error_code" in result
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_list_glossary(self) -> None:
        """Test listing glossary entries."""
        from mcp_docs.server import list_glossary

        result = await list_glossary()

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_delete_glossary_entry_not_found(self) -> None:
        """Test deleting non-existent glossary entry."""
        from mcp_docs.server import delete_glossary_entry

        result = await delete_glossary_entry("nonexistent_term_xyz_123")

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestHashLookup:
    """Tests for hash lookup functionality."""

    @pytest.mark.asyncio
    async def test_lookup_hash_not_found(self) -> None:
        """Test lookup_hash with non-existent hash."""
        from mcp_docs.server import lookup_hash

        result = await lookup_hash("sha256:nonexistent")

        # Returns error dict when not found
        assert "error_code" in result
        assert "no document found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_verify_document_reference_not_found(self) -> None:
        """Test verify_document_reference with non-existent hash."""
        from mcp_docs.server import verify_document_reference

        result = await verify_document_reference(
            content_hash="sha256:nonexistent",
            check_file=True,
        )

        # Returns status indicating document is missing
        assert result["status"] == "missing"


class TestBatchVerify:
    """Tests for batch verification."""

    @pytest.mark.asyncio
    async def test_batch_verify_empty(self) -> None:
        """Test batch_verify with empty list."""
        from mcp_docs.server import batch_verify_references

        result = await batch_verify_references([])

        # Returns empty list for empty input
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_batch_verify_nonexistent(self) -> None:
        """Test batch_verify with non-existent hashes."""
        from mcp_docs.server import batch_verify_references

        result = await batch_verify_references([
            "sha256:nonexistent1",
            "sha256:nonexistent2",
        ])

        # Returns list of verification results
        assert isinstance(result, list)
        assert len(result) == 2
        # All should have status 'missing'
        for item in result:
            assert item["status"] == "missing"
