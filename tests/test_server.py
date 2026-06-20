"""Tests for mcp-docs server module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
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
        assert validate_limit(200) == 200  # under default max (100000)

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
        """Test limit is capped at maximum (default max=100000)."""
        from vector_core import validate_limit

        assert validate_limit(50) == 50
        assert validate_limit(100) == 100
        assert validate_limit(150) == 150
        # Test with explicit max
        assert validate_limit(150, maximum=100) == 100
        assert validate_limit(1000, maximum=100) == 100

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
        from mcp_docs.storage.database import DocumentStore
        from mcp_docs.tools.documents import register_document

        # Use a temp DB to avoid polluting the production database
        temp_store = DocumentStore(db_path=temp_dir / "test_register.db")
        with patch("mcp_docs.tools.documents.get_document_store", return_value=temp_store):
            # Create a file with unknown extension
            unknown_file = temp_dir / "test.xyz"
            unknown_file.write_text("test content")

            result = await register_document(path=str(unknown_file))

            # Unknown types are registered with doc_type='unknown'
            assert "error_code" not in result
            assert result["doc_type"] == "unknown"

        temp_store.close()

    @pytest.mark.asyncio
    async def test_register_moved_duplicate_syncs_index_path(
        self, temp_dir: Path
    ) -> None:
        """Re-registering the same content under a new path updates the registry
        path, but the Qdrant payloads still carry the old path. The index must be
        synced, or search returns the dead old path while get_document returns
        the new one."""
        import mcp_docs.tools.documents as documents_mod
        from mcp_docs.storage.database import DocumentStore, compute_file_hash

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            a = temp_dir / "a.txt"
            a.write_text("same content")
            b = temp_dir / "b.txt"
            b.write_text("same content")  # same bytes -> same content hash
            doc = store.register(path=a, content_hash=compute_file_hash(a))

            indexer = AsyncMock()

            async def fake_get_indexer():
                return indexer

            with patch.object(
                documents_mod, "get_document_store", return_value=store
            ), patch.object(
                documents_mod, "get_document_indexer", fake_get_indexer
            ):
                result = await documents_mod.register_document(path=str(b))

            assert result.get("already_registered") is True
            indexer.update_document_path_in_index.assert_awaited_once()
            call = indexer.update_document_path_in_index.await_args
            assert call.args[0] == doc.id
            assert call.args[1] == str(b)
            # Different basename (a.txt -> b.txt): the filename payload is synced
            # too, and the registry filename tracks the new basename.
            indexer.update_document_filename_in_index.assert_awaited_once()
            assert store.read(doc.id).filename == "b.txt"
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_register_moved_same_basename_syncs_path_only(
        self, temp_dir: Path
    ) -> None:
        """A relocation that keeps the basename syncs the path payload but not
        the filename payload."""
        import mcp_docs.tools.documents as documents_mod
        from mcp_docs.storage.database import DocumentStore, compute_file_hash

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            d1 = temp_dir / "d1"
            d1.mkdir()
            a = d1 / "a.txt"
            a.write_text("content")
            d2 = temp_dir / "d2"
            d2.mkdir()
            b = d2 / "a.txt"
            b.write_text("content")  # same content, same basename, new dir
            store.register(path=a, content_hash=compute_file_hash(a))

            indexer = AsyncMock()

            async def fake_get_indexer():
                return indexer

            with patch.object(
                documents_mod, "get_document_store", return_value=store
            ), patch.object(
                documents_mod, "get_document_indexer", fake_get_indexer
            ):
                await documents_mod.register_document(path=str(b))

            indexer.update_document_path_in_index.assert_awaited_once()
            indexer.update_document_filename_in_index.assert_not_awaited()
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_register_same_path_does_not_sync_index(
        self, temp_dir: Path
    ) -> None:
        """Re-registering the exact same path leaves the path unchanged, so the
        index is not touched."""
        import mcp_docs.tools.documents as documents_mod
        from mcp_docs.storage.database import DocumentStore, compute_file_hash

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            a = temp_dir / "a.txt"
            a.write_text("content")
            store.register(path=a, content_hash=compute_file_hash(a))

            indexer = AsyncMock()

            async def fake_get_indexer():
                return indexer

            with patch.object(
                documents_mod, "get_document_store", return_value=store
            ), patch.object(
                documents_mod, "get_document_indexer", fake_get_indexer
            ):
                result = await documents_mod.register_document(path=str(a))

            assert result.get("already_registered") is True
            indexer.update_document_path_in_index.assert_not_awaited()
        finally:
            store.close()


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

    @pytest.mark.asyncio
    async def test_update_tags_syncs_index(self, tmp_path) -> None:
        """Updating tags also updates the vector-index payload, with the
        document's normalized (lowercased) tags."""
        from unittest.mock import AsyncMock, patch

        from mcp_docs.storage.database import DocumentStore
        from mcp_docs.tools import documents as docs_mod

        sample = tmp_path / "doc.txt"
        sample.write_text("content")
        store = DocumentStore(db_path=tmp_path / "tags.db")
        try:
            doc = store.register(sample)
            with (
                patch.object(docs_mod, "get_document_store", return_value=store),
                patch.object(docs_mod, "get_document_indexer") as mock_get_indexer,
            ):
                indexer = AsyncMock()
                mock_get_indexer.return_value = indexer
                result = await docs_mod.update_document_tags(
                    str(doc.id), ["Alpha", "BETA"]
                )

            assert "error_code" not in result
            assert sorted(result["tags"]) == ["alpha", "beta"]
            indexer.update_document_tags_in_index.assert_awaited_once()
            passed_doc = indexer.update_document_tags_in_index.await_args.args[0]
            assert passed_doc.id == doc.id
            assert sorted(passed_doc.tags) == ["alpha", "beta"]
        finally:
            store.close()


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


class TestRemoveDocumentRootDeletion:
    """remove_document_root(delete_documents=True) must clean up the index too.

    Regression test for the bug where it deleted the registry rows but left the
    documents' vectors orphaned in Qdrant (and their fact sources stale).
    """

    @pytest.mark.asyncio
    async def test_delete_documents_removes_index_points_and_sources(
        self, temp_dir: Path
    ) -> None:
        import mcp_docs.tools.documents as documents_mod
        import mcp_docs.tools.roots as roots_mod
        from mcp_docs.storage.database import DocumentStore

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            root_dir = temp_dir / "docs"
            root_dir.mkdir()
            (root_dir / "a.txt").write_text("a")
            (root_dir / "b.txt").write_text("b")
            store.add_root(str(root_dir))
            d1 = store.register(
                path=root_dir / "a.txt", content_hash="hash-a",
                document_root=str(root_dir),
            )
            d2 = store.register(
                path=root_dir / "b.txt", content_hash="hash-b",
                document_root=str(root_dir),
            )

            indexer = AsyncMock()
            integrity = MagicMock()
            integrity.mark_document_deleted.return_value = 1

            async def fake_get_indexer():
                return indexer

            with patch.object(roots_mod, "get_document_store", return_value=store), \
                 patch.object(documents_mod, "get_document_indexer", fake_get_indexer), \
                 patch.object(
                     documents_mod, "get_integrity_manager", return_value=integrity
                 ):
                result = await roots_mod.remove_document_root(
                    str(root_dir), delete_documents=True
                )

            assert result["success"] is True
            # The fix: vector-index points are purged for BOTH documents
            # (the bug left these orphaned in Qdrant).
            purged = {c.args[0] for c in indexer.delete_document_index.call_args_list}
            assert purged == {d1.id, d2.id}
            # Fact sources are marked deleted for both content hashes.
            marked = {c.args[0] for c in integrity.mark_document_deleted.call_args_list}
            assert marked == {"hash-a", "hash-b"}
            # Registry rows and the root are gone.
            assert store.read(d1.id) is None
            assert store.read(d2.id) is None
            assert store.get_root(str(root_dir)) is None
            assert result["documents_deleted"] == 2
            assert result["sources_marked_deleted"] == 2
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_without_delete_keeps_documents_and_index(
        self, temp_dir: Path
    ) -> None:
        import mcp_docs.tools.documents as documents_mod
        import mcp_docs.tools.roots as roots_mod
        from mcp_docs.storage.database import DocumentStore

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            root_dir = temp_dir / "docs"
            root_dir.mkdir()
            (root_dir / "a.txt").write_text("a")
            store.add_root(str(root_dir))
            d1 = store.register(
                path=root_dir / "a.txt", content_hash="hash-a",
                document_root=str(root_dir),
            )

            indexer = AsyncMock()

            async def fake_get_indexer():
                return indexer

            with patch.object(roots_mod, "get_document_store", return_value=store), \
                 patch.object(documents_mod, "get_document_indexer", fake_get_indexer):
                result = await roots_mod.remove_document_root(
                    str(root_dir), delete_documents=False
                )

            assert result["documents_deleted"] is None
            assert result["sources_marked_deleted"] is None
            # Document is kept; nothing purged from the index.
            assert store.read(d1.id) is not None
            indexer.delete_document_index.assert_not_called()
            # Root is still removed.
            assert store.get_root(str(root_dir)) is None
        finally:
            store.close()


class TestDeleteDocumentCleanup:
    """delete_document still removes index points + row via the shared helper."""

    @pytest.mark.asyncio
    async def test_delete_document_purges_index_and_row(self, temp_dir: Path) -> None:
        import mcp_docs.tools.documents as documents_mod
        from mcp_docs.storage.database import DocumentStore

        store = DocumentStore(db_path=temp_dir / "t.db")
        try:
            root_dir = temp_dir / "docs"
            root_dir.mkdir()
            (root_dir / "a.txt").write_text("a")
            store.add_root(str(root_dir))
            doc = store.register(
                path=root_dir / "a.txt", content_hash="hash-a",
                document_root=str(root_dir),
            )

            indexer = AsyncMock()
            integrity = MagicMock()
            integrity.mark_document_deleted.return_value = 3

            async def fake_get_indexer():
                return indexer

            with patch.object(documents_mod, "get_document_store", return_value=store), \
                 patch.object(documents_mod, "get_document_indexer", fake_get_indexer), \
                 patch.object(
                     documents_mod, "get_integrity_manager", return_value=integrity
                 ):
                result = await documents_mod.delete_document(str(doc.id))

            assert result["success"] is True
            assert result["sources_marked_deleted"] == 3
            indexer.delete_document_index.assert_awaited_once_with(doc.id)
            assert store.read(doc.id) is None
        finally:
            store.close()


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


class TestKeywordSearchValidation:
    """keyword_search returns a structured error on bad input instead of crashing.

    Both guards previously referenced ``ErrorCode.VALIDATION_ERROR``, which does
    not exist on vector-core's ``ErrorCode`` enum, so the validation paths raised
    ``AttributeError`` rather than returning a friendly error.
    """

    @pytest.mark.asyncio
    async def test_empty_keyword_returns_validation_error(self) -> None:
        """An empty keyword returns a validation error without touching services."""
        from mcp_docs.tools.search import keyword_search

        result = await keyword_search("")

        assert result["error_code"] == "validation_failed"
        assert "empty" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_no_search_field_returns_validation_error(self) -> None:
        """Disabling both filename and content search returns a validation error
        without initializing the search engine."""
        from mcp_docs.tools.search import keyword_search

        with patch(
            "mcp_docs.tools.search.get_search_engine", new=AsyncMock()
        ) as mock_get_engine:
            result = await keyword_search(
                "term", search_filename=False, search_content=False
            )

        assert result["error_code"] == "validation_failed"
        mock_get_engine.assert_not_called()
