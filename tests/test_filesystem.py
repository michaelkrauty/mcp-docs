"""Tests for filesystem management tools."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from vector_core.errors import ErrorCode

from mcp_docs.models import DocumentStatus, ExtractionStatus
from mcp_docs.storage.database import DocumentStore
from mcp_docs.tools.filesystem import (
    create_directory,
    delete_directory,
    move_directory,
    move_file,
    rename_directory,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir: Path):
    """Create a temporary DocumentStore."""
    db_path = temp_dir / "test_docs.db"
    store = DocumentStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def doc_root(temp_dir: Path, store: DocumentStore) -> Path:
    """Create a document root for testing."""
    root_path = temp_dir / "documents"
    root_path.mkdir()
    store.add_root(str(root_path), name="Test Root")
    return root_path


@pytest.fixture
def sample_file(doc_root: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = doc_root / "sample.txt"
    file_path.write_text("Hello, World! This is a test document.")
    return file_path


@pytest.fixture
def registered_doc(store: DocumentStore, sample_file: Path):
    """Register a document for testing."""
    doc = store.register(sample_file)
    return doc


@pytest.fixture
def mock_processor():
    """Mock DocumentProcessor."""
    with patch("mcp_docs.tools.filesystem.get_document_processor") as mock:
        processor = AsyncMock()
        processor.wait_for_documents = AsyncMock(return_value=True)
        mock.return_value = processor
        yield processor


@pytest.fixture
def mock_indexer():
    """Mock DocumentIndexer."""
    with patch("mcp_docs.tools.filesystem.get_document_indexer") as mock:
        indexer = AsyncMock()
        indexer.update_document_path_in_index = AsyncMock()
        indexer.update_document_filename_in_index = AsyncMock()
        indexer.update_paths_batch_in_index = AsyncMock(return_value=5)
        mock.return_value = indexer
        yield indexer


@pytest.fixture
def mock_store(store):
    """Mock get_document_store to return our test store."""
    with patch("mcp_docs.tools.filesystem.get_document_store") as mock:
        mock.return_value = store
        yield store


class TestMoveFile:
    """Tests for move_file tool."""

    @pytest.mark.asyncio
    async def test_move_file_success(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """Test basic move file within same root."""
        # Setup
        source_path = str(registered_doc.path)
        dest_path = str(doc_root / "moved_sample.txt")
        
        # Execute
        result = await move_file(source_path, dest_path)
        
        # Verify
        assert result["success"] is True
        assert result["old_path"] == source_path
        assert result["new_path"] == dest_path
        assert result["document_id"] == str(registered_doc.id)
        
        # Check file actually moved
        assert not Path(source_path).exists()
        assert Path(dest_path).exists()
        
        # Check database updated
        updated_doc = mock_store.read(registered_doc.id)
        assert updated_doc.path == dest_path
        
        # Check processor and indexer called
        mock_processor.wait_for_documents.assert_called_once_with(
            [registered_doc.id], timeout=60.0, require_completed=False
        )
        mock_indexer.update_document_path_in_index.assert_called_once_with(registered_doc.id, dest_path)

    @pytest.mark.asyncio
    async def test_move_file_cross_root(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        temp_dir: Path,
    ):
        """Test move file between roots, verify document_root updated."""
        # Create second root
        root2_path = temp_dir / "documents2"
        root2_path.mkdir()
        mock_store.add_root(str(root2_path), name="Test Root 2")
        
        source_path = str(registered_doc.path)
        dest_path = str(root2_path / "moved_sample.txt")
        
        # Execute
        result = await move_file(source_path, dest_path)
        
        # Verify
        assert result["success"] is True
        assert result["new_document_root"] == str(root2_path)
        
        # Check database updated with new root
        updated_doc = mock_store.read(registered_doc.id)
        assert updated_doc.document_root == str(root2_path)

    @pytest.mark.asyncio
    async def test_move_file_not_registered(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        doc_root: Path,
    ):
        """Test error for unregistered file."""
        # Create unregistered file
        unregistered_file = doc_root / "unregistered.txt"
        unregistered_file.write_text("unregistered content")
        
        source_path = str(unregistered_file)
        dest_path = str(doc_root / "moved_unregistered.txt")
        
        # Execute
        result = await move_file(source_path, dest_path)
        
        # Verify error
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.NOT_FOUND.value
        assert "not registered" in result["message"]

    @pytest.mark.asyncio
    async def test_move_file_destination_exists(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """Test error when destination exists."""
        # Create destination file
        dest_file = doc_root / "existing.txt"
        dest_file.write_text("existing content")
        
        source_path = str(registered_doc.path)
        dest_path = str(dest_file)
        
        # Execute
        result = await move_file(source_path, dest_path)
        
        # Verify error
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.CONFLICT.value
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_move_file_updates_vector_index(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """Test that vector index path is updated."""
        source_path = str(registered_doc.path)
        dest_path = str(doc_root / "moved_sample.txt")
        
        await move_file(source_path, dest_path)

        # Verify index update called
        mock_indexer.update_document_path_in_index.assert_called_once_with(
            registered_doc.id, dest_path
        )

    @pytest.mark.asyncio
    async def test_move_file_rename_updates_filename(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """A rename-move updates the stored basename, not only the path."""
        source_path = str(registered_doc.path)
        dest_path = str(doc_root / "renamed.txt")

        result = await move_file(source_path, dest_path)

        assert result["success"] is True
        updated_doc = mock_store.read(registered_doc.id)
        assert updated_doc.filename == "renamed.txt"

    @pytest.mark.asyncio
    async def test_move_file_rename_resyncs_filename_in_index(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """A basename change resyncs the filename payload/summary in the index."""
        source_path = str(registered_doc.path)
        dest_path = str(doc_root / "renamed.txt")

        await move_file(source_path, dest_path)

        mock_indexer.update_document_filename_in_index.assert_called_once()
        passed_doc = mock_indexer.update_document_filename_in_index.call_args.args[0]
        assert passed_doc.filename == "renamed.txt"

    @pytest.mark.asyncio
    async def test_move_file_same_name_skips_filename_resync(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        temp_dir: Path,
    ):
        """A pure relocation (same basename) leaves the filename untouched."""
        root2 = temp_dir / "documents2"
        root2.mkdir()
        mock_store.add_root(str(root2), name="Test Root 2")
        source_path = str(registered_doc.path)
        dest_path = str(root2 / "sample.txt")  # same basename, different directory

        await move_file(source_path, dest_path)

        mock_indexer.update_document_filename_in_index.assert_not_called()
        updated_doc = mock_store.read(registered_doc.id)
        assert updated_doc.filename == "sample.txt"

    @pytest.mark.asyncio
    async def test_move_file_processing_timeout(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        registered_doc,
        doc_root: Path,
    ):
        """Test error when processing doesn't complete in time."""
        # Setup processor to timeout
        mock_processor.wait_for_documents.return_value = False
        
        source_path = str(registered_doc.path)
        dest_path = str(doc_root / "moved_sample.txt")
        
        # Execute
        result = await move_file(source_path, dest_path)
        
        # Verify error
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.TIMEOUT.value


class TestCreateDirectory:
    """Tests for create_directory tool."""

    @pytest.mark.asyncio
    async def test_create_directory_basic(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test basic directory creation."""
        dir_path = str(doc_root / "new_dir")
        
        result = await create_directory(dir_path)
        
        assert result["success"] is True
        assert result["path"] == dir_path
        assert result["created"] is True
        assert Path(dir_path).exists()
        assert Path(dir_path).is_dir()

    @pytest.mark.asyncio
    async def test_create_directory_parents(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test directory creation with parent directories."""
        dir_path = str(doc_root / "parent" / "child" / "grandchild")
        
        result = await create_directory(dir_path, parents=True)
        
        assert result["success"] is True
        assert result["created"] is True
        assert Path(dir_path).exists()
        assert Path(dir_path).is_dir()

    @pytest.mark.asyncio
    async def test_create_directory_idempotent(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test that creating existing directory succeeds (idempotent)."""
        # Create directory first
        existing_dir = doc_root / "existing"
        existing_dir.mkdir()
        
        result = await create_directory(str(existing_dir))
        
        assert result["success"] is True
        assert result["created"] is False
        assert "already exists" in result["message"]


class TestRenameDirectory:
    """Tests for rename_directory tool."""

    @pytest.mark.asyncio
    async def test_rename_directory_updates_docs(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        doc_root: Path,
    ):
        """Test that renaming directory updates document paths."""
        # Setup directory with documents
        subdir = doc_root / "subdir"
        subdir.mkdir()
        
        # Register documents in subdirectory
        file1 = subdir / "file1.txt"
        file1.write_text("content 1")
        doc1 = mock_store.register(file1)
        
        file2 = subdir / "file2.txt"
        file2.write_text("content 2")
        doc2 = mock_store.register(file2)
        
        # Execute rename
        result = await rename_directory(str(subdir), "renamed_subdir")
        
        # Verify result
        assert result["success"] is True
        assert result["old_path"] == str(subdir)
        assert result["new_path"] == str(doc_root / "renamed_subdir")
        assert result["documents_updated"] == 2
        
        # Check directory actually renamed
        assert not subdir.exists()
        assert (doc_root / "renamed_subdir").exists()
        
        # Check processor and indexer called
        mock_processor.wait_for_documents.assert_called_once_with(
            [doc1.id, doc2.id], timeout=120.0, require_completed=False
        )
        mock_indexer.update_paths_batch_in_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_rename_directory_invalid_name(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when new name contains path separators."""
        subdir = doc_root / "subdir"
        subdir.mkdir()
        
        result = await rename_directory(str(subdir), "invalid/name")
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value
        assert "path separators" in result["message"]


class TestMoveDirectory:
    """Tests for move_directory tool."""

    @pytest.mark.asyncio
    async def test_move_directory_updates_docs(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        temp_dir: Path,
    ):
        """Test that moving directory updates document paths."""
        # Setup source and destination roots
        source_root = temp_dir / "source_root"
        source_root.mkdir()
        mock_store.add_root(str(source_root), name="Source Root")
        
        dest_root = temp_dir / "dest_root"
        dest_root.mkdir()
        mock_store.add_root(str(dest_root), name="Dest Root")
        
        # Create directory with documents in source
        source_dir = source_root / "moveme"
        source_dir.mkdir()
        
        file1 = source_dir / "file1.txt"
        file1.write_text("content 1")
        doc1 = mock_store.register(file1)
        
        # Execute move
        dest_dir = dest_root / "moved_dir"
        result = await move_directory(str(source_dir), str(dest_dir))
        
        # Verify
        assert result["success"] is True
        assert result["old_path"] == str(source_dir)
        assert result["new_path"] == str(dest_dir)
        assert result["documents_updated"] == 1
        
        # Check directory moved
        assert not source_dir.exists()
        assert dest_dir.exists()
        
        # Check indexer called
        mock_indexer.update_paths_batch_in_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_directory_rejects_source_outside_root(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        temp_dir: Path,
    ):
        """A source directory outside every document root is rejected, unmoved."""
        dest_root = temp_dir / "dest_root"
        dest_root.mkdir()
        mock_store.add_root(str(dest_root), name="Dest Root")

        # Source lives outside any registered root.
        outside = temp_dir / "outside"
        outside.mkdir()
        (outside / "file.txt").write_text("content")
        dest_dir = dest_root / "moved"

        result = await move_directory(str(outside), str(dest_dir))

        assert "error_code" in result
        assert result["error_code"] == ErrorCode.PERMISSION_DENIED.value
        # Nothing was moved and the index was not touched.
        assert outside.exists()
        assert not dest_dir.exists()
        mock_indexer.update_paths_batch_in_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_move_directory_rejects_dest_outside_root(
        self,
        mock_store: DocumentStore,
        mock_processor: AsyncMock,
        mock_indexer: AsyncMock,
        temp_dir: Path,
    ):
        """A destination outside every document root is rejected, unmoved."""
        source_root = temp_dir / "source_root"
        source_root.mkdir()
        mock_store.add_root(str(source_root), name="Source Root")
        source_dir = source_root / "moveme"
        source_dir.mkdir()
        file1 = source_dir / "file1.txt"
        file1.write_text("content 1")
        mock_store.register(file1)

        # Destination is outside any registered root.
        outside_parent = temp_dir / "outside"
        outside_parent.mkdir()
        dest_dir = outside_parent / "moved"

        result = await move_directory(str(source_dir), str(dest_dir))

        assert "error_code" in result
        assert result["error_code"] == ErrorCode.PERMISSION_DENIED.value
        # Nothing was moved and the index was not touched.
        assert source_dir.exists()
        assert not dest_dir.exists()
        mock_indexer.update_paths_batch_in_index.assert_not_called()


class TestDeleteDirectory:
    """Tests for delete_directory tool."""

    @pytest.mark.asyncio
    async def test_delete_directory_empty(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test deletion of empty directory."""
        empty_dir = doc_root / "empty"
        empty_dir.mkdir()
        
        result = await delete_directory(str(empty_dir))
        
        assert result["success"] is True
        assert result["path"] == str(empty_dir)
        assert result["recursive"] is False
        assert not empty_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_directory_has_files(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when directory contains files."""
        dir_with_file = doc_root / "has_file"
        dir_with_file.mkdir()
        (dir_with_file / "file.txt").write_text("content")
        
        result = await delete_directory(str(dir_with_file))
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.CONFLICT.value
        assert "contains files" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_directory_is_root(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when trying to delete a document root."""
        result = await delete_directory(str(doc_root))
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.CONFLICT.value
        assert "document root" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_directory_has_registered_docs(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when directory contains registered documents."""
        subdir = doc_root / "subdir"
        subdir.mkdir()
        
        # Register a document in subdirectory
        file_in_subdir = subdir / "registered.txt"
        file_in_subdir.write_text("registered content")
        mock_store.register(file_in_subdir)
        
        result = await delete_directory(str(subdir))
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.CONFLICT.value
        assert "registered documents" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_directory_recursive_empty_tree(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test recursive deletion of empty directory tree."""
        # Create nested empty directories
        parent = doc_root / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        grandchild = child / "grandchild"
        grandchild.mkdir()
        
        result = await delete_directory(str(parent), recursive=True)
        
        assert result["success"] is True
        assert result["recursive"] is True
        assert not parent.exists()

    @pytest.mark.asyncio
    async def test_delete_directory_recursive_has_files(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when recursive tree contains files."""
        parent = doc_root / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        
        # Add a file deep in the tree
        (child / "file.txt").write_text("content")
        
        result = await delete_directory(str(parent), recursive=True)
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.CONFLICT.value
        assert "contains files" in result["message"]


class TestValidationHelpers:
    """Tests for validation helper functions."""

    @pytest.mark.asyncio
    async def test_path_outside_document_root(
        self,
        mock_store: DocumentStore,
        temp_dir: Path,
    ):
        """Test error when path is outside any document root."""
        # Create file outside document root
        outside_file = temp_dir / "outside.txt"
        outside_file.write_text("outside content")
        
        result = await move_file(str(outside_file), str(outside_file.parent / "moved_outside.txt"))

        assert "error_code" in result
        # File outside doc root is also unregistered, so NOT_FOUND fires first
        assert result["error_code"] == ErrorCode.NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_relative_path_error(
        self,
        mock_store: DocumentStore,
    ):
        """Test error for relative paths (resolved to absolute, then not found)."""
        result = await move_file("relative/path.txt", "other/relative/path.txt")

        assert "error_code" in result
        # Path.resolve() makes relative paths absolute, so FILE_NOT_FOUND fires
        assert result["error_code"] == ErrorCode.FILE_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_source_file_not_found(
        self,
        mock_store: DocumentStore,
        doc_root: Path,
    ):
        """Test error when source file doesn't exist."""
        nonexistent = str(doc_root / "nonexistent.txt")
        dest = str(doc_root / "dest.txt")
        
        result = await move_file(nonexistent, dest)
        
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.FILE_NOT_FOUND.value
        assert "does not exist" in result["message"]