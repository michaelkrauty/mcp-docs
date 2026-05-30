"""Tests for document scanning."""

import tempfile
from pathlib import Path

import pytest

from mcp_docs.scanning import DocumentScanner, ScanResult
from mcp_docs.storage.database import DocumentStore


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
def document_root(temp_dir: Path) -> Path:
    """Create a document root with sample files."""
    root = temp_dir / "documents"
    root.mkdir()

    # Create sample files
    (root / "file1.txt").write_text("This is file 1")
    (root / "file2.md").write_text("# File 2\n\nMarkdown content")
    (root / "subdir").mkdir()
    (root / "subdir" / "file3.txt").write_text("Nested file")

    # Create unsupported file
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    return root


class TestDocumentScanner:
    """Tests for DocumentScanner."""

    def test_is_supported(self, store: DocumentStore) -> None:
        """is_supported correctly identifies supported extensions."""
        scanner = DocumentScanner(store)

        assert scanner.is_supported(Path("test.txt"))
        assert scanner.is_supported(Path("test.md"))
        assert scanner.is_supported(Path("test.pdf"))
        assert scanner.is_supported(Path("test.docx"))
        assert scanner.is_supported(Path("notebook.ipynb"))
        assert scanner.is_supported(Path("test.TXT"))  # Case insensitive

        assert not scanner.is_supported(Path("test.png"))
        assert not scanner.is_supported(Path("test.jpg"))
        assert not scanner.is_supported(Path("test.exe"))

    @pytest.mark.asyncio
    async def test_scan_root_finds_files(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """scan_root finds all supported files."""
        scanner = DocumentScanner(store, recursive=True)

        # Add root to store
        root = store.add_root(str(document_root))

        result = await scanner.scan_root(root)

        assert result.files_found == 3  # file1.txt, file2.md, subdir/file3.txt
        assert result.files_new == 3
        assert result.files_skipped >= 1  # image.png
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_scan_root_non_recursive(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """Non-recursive scan only finds top-level files."""
        scanner = DocumentScanner(store, recursive=False)

        # Add root to store
        root = store.add_root(str(document_root), recursive=False)

        result = await scanner.scan_root(root)

        assert result.files_found == 2  # file1.txt, file2.md only
        assert result.files_new == 2

    @pytest.mark.asyncio
    async def test_scan_root_detects_modifications(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """scan_root detects modified files."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # First scan
        await scanner.scan_root(root)

        # Modify a file
        (document_root / "file1.txt").write_text("Modified content")

        # Second scan
        result = await scanner.scan_root(root)

        assert result.files_modified == 1
        assert result.files_new == 0

    @pytest.mark.asyncio
    async def test_scan_root_detects_deletions(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """scan_root marks deleted files."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # First scan
        await scanner.scan_root(root)

        # Delete a file
        (document_root / "file1.txt").unlink()

        # Second scan
        result = await scanner.scan_root(root)

        assert result.files_deleted == 1

    @pytest.mark.asyncio
    async def test_scan_root_skips_hidden_files(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """scan_root skips hidden files."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # Create hidden file
        (document_root / ".hidden.txt").write_text("Hidden")

        result = await scanner.scan_root(root)

        # Should skip the hidden file
        assert result.files_found == 3  # Only the original 3

    @pytest.mark.asyncio
    async def test_scan_root_with_callback(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """scan_root calls enqueue callback for new files."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        enqueued = []

        async def callback(doc_id, path):
            enqueued.append((doc_id, path))

        await scanner.scan_root(root, enqueue_callback=callback)

        assert len(enqueued) == 3

    @pytest.mark.asyncio
    async def test_scan_root_delete_callback_on_deletion(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """Regression test: delete_callback is invoked when files are deleted.

        This tests the fix for the bug where deleted files remained in the
        search index because the scanner wasn't cleaning up the vector index.
        """
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # Track documents registered during first scan
        registered_docs = {}

        async def enqueue_callback(doc_id, path):
            registered_docs[str(path)] = doc_id  # Convert PosixPath to string

        # First scan to register files
        await scanner.scan_root(root, enqueue_callback=enqueue_callback)

        # Find the doc_id for file1.txt
        file1_path = str(document_root / "file1.txt")
        assert file1_path in registered_docs
        deleted_doc_id = registered_docs[file1_path]

        # Delete a file
        (document_root / "file1.txt").unlink()

        # Track delete callbacks
        deleted_ids = []

        async def delete_callback(doc_id):
            deleted_ids.append(doc_id)

        # Second scan with delete_callback
        result = await scanner.scan_root(root, delete_callback=delete_callback)

        # Verify delete_callback was invoked with correct document ID
        assert result.files_deleted == 1
        assert len(deleted_ids) == 1
        assert deleted_ids[0] == deleted_doc_id

    @pytest.mark.asyncio
    async def test_scan_root_delete_callback_multiple_deletions(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """Regression test: delete_callback is invoked for each deleted file."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # First scan to register files
        await scanner.scan_root(root)

        # Delete multiple files
        (document_root / "file1.txt").unlink()
        (document_root / "file2.md").unlink()

        deleted_ids = []

        async def delete_callback(doc_id):
            deleted_ids.append(doc_id)

        # Second scan with delete_callback
        result = await scanner.scan_root(root, delete_callback=delete_callback)

        # Verify delete_callback was invoked for both deleted files
        assert result.files_deleted == 2
        assert len(deleted_ids) == 2

    @pytest.mark.asyncio
    async def test_scan_nonexistent_root(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """scan_root handles nonexistent directory."""
        scanner = DocumentScanner(store)
        root = store.add_root(str(temp_dir / "nonexistent"))

        result = await scanner.scan_root(root)

        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0]


class TestScanResult:
    """Tests for ScanResult."""

    def test_to_dict(self) -> None:
        """ScanResult.to_dict works correctly."""
        from datetime import UTC, datetime

        result = ScanResult(
            root_path="/test/path",
            scanned_at=datetime.now(UTC),
            files_found=10,
            files_new=5,
            files_modified=2,
            files_deleted=1,
            files_skipped=3,
        )

        d = result.to_dict()

        assert d["root_path"] == "/test/path"
        assert d["files_found"] == 10
        assert d["files_new"] == 5
        assert d["files_modified"] == 2
        assert d["files_deleted"] == 1
        assert d["files_skipped"] == 3
