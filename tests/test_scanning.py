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
