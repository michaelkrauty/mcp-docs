"""Tests for document scanning."""

import os
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from mcp_docs.models import DocumentStatus
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
    async def test_scan_root_honors_root_recursive_over_instance_flag(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """A recursive=False root is scanned non-recursively even when the
        scanner instance defaults to recursive=True.

        The production scanner is a process-wide singleton built with the
        default recursive=True, so the per-root flag must win at scan time.
        """
        # Mirror the production singleton: default recursive=True instance.
        scanner = DocumentScanner(store)
        assert scanner.recursive is True

        # Root explicitly opts out of recursion.
        root = store.add_root(str(document_root), recursive=False)

        result = await scanner.scan_root(root)

        # Only top-level supported files; subdir/file3.txt must be excluded.
        assert result.files_found == 2  # file1.txt, file2.md only
        assert result.files_new == 2

    @pytest.mark.asyncio
    async def test_scan_root_instance_recursive_false_overrides_recursive_root(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """A scanner built with recursive=False stays non-recursive even for a
        root whose recursive flag is the default True.

        Either non-recursive setting prevents descent, so the instance-level
        override is preserved for direct callers of DocumentScanner.
        """
        scanner = DocumentScanner(store, recursive=False)
        root = store.add_root(str(document_root), recursive=True)

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
    async def test_scan_root_does_not_loop_on_hash_collision(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """A file edited to byte-match an already-registered file must not be
        re-flagged as modified and re-enqueued on every scan. Its new hash
        already belongs to the other document, so the content_hash UNIQUE
        constraint drops the hash update; without a guard the scanner would
        re-detect it as modified forever and re-extract/re-embed it."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))
        await scanner.scan_root(root)

        file1 = document_root / "file1.txt"
        file1_doc = next(
            d
            for d in store.list_summaries(document_root=str(document_root))
            if d.path == str(file1)
        )

        # Edit file1 to be byte-identical to file2 (hash collision).
        file1.write_text((document_root / "file2.md").read_text())

        enqueued: list[UUID] = []

        async def enqueue_cb(doc_id: UUID, path: Path) -> None:
            enqueued.append(doc_id)

        # Repeated scans must not perpetually re-flag/re-enqueue the duplicate.
        await scanner.scan_root(root, enqueue_callback=enqueue_cb)
        result = await scanner.scan_root(root, enqueue_callback=enqueue_cb)

        assert result.files_modified == 0
        assert file1_doc.id not in enqueued

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


class TestIncompleteScanDeletionSafety:
    """A scan that does not finish must never reconcile deletions.

    Regression tests for the data-loss bug where a walk truncated by the
    per-root file limit, or aborted by an error mid-traversal, left
    ``seen_paths`` incomplete and the deletion pass then marked every
    unvisited (but still on-disk) document as DELETED and purged its vectors.
    """

    @pytest.mark.asyncio
    async def test_complete_flag_true_on_normal_scan(
        self, store: DocumentStore, document_root: Path
    ) -> None:
        """A full walk reports complete=True and still reconciles deletions."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        result = await scanner.scan_root(root)
        assert result.complete is True

        # A genuinely removed file is still marked deleted on a complete scan.
        (document_root / "file1.txt").unlink()
        result2 = await scanner.scan_root(root)
        assert result2.complete is True
        assert result2.files_deleted == 1

    @pytest.mark.asyncio
    async def test_file_limit_truncation_skips_deletion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: DocumentStore,
        document_root: Path,
    ) -> None:
        """Hitting MAX_FILES_PER_ROOT must not delete the unvisited documents."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        # Full scan registers all three supported files.
        registered: dict[str, object] = {}

        async def enqueue(doc_id, path):
            registered[str(path)] = doc_id

        await scanner.scan_root(root, enqueue_callback=enqueue)
        assert len(registered) == 3

        # Cap the walk at a single file: the other two are never visited.
        monkeypatch.setattr("mcp_docs.scanning.scanner.MAX_FILES_PER_ROOT", 1)

        deleted_ids: list[object] = []

        async def delete_callback(doc_id):
            deleted_ids.append(doc_id)

        result = await scanner.scan_root(root, delete_callback=delete_callback)

        # The unvisited documents must NOT be deleted or purged from the index.
        assert result.files_deleted == 0
        assert deleted_ids == []
        # No registered document was marked deleted.
        for doc_id in registered.values():
            assert store.read(doc_id).status != DocumentStatus.DELETED
        # ...and the scan must report itself incomplete.
        assert result.complete is False
        assert any("incomplete" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_walk_error_skips_deletion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: DocumentStore,
        document_root: Path,
    ) -> None:
        """An error during directory traversal must not delete any document."""
        scanner = DocumentScanner(store, recursive=True)
        root = store.add_root(str(document_root))

        registered: dict[str, object] = {}

        async def enqueue(doc_id, path):
            registered[str(path)] = doc_id

        await scanner.scan_root(root, enqueue_callback=enqueue)
        assert len(registered) == 3

        def failing_walk(*args, **kwargs):
            raise OSError("simulated traversal failure")

        monkeypatch.setattr("os.walk", failing_walk)

        deleted_ids: list[object] = []

        async def delete_callback(doc_id):
            deleted_ids.append(doc_id)

        result = await scanner.scan_root(root, delete_callback=delete_callback)

        # No document may be deleted or purged when traversal fails.
        assert result.files_deleted == 0
        assert deleted_ids == []
        for doc_id in registered.values():
            assert store.read(doc_id).status != DocumentStatus.DELETED
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_missing_root_reports_incomplete(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """A root that does not exist reports complete=False (no walk ran)."""
        scanner = DocumentScanner(store)
        root = store.add_root(str(temp_dir / "gone"))

        result = await scanner.scan_root(root)

        assert result.complete is False
        assert result.files_deleted == 0

    @pytest.mark.asyncio
    async def test_non_directory_root_reports_incomplete(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """A root path that is a file (not a directory) reports complete=False."""
        file_root = temp_dir / "not_a_dir.txt"
        file_root.write_text("regular file")
        scanner = DocumentScanner(store)
        root = store.add_root(str(file_root))

        result = await scanner.scan_root(root)

        assert result.complete is False
        assert result.files_deleted == 0

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="requires non-root POSIX (chmod-based permission denial)",
    )
    @pytest.mark.asyncio
    async def test_unreadable_subdir_skips_deletion(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """A subdir that becomes unreadable must not delete documents inside it.

        ``os.walk`` (like pathlib) silently omits the children of a directory
        it cannot list rather than raising, so the scan must detect the
        ``onerror`` callback and refuse to reconcile deletions.
        """
        root = temp_dir / "docs"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (root / "top.txt").write_text("top level")
        (sub / "child.txt").write_text("child document")

        scanner = DocumentScanner(store, recursive=True)
        docroot = store.add_root(str(root))

        ids_by_name: dict[str, object] = {}

        async def enqueue(doc_id, path):
            ids_by_name[Path(path).name] = doc_id

        result1 = await scanner.scan_root(docroot, enqueue_callback=enqueue)
        assert result1.complete is True
        assert set(ids_by_name) == {"top.txt", "child.txt"}
        child_id = ids_by_name["child.txt"]

        # Make the subdirectory unreadable: its child can no longer be listed.
        sub.chmod(0)
        try:
            deleted_ids: list[object] = []

            async def delete_callback(doc_id):
                deleted_ids.append(doc_id)

            result2 = await scanner.scan_root(docroot, delete_callback=delete_callback)
        finally:
            sub.chmod(0o700)

        # The child under the unreadable subdir must survive untouched.
        assert result2.complete is False
        assert result2.files_deleted == 0
        assert deleted_ids == []
        assert store.read(child_id).status != DocumentStatus.DELETED

    @pytest.mark.asyncio
    async def test_scan_all_roots_failure_reports_incomplete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: DocumentStore,
        document_root: Path,
    ) -> None:
        """If scan_root raises, scan_all_roots reports that root incomplete."""
        scanner = DocumentScanner(store, recursive=True)
        store.add_root(str(document_root))

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated scan failure")

        monkeypatch.setattr(scanner, "scan_root", boom)

        results = await scanner.scan_all_roots()

        assert len(results) == 1
        assert results[0].complete is False


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

    def test_to_dict_includes_complete(self) -> None:
        """to_dict surfaces the completeness flag (defaults True)."""
        from datetime import UTC, datetime

        result = ScanResult(root_path="/p", scanned_at=datetime.now(UTC))
        assert result.to_dict()["complete"] is True

        result.complete = False
        assert result.to_dict()["complete"] is False
