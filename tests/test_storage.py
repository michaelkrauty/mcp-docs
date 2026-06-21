"""Tests for document storage."""

import sqlite3
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from mcp_docs.models import Document, DocumentStatus, DocumentType, ExtractionStatus
from mcp_docs.storage.database import DocumentStore, compute_file_hash


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
def sample_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World! This is a test document.")
    return file_path


@pytest.fixture
def sample_pdf(temp_dir: Path) -> Path:
    """Create a sample PDF path (not a real PDF)."""
    file_path = temp_dir / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content for testing")
    return file_path


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_consistent_hash(self, sample_file: Path) -> None:
        """Same content produces same hash."""
        hash1 = compute_file_hash(sample_file)
        hash2 = compute_file_hash(sample_file)
        assert hash1 == hash2

    def test_different_content_different_hash(self, temp_dir: Path) -> None:
        """Different content produces different hash."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        assert hash1 != hash2

    def test_hash_is_sha256(self, sample_file: Path) -> None:
        """Hash is a valid SHA-256 hex string."""
        file_hash = compute_file_hash(sample_file)
        assert len(file_hash) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in file_hash)


class TestDocumentStoreBasic:
    """Basic DocumentStore tests."""

    def test_register_document(self, store: DocumentStore, sample_file: Path) -> None:
        """Can register a new document."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        assert doc.id is not None
        assert isinstance(doc.id, UUID)
        assert doc.content_hash == content_hash
        assert doc.path == str(sample_file)
        assert doc.filename == "sample.txt"
        assert doc.doc_type == DocumentType.TXT
        assert doc.status == DocumentStatus.ACTIVE
        assert doc.extraction_status == ExtractionStatus.QUEUED

    def test_register_with_tags(self, store: DocumentStore, sample_file: Path) -> None:
        """Can register a document with tags."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash, tags=["test", "sample"])

        assert set(doc.tags) == {"test", "sample"}

    def test_register_drops_blank_tags(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """A whitespace-only/empty tag is dropped at registration, matching
        update_tags (which guards `if normalized:`); otherwise register would
        store an empty-string tag that update_tags filters out."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash, tags=["Finance", "  ", ""])

        assert set(doc.tags) == {"finance"}

    def test_read_document(self, store: DocumentStore, sample_file: Path) -> None:
        """Can read a registered document."""
        content_hash = compute_file_hash(sample_file)
        created = store.register(sample_file, content_hash)

        doc = store.read(created.id)
        assert doc is not None
        assert doc.id == created.id
        assert doc.content_hash == content_hash

    def test_read_nonexistent(self, store: DocumentStore) -> None:
        """Reading nonexistent document returns None."""
        fake_id = UUID("00000000-0000-0000-0000-000000000000")
        doc = store.read(fake_id)
        assert doc is None

    def test_get_by_hash(self, store: DocumentStore, sample_file: Path) -> None:
        """Can retrieve document by content hash."""
        content_hash = compute_file_hash(sample_file)
        created = store.register(sample_file, content_hash)

        doc = store.get_by_hash(content_hash)
        assert doc is not None
        assert doc.id == created.id

    def test_get_by_hash_not_found(self, store: DocumentStore) -> None:
        """get_by_hash returns None for unknown hash."""
        doc = store.get_by_hash("0" * 64)
        assert doc is None


class TestDocumentStoreUpdate:
    """Tests for document updates."""

    def test_update_status(self, store: DocumentStore, sample_file: Path) -> None:
        """Can update document status."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        updated = store.update(doc.id, status=DocumentStatus.MODIFIED)
        assert updated.status == DocumentStatus.MODIFIED

    def test_update_extraction_status(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Can update extraction status."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        updated = store.update(doc.id, extraction_status=ExtractionStatus.EXTRACTED)
        assert updated.extraction_status == ExtractionStatus.EXTRACTED

    def test_update_tags(self, store: DocumentStore, sample_file: Path) -> None:
        """Can update document tags."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash, tags=["old"])

        store.update_tags(doc.id, ["new1", "new2"])
        updated = store.read(doc.id)
        assert set(updated.tags) == {"new1", "new2"}

    def test_update_multiple_fields(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Can update multiple fields at once."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        updated = store.update(
            doc.id,
            title="New Title",
            page_count=5,
            word_count=100,
        )
        assert updated.title == "New Title"
        assert updated.page_count == 5
        assert updated.word_count == 100

    def test_update_filename(self, store: DocumentStore, sample_file: Path) -> None:
        """Can update the stored basename (for rename-moves)."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)
        assert doc.filename == "sample.txt"

        updated = store.update(doc.id, filename="renamed.txt")
        assert updated.filename == "renamed.txt"
        assert store.read(doc.id).filename == "renamed.txt"


class TestDocumentStoreDelete:
    """Tests for document deletion."""

    def test_delete_document(self, store: DocumentStore, sample_file: Path) -> None:
        """Can delete a document."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        store.delete(doc.id)
        assert store.read(doc.id) is None

    def test_delete_removes_tags(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Deleting document removes its tags."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash, tags=["test"])

        store.delete(doc.id)
        # Tags should be cleaned up (cascade delete)
        assert store.read(doc.id) is None


class TestDocumentStoreQuery:
    """Tests for document queries."""

    def test_list_summaries(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can list document summaries."""
        # Create multiple files
        for i in range(3):
            path = temp_dir / f"file{i}.txt"
            path.write_text(f"Content {i}")
            store.register(path, compute_file_hash(path))

        summaries = store.list_summaries()
        assert len(summaries) == 3

    def test_list_summaries_with_tag_filter(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Can filter summaries by tag."""
        path1 = temp_dir / "file1.txt"
        path2 = temp_dir / "file2.txt"
        path1.write_text("Content 1")
        path2.write_text("Content 2")

        store.register(path1, compute_file_hash(path1), tags=["important"])
        store.register(path2, compute_file_hash(path2), tags=["other"])

        summaries = store.list_summaries(tags=["important"])
        assert len(summaries) == 1
        assert "important" in summaries[0].tags

    def test_list_summaries_limit(self, store: DocumentStore, temp_dir: Path) -> None:
        """List respects limit parameter."""
        for i in range(10):
            path = temp_dir / f"file{i}.txt"
            path.write_text(f"Content {i}")
            store.register(path, compute_file_hash(path))

        summaries = store.list_summaries(limit=5)
        assert len(summaries) == 5


class TestDocumentRoots:
    """Tests for document root management."""

    def test_add_root(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can add a document root."""
        root = store.add_root(str(temp_dir))

        assert root.path == str(temp_dir)
        assert root.file_count == 0
        assert root.last_scanned is None

    def test_list_roots(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can list document roots."""
        store.add_root(str(temp_dir))

        roots = store.list_roots()
        assert len(roots) == 1
        assert roots[0].path == str(temp_dir)

    def test_get_root(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can get a specific root."""
        store.add_root(str(temp_dir))

        root = store.get_root(str(temp_dir))
        assert root is not None
        assert root.path == str(temp_dir)

    def test_update_root_scan(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can update root scan info."""
        store.add_root(str(temp_dir))

        store.update_root_scan(str(temp_dir), file_count=10)
        root = store.get_root(str(temp_dir))

        assert root.file_count == 10
        assert root.last_scanned is not None

    def test_remove_root(self, store: DocumentStore, temp_dir: Path) -> None:
        """Can remove a document root."""
        store.add_root(str(temp_dir))
        store.remove_root(str(temp_dir))

        assert store.get_root(str(temp_dir)) is None


class TestConcurrentRegistration:
    """Regression tests for concurrent document registration.

    HIGH-3: The register() method had a TOCTOU race condition where concurrent
    calls could cause duplicate entries or constraint violations. Fixed by
    using INSERT OR IGNORE with SQLite's UNIQUE constraint on content_hash.
    """

    def test_register_returns_same_document_for_duplicate_hash(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Registering same file twice returns same document (no duplicates)."""
        content_hash = compute_file_hash(sample_file)

        doc1 = store.register(sample_file, content_hash)
        doc2 = store.register(sample_file, content_hash)

        # Should return same document, not create duplicate
        assert doc1.id == doc2.id
        assert doc1.content_hash == doc2.content_hash

    def test_concurrent_registration_no_duplicates(
        self, temp_dir: Path
    ) -> None:
        """Concurrent registration of same file doesn't create duplicates.

        Regression test for HIGH-3 TOCTOU race condition.
        """
        import concurrent.futures

        # Create a dedicated store for this test
        db_path = temp_dir / "concurrent_test.db"
        store = DocumentStore(db_path=db_path)

        # Create test file
        test_file = temp_dir / "concurrent_test.txt"
        test_file.write_text("Content for concurrent test")
        content_hash = compute_file_hash(test_file)

        # Define registration function
        def register_doc():
            return store.register(test_file, content_hash)

        # Run multiple concurrent registrations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(register_doc) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should return the same document ID (no duplicates created)
        doc_ids = {doc.id for doc in results}
        assert len(doc_ids) == 1, f"Expected 1 unique document, got {len(doc_ids)}"

        # Verify only one document exists in DB
        all_docs = store.query()
        matching = [d for d in all_docs if d.content_hash == content_hash]
        assert len(matching) == 1, f"Expected 1 document in DB, got {len(matching)}"

        store.close()

    def test_register_updates_path_on_file_move(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Registering moved file updates path, not creates duplicate."""
        # Create file at location A
        file_a = temp_dir / "subdir_a" / "file.txt"
        file_a.parent.mkdir()
        file_a.write_text("Same content")
        content_hash = compute_file_hash(file_a)

        # Register at location A
        doc1 = store.register(file_a, content_hash)
        original_id = doc1.id

        # "Move" file to location B (same content, different path)
        file_b = temp_dir / "subdir_b" / "file.txt"
        file_b.parent.mkdir()
        file_b.write_text("Same content")
        # Content hash is same

        # Register at location B
        doc2 = store.register(file_b, content_hash)

        # Should be same document with updated path
        assert doc2.id == original_id
        assert doc2.path == str(file_b)


class TestBatchPathUpdates:
    """Prefix updates must stop at the directory boundary and treat LIKE
    wildcards in directory names literally."""

    def _register_at(self, store: DocumentStore, temp_dir: Path, rel: str) -> Document:
        file_path = temp_dir / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"unique content for {rel}")
        return store.register(file_path)

    def test_rename_does_not_touch_sibling_with_shared_prefix(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        doc_a = self._register_at(store, temp_dir, "docs/a.txt")
        doc_b = self._register_at(store, temp_dir, "docs2/b.txt")

        count = store.update_paths_batch(
            str(temp_dir / "docs"), str(temp_dir / "archive")
        )

        assert count == 1
        assert store.read(doc_a.id).path == str(temp_dir / "archive" / "a.txt")
        assert store.read(doc_b.id).path == str(temp_dir / "docs2" / "b.txt")

    def test_underscore_in_directory_name_is_literal(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        doc_a = self._register_at(store, temp_dir, "my_dir/a.txt")
        doc_b = self._register_at(store, temp_dir, "myxdir/b.txt")

        count = store.update_paths_batch(
            str(temp_dir / "my_dir"), str(temp_dir / "renamed")
        )

        assert count == 1
        assert store.read(doc_a.id).path == str(temp_dir / "renamed" / "a.txt")
        assert store.read(doc_b.id).path == str(temp_dir / "myxdir" / "b.txt")

    def test_trailing_slash_on_prefix_is_accepted(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        doc_a = self._register_at(store, temp_dir, "docs/a.txt")

        count = store.update_paths_batch(
            str(temp_dir / "docs") + "/", str(temp_dir / "archive") + "/"
        )

        assert count == 1
        assert store.read(doc_a.id).path == str(temp_dir / "archive" / "a.txt")

    def test_document_roots_batch_respects_boundary(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        doc_a = self._register_at(store, temp_dir, "docs/a.txt")
        doc_b = self._register_at(store, temp_dir, "docs2/b.txt")

        count = store.update_document_roots_batch(
            str(temp_dir / "docs"), "/new/root"
        )

        assert count == 1
        assert store.read(doc_a.id).document_root == "/new/root"
        assert store.read(doc_b.id).document_root != "/new/root"

    def test_case_only_sibling_directory_untouched(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """SQLite LIKE is case-insensitive; the prefix match must not be."""
        doc_a = self._register_at(store, temp_dir, "docs/a.txt")
        doc_b = self._register_at(store, temp_dir, "Docs/b.txt")

        count = store.update_paths_batch(
            str(temp_dir / "docs"), str(temp_dir / "archive")
        )

        assert count == 1
        assert store.read(doc_a.id).path == str(temp_dir / "archive" / "a.txt")
        assert store.read(doc_b.id).path == str(temp_dir / "Docs" / "b.txt")


class TestQueryByPathPrefix:
    """Directory queries must stop at the path boundary and treat LIKE
    wildcards / case literally, like the batch update functions."""

    def _register_at(self, store: DocumentStore, temp_dir: Path, rel: str) -> Document:
        file_path = temp_dir / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"unique content for {rel}")
        return store.register(file_path)

    def test_returns_only_children_under_dir(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        self._register_at(store, temp_dir, "docs/a.txt")
        self._register_at(store, temp_dir, "docs/sub/b.txt")
        self._register_at(store, temp_dir, "docs2/c.txt")

        results = store.query_by_path_prefix(str(temp_dir / "docs") + "/")

        paths = {r.path for r in results}
        assert paths == {
            str(temp_dir / "docs" / "a.txt"),
            str(temp_dir / "docs" / "sub" / "b.txt"),
        }

    def test_excludes_underscore_sibling(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """SQLite LIKE treats "_" as a single-char wildcard; the match must not."""
        doc_a = self._register_at(store, temp_dir, "my_docs/a.txt")
        self._register_at(store, temp_dir, "myXdocs/b.txt")

        results = store.query_by_path_prefix(str(temp_dir / "my_docs") + "/")

        assert [r.id for r in results] == [doc_a.id]

    def test_excludes_case_only_sibling(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """SQLite LIKE is ASCII case-insensitive; the prefix match must not be."""
        doc_a = self._register_at(store, temp_dir, "docs/a.txt")
        self._register_at(store, temp_dir, "Docs/b.txt")

        results = store.query_by_path_prefix(str(temp_dir / "docs") + "/")

        assert [r.id for r in results] == [doc_a.id]

    def test_prefix_without_trailing_slash_is_anchored(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        doc_a = self._register_at(store, temp_dir, "my_docs/a.txt")
        self._register_at(store, temp_dir, "myXdocs/b.txt")

        results = store.query_by_path_prefix(str(temp_dir / "my_docs"))

        assert [r.id for r in results] == [doc_a.id]


class TestExtractionErrorSentinel:
    """extraction_error=None clears the stored error; UNSET leaves it."""

    def test_none_clears_stale_error(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file)
        store.update(
            doc.id,
            extraction_status=ExtractionStatus.FAILED,
            extraction_error="boom",
        )

        store.update(doc.id, extraction_error=None)

        assert store.read(doc.id).extraction_error is None

    def test_unset_default_preserves_error(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file)
        store.update(
            doc.id,
            extraction_status=ExtractionStatus.FAILED,
            extraction_error="boom",
        )

        store.update(doc.id, title="new title")

        assert store.read(doc.id).extraction_error == "boom"


class TestIterAll:
    """iter_all() must yield the entire matching corpus (no 50-row cap), filter
    by extraction status, and survive individual unreadable rows while staying
    loud on systemic database errors.

    This underpins the index_all fix: index_all only indexed the first 50
    extracted documents because it enumerated via the 50-capped query() instead
    of iter_all().
    """

    def _make_extracted(
        self, store: DocumentStore, temp_dir: Path, n: int
    ) -> list[UUID]:
        ids: list[UUID] = []
        for i in range(n):
            f = temp_dir / f"iter_doc_{i}.txt"
            f.write_text(f"content {i}")
            doc = store.register(f)
            store.update(doc.id, extraction_status=ExtractionStatus.EXTRACTED)
            ids.append(doc.id)
        return ids

    def test_yields_more_than_default_query_limit(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """iter_all returns the whole corpus, well past query()'s default 50."""
        self._make_extracted(store, temp_dir, 60)

        assert len(list(store.iter_all())) == 60

    def test_filters_by_extraction_status(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Only documents in the requested extraction state are yielded."""
        extracted = set(self._make_extracted(store, temp_dir, 5))
        # Three more left in the default QUEUED state.
        for i in range(3):
            f = temp_dir / f"queued_{i}.txt"
            f.write_text(f"queued {i}")
            store.register(f)

        seen = {
            d.id
            for d in store.iter_all(extraction_status=ExtractionStatus.EXTRACTED)
        }

        assert seen == extracted

    def test_skips_vanished_document(
        self,
        store: DocumentStore,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A row whose read() returns None (deleted between the id snapshot and
        its read) is skipped, not yielded as None."""
        gone, kept = self._make_extracted(store, temp_dir, 2)
        real_read = store.read

        def flaky_read(document_id: UUID):
            return None if document_id == gone else real_read(document_id)

        monkeypatch.setattr(store, "read", flaky_read)

        seen = [d.id for d in store.iter_all()]

        assert gone not in seen
        assert seen == [kept]

    def test_skips_unreadable_document(
        self,
        store: DocumentStore,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A malformed row (read raises ValueError) is skipped rather than
        aborting iteration over the rest of the corpus."""
        bad, good = self._make_extracted(store, temp_dir, 2)
        real_read = store.read

        def flaky_read(document_id: UUID):
            if document_id == bad:
                raise ValueError("malformed stored row")
            return real_read(document_id)

        monkeypatch.setattr(store, "read", flaky_read)

        seen = [d.id for d in store.iter_all()]

        assert bad not in seen
        assert seen == [good]

    def test_propagates_systemic_db_error(
        self,
        store: DocumentStore,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A systemic DB failure must propagate, not be swallowed as one bad
        row: otherwise a force reindex sees an empty corpus and deletes every
        point."""
        self._make_extracted(store, temp_dir, 1)

        def locked_read(document_id: UUID):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "read", locked_read)

        with pytest.raises(sqlite3.OperationalError):
            list(store.iter_all())
