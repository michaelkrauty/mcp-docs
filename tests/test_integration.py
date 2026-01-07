"""Tests for facts integration (hash verification)."""

import tempfile
from pathlib import Path

import pytest

from mcp_docs.integration import (
    HashVerificationResult,
    batch_verify_document_hashes,
    lookup_document_by_hash,
    verify_document_hash,
)
from mcp_docs.integration.hash_api import VerificationStatus
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
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World! This is a test document.")
    return file_path


class TestLookupDocumentByHash:
    """Tests for lookup_document_by_hash."""

    def test_lookup_existing(self, store: DocumentStore, sample_file: Path) -> None:
        """Can look up an existing document by hash."""
        content_hash = compute_file_hash(sample_file)
        store.register(sample_file, content_hash)

        result = lookup_document_by_hash(store, content_hash)

        assert result is not None
        assert result["content_hash"] == content_hash
        assert result["filename"] == "sample.txt"

    def test_lookup_nonexistent(self, store: DocumentStore) -> None:
        """Returns None for nonexistent hash."""
        result = lookup_document_by_hash(store, "nonexistent_hash")
        assert result is None


class TestVerifyDocumentHash:
    """Tests for verify_document_hash."""

    def test_verify_valid(self, store: DocumentStore, sample_file: Path) -> None:
        """Valid document returns VALID status."""
        content_hash = compute_file_hash(sample_file)
        store.register(sample_file, content_hash)

        result = verify_document_hash(store, content_hash)

        assert result.status == VerificationStatus.VALID
        assert result.document_id is not None
        assert result.path == str(sample_file)
        assert result.error is None

    def test_verify_missing(self, store: DocumentStore) -> None:
        """Nonexistent hash returns MISSING status."""
        result = verify_document_hash(store, "nonexistent_hash")

        assert result.status == VerificationStatus.MISSING
        assert result.document_id is None

    def test_verify_modified(self, store: DocumentStore, sample_file: Path) -> None:
        """Modified file returns MODIFIED status."""
        content_hash = compute_file_hash(sample_file)
        store.register(sample_file, content_hash)

        # Modify the file
        sample_file.write_text("Modified content")

        result = verify_document_hash(store, content_hash)

        assert result.status == VerificationStatus.MODIFIED
        assert result.current_hash is not None
        assert result.current_hash != content_hash

    def test_verify_file_deleted(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Deleted file returns FILE_DELETED status."""
        content_hash = compute_file_hash(sample_file)
        store.register(sample_file, content_hash)

        # Delete the file
        sample_file.unlink()

        result = verify_document_hash(store, content_hash)

        assert result.status == VerificationStatus.FILE_DELETED

    def test_verify_skip_file_check(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Can skip file verification."""
        content_hash = compute_file_hash(sample_file)
        store.register(sample_file, content_hash)

        # Delete the file but skip check
        sample_file.unlink()

        result = verify_document_hash(store, content_hash, check_file=False)

        # Should still be valid since we didn't check the file
        assert result.status == VerificationStatus.VALID


class TestBatchVerifyDocumentHashes:
    """Tests for batch_verify_document_hashes."""

    def test_batch_verify_multiple(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Can verify multiple hashes at once."""
        # Create multiple files
        files = []
        hashes = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)
            h = compute_file_hash(f)
            hashes.append(h)
            store.register(f, h)

        results = batch_verify_document_hashes(store, hashes)

        assert len(results) == 3
        assert all(r.status == VerificationStatus.VALID for r in results)

    def test_batch_verify_mixed_status(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Returns mixed statuses correctly."""
        # Create one file
        file1 = temp_dir / "file1.txt"
        file1.write_text("Content 1")
        hash1 = compute_file_hash(file1)
        store.register(file1, hash1)

        # Use a nonexistent hash
        hash2 = "nonexistent_hash"

        results = batch_verify_document_hashes(store, [hash1, hash2])

        assert len(results) == 2
        assert results[0].status == VerificationStatus.VALID
        assert results[1].status == VerificationStatus.MISSING


class TestHashVerificationResult:
    """Tests for HashVerificationResult."""

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        from uuid import uuid4

        result = HashVerificationResult(
            content_hash="abc123",
            status=VerificationStatus.VALID,
            document_id=uuid4(),
            path="/test/path",
        )

        d = result.to_dict()

        assert d["content_hash"] == "abc123"
        assert d["status"] == "valid"
        assert d["path"] == "/test/path"
        assert "document_id" in d

    def test_to_dict_with_modification(self) -> None:
        """to_dict includes current_hash when modified."""
        result = HashVerificationResult(
            content_hash="old_hash",
            status=VerificationStatus.MODIFIED,
            current_hash="new_hash",
        )

        d = result.to_dict()

        assert d["current_hash"] == "new_hash"
