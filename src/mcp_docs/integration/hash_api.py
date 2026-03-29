"""Hash-based document verification API for mcp-notes integration.

This module provides the integration layer between mcp-docs and mcp-notes,
allowing facts to reference documents by content hash and verify that
the referenced documents still exist and haven't changed.

Workflow:
1. When a fact references a document, it stores the document's content_hash
2. When verifying a fact's sources, lookup the hash to find the document
3. Verify the document still exists and hash matches (detect modifications)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import UUID

from mcp_docs.storage.database import DocumentStore, compute_file_hash

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of hash verification."""

    VALID = "valid"  # Hash matches, document exists
    MODIFIED = "modified"  # Document exists but hash changed
    MISSING = "missing"  # Document not found in registry
    FILE_DELETED = "file_deleted"  # Document registered but file doesn't exist


@dataclass
class HashVerificationResult:
    """Result of verifying a document hash."""

    content_hash: str
    status: VerificationStatus
    document_id: UUID | None = None
    path: str | None = None
    current_hash: str | None = None  # If modified, the new hash
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "content_hash": self.content_hash,
            "status": self.status.value,
            "document_id": str(self.document_id) if self.document_id else None,
            "path": self.path,
        }
        if self.current_hash and self.current_hash != self.content_hash:
            result["current_hash"] = self.current_hash
        if self.error:
            result["error"] = self.error
        return result


def lookup_document_by_hash(
    store: DocumentStore,
    content_hash: str,
) -> dict | None:
    """
    Look up a document by its content hash.

    Used by mcp-notes to resolve hash references to documents.

    Args:
        store: DocumentStore instance
        content_hash: SHA-256 content hash

    Returns:
        Document metadata dict if found, None otherwise
    """
    document = store.get_by_hash(content_hash)
    if document:
        return document.to_dict()
    return None


def verify_document_hash(
    store: DocumentStore,
    content_hash: str,
    check_file: bool = True,
) -> HashVerificationResult:
    """
    Verify a document hash reference.

    Checks:
    1. Document exists in registry
    2. File still exists on disk (if check_file=True)
    3. File content hash still matches (if check_file=True)

    Args:
        store: DocumentStore instance
        content_hash: SHA-256 content hash to verify
        check_file: Whether to verify the file on disk

    Returns:
        HashVerificationResult with status
    """
    # Look up document
    document = store.get_by_hash(content_hash)

    if document is None:
        return HashVerificationResult(
            content_hash=content_hash,
            status=VerificationStatus.MISSING,
        )

    result = HashVerificationResult(
        content_hash=content_hash,
        status=VerificationStatus.VALID,
        document_id=document.id,
        path=document.path,
    )

    # Optionally verify file on disk
    if check_file:
        file_path = Path(document.path)

        if not file_path.exists():
            result.status = VerificationStatus.FILE_DELETED
            result.error = f"File no longer exists: {document.path}"
            return result

        try:
            current_hash = compute_file_hash(file_path)
            if current_hash != content_hash:
                result.status = VerificationStatus.MODIFIED
                result.current_hash = current_hash
                result.error = "Document content has changed since indexed"
        except Exception as e:
            result.error = f"Error computing file hash: {e}"

    return result


def batch_verify_document_hashes(
    store: DocumentStore,
    content_hashes: list[str],
    check_files: bool = True,
) -> list[HashVerificationResult]:
    """
    Verify multiple document hashes.

    Args:
        store: DocumentStore instance
        content_hashes: List of SHA-256 content hashes
        check_files: Whether to verify files on disk

    Returns:
        List of HashVerificationResult for each hash
    """
    results = []
    for content_hash in content_hashes:
        result = verify_document_hash(store, content_hash, check_file=check_files)
        results.append(result)
    return results
