"""Facts integration module for document verification."""

from mcp_docs.integration.hash_api import (
    HashVerificationResult,
    batch_verify_document_hashes,
    lookup_document_by_hash,
    verify_document_hash,
)

__all__ = [
    "HashVerificationResult",
    "lookup_document_by_hash",
    "verify_document_hash",
    "batch_verify_document_hashes",
]
