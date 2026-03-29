"""Hash verification tools for mcp-notes integration.

Tools:
- lookup_hash: Look up a document by its content hash
- verify_document_reference: Verify a document reference by hash
- batch_verify_references: Verify multiple document references
"""

from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.integration import (
    batch_verify_document_hashes,
    lookup_document_by_hash,
    verify_document_hash,
)
from mcp_docs.singletons import get_document_store


@mcp.tool()
async def lookup_hash(content_hash: str) -> dict:
    """
    Look up a document by its content hash.

    Used by mcp-notes to resolve document references stored as hashes.

    Args:
        content_hash: SHA-256 content hash of the document

    Returns:
        Document metadata if found, or error dict
    """
    store = get_document_store()
    result = lookup_document_by_hash(store, content_hash)

    if result is None:
        return error_response(
            ErrorCode.NOT_FOUND,
            f"No document found with hash: {content_hash[:16]}...",
        )

    return result


@mcp.tool()
async def verify_document_reference(
    content_hash: str,
    check_file: bool = True,
) -> dict:
    """
    Verify a document reference by its content hash.

    Checks that:
    1. A document with this hash exists in the registry
    2. The file still exists on disk (if check_file=True)
    3. The file content hasn't changed (if check_file=True)

    Useful for validating fact sources that reference documents.

    Args:
        content_hash: SHA-256 content hash to verify
        check_file: Whether to verify the actual file on disk

    Returns:
        Verification result with status (valid, modified, missing, file_deleted)
    """
    store = get_document_store()
    result = verify_document_hash(store, content_hash, check_file=check_file)
    return result.to_dict()


@mcp.tool()
async def batch_verify_references(
    content_hashes: list[str],
    check_files: bool = True,
) -> list[dict]:
    """
    Verify multiple document references by their content hashes.

    Useful for validating all sources of a fact at once.

    Args:
        content_hashes: List of SHA-256 content hashes to verify
        check_files: Whether to verify actual files on disk

    Returns:
        List of verification results
    """
    store = get_document_store()
    results = batch_verify_document_hashes(store, content_hashes, check_files=check_files)
    return [r.to_dict() for r in results]
