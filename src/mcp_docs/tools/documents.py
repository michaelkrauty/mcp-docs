"""Core document CRUD operations.

Tools:
- register_document: Register a document for indexing
- get_document: Get document metadata by ID
- get_document_by_hash: Get document by content hash
- update_document_tags: Update tags for a document
- delete_document: Delete a document from the registry
- list_documents: List registered documents with filters
"""

import logging
from pathlib import Path
from uuid import UUID

from vector_core import parse_uuid_or_none, validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.models import DocumentStatus, ExtractionStatus
from mcp_docs.singletons import (
    get_document_indexer,
    get_document_processor,
    get_document_store,
    get_integrity_manager,
)
from mcp_docs.storage.database import DocumentStore, compute_file_hash
from mcp_docs.tools._validation import validate_doc_type

logger = logging.getLogger(__name__)


@mcp.tool()
async def register_document(
    path: str,
    tags: list[str] | None = None,
) -> dict:
    """
    Register a document for indexing.

    Computes content hash to detect duplicates. If the same content
    exists under a different path, returns the existing document.

    Args:
        path: Absolute path to the document file
        tags: Optional list of tags to apply

    Returns:
        Document metadata dict with id, status, and extraction_status
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        return error_response(ErrorCode.FILE_NOT_FOUND, f"File not found: {path}")

    if not file_path.is_file():
        return error_response(ErrorCode.INVALID_INPUT, f"Not a file: {path}")

    store = get_document_store()

    # Compute content hash
    content_hash = compute_file_hash(file_path)

    # Check existing before registration for the "already_registered" flag and
    # to detect a path change (register() handles race conditions atomically via
    # INSERT OR IGNORE).
    existing = store.get_by_hash(content_hash)
    was_registered = existing is not None
    old_path = existing.path if existing else None

    # Register document (atomically handles duplicates)
    document = store.register(
        path=file_path,
        content_hash=content_hash,
        tags=tags or [],
    )

    # Return with already_registered flag if it was pre-existing
    if was_registered:
        # register() moves the registry path when the same content is
        # re-registered at a new location, but the Qdrant payloads still carry
        # the old path. Sync the index (as move_file does) so search does not
        # return a path that no longer exists while get_document returns the new
        # one.
        if old_path is not None and document.path != old_path:
            try:
                indexer = await get_document_indexer()
                await indexer.update_document_path_in_index(
                    document.id, document.path
                )
            except Exception as e:
                logger.warning(f"Failed to sync index path for {document.id}: {e}")
        return {
            **document.to_dict(),
            "already_registered": True,
        }

    # Enqueue for background processing
    try:
        processor = await get_document_processor()
        await processor.enqueue(document.id, file_path)
    except Exception as e:
        logger.warning(f"Failed to enqueue document for processing: {e}")

    return document.to_dict()


@mcp.tool()
async def get_document(document_id: str) -> dict:
    """
    Get document metadata by ID.

    Args:
        document_id: Document UUID string

    Returns:
        Document metadata dict or error
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    store = get_document_store()
    document = store.read(uuid)

    if document is None:
        return error_response(ErrorCode.NOT_FOUND, f"Document not found: {document_id}")

    return document.to_dict()


@mcp.tool()
async def get_document_by_hash(content_hash: str) -> dict:
    """
    Get document metadata by content hash.

    Useful when you have a file and want to check if it's registered.

    Args:
        content_hash: SHA-256 hash of file content

    Returns:
        Document metadata dict or error
    """
    store = get_document_store()
    document = store.get_by_hash(content_hash)

    if document is None:
        return error_response(
            ErrorCode.NOT_FOUND,
            f"Document not found with hash: {content_hash[:16]}...",
        )

    return document.to_dict()


@mcp.tool()
async def update_document_tags(
    document_id: str,
    tags: list[str],
) -> dict:
    """
    Update tags for a document.

    Replaces all existing tags with the provided list.

    Args:
        document_id: Document UUID string
        tags: New list of tags (replaces existing)

    Returns:
        Updated document metadata dict or error
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    store = get_document_store()

    # Check document exists
    document = store.read(uuid)
    if document is None:
        return error_response(ErrorCode.NOT_FOUND, f"Document not found: {document_id}")

    # Update tags (update_tags returns the refreshed document).
    updated = store.update_tags(uuid, tags)

    # Keep the vector-index payload in sync so tag filters and result metadata
    # reflect the new tags; otherwise search keeps matching and showing the
    # document's previous tags until a full reindex.
    try:
        indexer = await get_document_indexer()
        await indexer.update_document_tags_in_index(updated)
    except Exception as e:
        logger.warning(f"Failed to sync tags to index for {document_id}: {e}")

    return updated.to_dict()


async def delete_document_artifacts(
    store: DocumentStore,
    document_id: UUID,
    content_hash: str | None,
) -> int:
    """Remove one document's side effects, then its registry row.

    Marks any fact sources that cite the document's content hash as deleted,
    removes the document's points from the vector index, and deletes the
    registry row. Index and fact-source cleanup are best-effort (failures are
    logged) so a transient backend error never blocks removal of the row, but
    the vector-index cleanup is always attempted. Returns the number of fact
    sources marked deleted.

    Shared by delete_document and remove_document_root so both deletion paths
    clean up identically; a prior divergence left remove_document_root deleting
    registry rows while orphaning their vector-index points.
    """
    sources_marked = 0
    if content_hash:
        try:
            integrity = get_integrity_manager()
            sources_marked = integrity.mark_document_deleted(content_hash)
            if sources_marked > 0:
                logger.info(
                    f"Marked {sources_marked} fact sources as deleted "
                    f"for document {content_hash[:16]}..."
                )
        except Exception as e:
            logger.warning(f"Failed to mark fact sources as deleted: {e}")

    try:
        indexer = await get_document_indexer()
        await indexer.delete_document_index(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete document index: {e}")

    store.delete(document_id)
    return sources_marked


@mcp.tool()
async def delete_document(document_id: str) -> dict:
    """
    Delete a document from the registry.

    Does not delete the actual file, only removes from registry.

    Args:
        document_id: Document UUID string

    Returns:
        Success status or error
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    store = get_document_store()

    # Check document exists
    document = store.read(uuid)
    if document is None:
        return error_response(ErrorCode.NOT_FOUND, f"Document not found: {document_id}")

    sources_marked = await delete_document_artifacts(store, uuid, document.content_hash)

    return {
        "success": True,
        "deleted_id": document_id,
        "deleted_path": document.path,
        "sources_marked_deleted": sources_marked,
    }


@mcp.tool()
async def list_documents(
    tags: list[str] | None = None,
    status: str | None = None,
    extraction_status: str | None = None,
    doc_type: str | None = None,
    document_root: str | None = None,
    limit: int = 50,
) -> list[dict] | dict:
    """
    List registered documents with optional filters.

    Args:
        tags: Filter by tags (document must have ALL tags)
        status: Filter by document status (active, modified, relocated, deleted)
        extraction_status: Filter by extraction status (queued, processing, extracted, etc.)
        doc_type: Filter by document type (pdf, docx, txt, md, etc.)
        document_root: Filter by document root path
        limit: Maximum documents to return (default 50)

    Returns:
        List of document summary dicts, or error dict on invalid input
    """
    store = get_document_store()
    limit = validate_limit(limit, default=50)

    # Parse status enum if provided
    status_enum = None
    if status:
        try:
            status_enum = DocumentStatus(status.lower())
        except ValueError:
            valid = ", ".join(s.value for s in DocumentStatus)
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid status: {status}. Valid values: {valid}",
            )

    extraction_enum = None
    if extraction_status:
        try:
            extraction_enum = ExtractionStatus(extraction_status.lower())
        except ValueError:
            valid = ", ".join(s.value for s in ExtractionStatus)
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid extraction_status: {extraction_status}. Valid values: {valid}",
            )

    doc_type_value, doc_type_error = validate_doc_type(doc_type)
    if doc_type_error is not None:
        return doc_type_error

    summaries = store.list_summaries(
        tags=tags,
        status=status_enum,
        extraction_status=extraction_enum,
        doc_type=doc_type_value,
        document_root=document_root,
        limit=limit,
    )

    return [s.to_dict() for s in summaries]
