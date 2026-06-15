"""Document processing queue operations.

Tools:
- get_processing_status: Get processing status for a document
- list_queued_documents: List documents in processing queue
- wait_for_document: Wait for document processing to complete
- cancel_processing: Cancel queued document processing
"""

from vector_core import parse_uuid_or_none
from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.singletons import get_document_processor


@mcp.tool()
async def get_processing_status(document_id: str) -> dict:
    """
    Get processing status for a document.

    Args:
        document_id: Document UUID string

    Returns:
        Status dict with processing state, queue position, etc.
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    processor = await get_document_processor()
    return processor.get_status(uuid)


@mcp.tool()
async def list_queued_documents() -> list[dict]:
    """
    List documents currently in the processing queue.

    Returns:
        List of queued/processing documents with their status
    """
    processor = await get_document_processor()
    return processor.list_queued()


@mcp.tool()
async def wait_for_document(
    document_id: str,
    timeout: float = 300.0,
) -> dict:
    """
    Wait for document processing to complete.

    Blocks until the document is fully processed or timeout expires.

    Args:
        document_id: Document UUID string
        timeout: Maximum time to wait in seconds (default 300s = 5 min)

    Returns:
        Processing result dict or error on timeout
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    processor = await get_document_processor()
    result = await processor.wait_for(uuid, timeout=timeout)

    if result is None:
        return error_response(
            ErrorCode.TIMEOUT,
            f"Timeout waiting for document {document_id} to process",
            details={"document_id": document_id, "timeout_seconds": timeout},
        )

    return result.to_dict()


@mcp.tool()
async def cancel_processing(document_id: str) -> dict:
    """
    Cancel a queued document's processing.

    Only documents that are still queued can be cancelled; a document that is
    already being processed, extracted, indexed, failed, or cancelled is left
    unchanged.

    Args:
        document_id: Document UUID string

    Returns:
        Success status or error
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    processor = await get_document_processor()
    success = processor.cancel(uuid)

    if success:
        return {"success": True, "document_id": document_id}
    else:
        return error_response(
            ErrorCode.CONFLICT,
            "Cannot cancel: document is not queued (it may be processing, "
            "already complete, failed, or cancelled)",
            details={"document_id": document_id},
        )
