"""Document indexing operations.

Tools:
- index_document: Index a single document for search
- index_all_documents: Index all extracted documents
"""

import logging
from pathlib import Path

from vector_core import parse_uuid_or_none
from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.models import ExtractionStatus, DocumentType
from mcp_docs.singletons import get_document_indexer, get_document_store

logger = logging.getLogger(__name__)


@mcp.tool()
async def index_document(document_id: str) -> dict:
    """
    Index a single document for search.

    The document must already be extracted (extraction_status=extracted).
    Creates dense and sparse vectors for hybrid search.

    Args:
        document_id: Document UUID string

    Returns:
        Indexing result with points count, or error dict
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    store = get_document_store()
    document = store.read(uuid)

    if document is None:
        return error_response(ErrorCode.NOT_FOUND, f"Document not found: {document_id}")

    if document.extraction_status != ExtractionStatus.EXTRACTED:
        return {
            "error": f"Document not ready for indexing. Status: {document.extraction_status.value}",
            "hint": "Wait for extraction to complete or use wait_for_document first.",
        }

    try:
        indexer = await get_document_indexer()

        # Re-extract content for indexing (not stored in DB)
        from mcp_docs.extraction.extractor import extract_content

        path = Path(document.path)
        if not path.exists():
            return error_response(ErrorCode.FILE_NOT_FOUND, f"Document file not found: {document.path}")

        extracted = extract_content(path, DocumentType(document.doc_type))
        points = await indexer.index_document(uuid, extracted.text)

        # Update status
        store.update(uuid, extraction_status=ExtractionStatus.INDEXED)

        return {
            "success": True,
            "document_id": document_id,
            "points_indexed": points,
            "status": "indexed",
        }
    except Exception as e:
        logger.error(f"Failed to index document {document_id}: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Indexing failed: {e}")


@mcp.tool()
async def index_all_documents(force: bool = False) -> dict:
    """
    Index all extracted documents for search.

    Uses two-pass indexing:
    1. Collect tokens from all documents for vocabulary training
    2. Generate embeddings and sparse vectors for each document

    Args:
        force: If True, reindex all documents. If False, only index new/changed documents.

    Returns:
        Indexing result with counts and any errors
    """
    try:
        indexer = await get_document_indexer()
        result = await indexer.index_all(force=force)
        return result
    except Exception as e:
        logger.error(f"Failed to index documents: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Indexing failed: {e}")
