"""Document search and similarity operations.

Tools:
- search_documents: Search documents using hybrid semantic search
- find_similar_documents: Find documents similar to a given document
- get_document_chunks: Get all indexed chunks for a document
"""

from vector_core import parse_uuid_or_none, validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.singletons import get_search_engine


@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 10,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    include_chunks: bool = True,
) -> list[dict]:
    """
    Search documents using hybrid semantic search.

    Combines dense (meaning) and sparse (keyword) vectors for best results.

    Args:
        query: Natural language search query
        limit: Maximum results to return (default 10, max 100)
        doc_type: Filter by document type (pdf, docx, txt, md, etc.)
        tags: Filter by tags (document must have ALL tags)
        include_chunks: If True, search within document chunks. If False, only match whole documents.

    Returns:
        List of search results with relevance scores
    """
    engine = await get_search_engine()
    limit = validate_limit(limit)

    results = await engine.search(
        query=query,
        limit=limit,
        doc_type=doc_type,
        tags=tags,
        include_chunks=include_chunks,
    )

    return [r.to_dict() for r in results]


@mcp.tool()
async def find_similar_documents(
    document_id: str,
    limit: int = 5,
) -> list[dict] | dict:
    """
    Find documents similar to a given document.

    Uses vector similarity to find semantically related documents.

    Args:
        document_id: Document UUID string
        limit: Maximum results to return (default 5, max 100)

    Returns:
        List of similar documents with similarity scores, or error dict
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    engine = await get_search_engine()
    limit = validate_limit(limit, default=5)

    results = await engine.find_similar(
        document_id=uuid,
        limit=limit,
    )

    if not results:
        return error_response(ErrorCode.NOT_FOUND, f"Document not found in index: {document_id}")

    return [r.to_dict() for r in results]


@mcp.tool()
async def get_document_chunks(document_id: str) -> list[dict] | dict:
    """
    Get all indexed chunks for a document.

    Useful for seeing how a document was chunked for search.

    Args:
        document_id: Document UUID string

    Returns:
        List of document chunks ordered by position, or error dict
    """
    uuid = parse_uuid_or_none(document_id)
    if uuid is None:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid document ID: {document_id}")

    engine = await get_search_engine()
    chunks = await engine.get_document_chunks(uuid)

    return [c.to_dict() for c in chunks]
