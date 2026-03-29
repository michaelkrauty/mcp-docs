"""Document search and similarity operations.

Tools:
- search_documents: Search documents using hybrid semantic search
- keyword_search: Search documents using exact keyword matching
- find_similar_documents: Find documents similar to a given document
- get_document_chunks: Get all indexed chunks for a document
"""


from qdrant_client.models import FieldCondition, Filter, MatchText
from vector_core import parse_uuid_or_none, validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.settings import settings
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
        include_chunks: If True, search within document chunks.
            If False, only match whole documents.

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
async def keyword_search(
    keyword: str,
    limit: int = 100,
    doc_type: str | None = None,
    search_filename: bool = True,
    search_content: bool = True,
) -> list[dict]:
    """
    Search documents using exact keyword matching.

    Use this for proper nouns, specific terms, or exact phrase searches
    where semantic search may miss results.

    Args:
        keyword: Exact keyword or phrase to search for (case-insensitive)
        limit: Maximum results to return (default 100, max 1000)
        doc_type: Filter by document type (pdf, docx, txt, md, etc.)
        search_filename: Search in filenames (default True)
        search_content: Search in document content (default True)

    Returns:
        List of matching documents with their paths
    """
    if not keyword or not keyword.strip():
        return error_response(ErrorCode.VALIDATION_ERROR, "Keyword cannot be empty")

    keyword = keyword.strip()
    limit = min(max(1, limit), 1000)

    engine = await get_search_engine()
    await engine._ensure_components()
    client = await engine.storage.get_client()

    # Build filter conditions for keyword matching
    should_conditions = []
    if search_content:
        should_conditions.append(
            FieldCondition(key="content", match=MatchText(text=keyword))
        )
    if search_filename:
        should_conditions.append(
            FieldCondition(key="filename", match=MatchText(text=keyword))
        )

    if not should_conditions:
        return error_response(
            ErrorCode.VALIDATION_ERROR,
            "At least one of search_filename or search_content must be True",
        )

    # Build must conditions for additional filters
    must_conditions = []
    if doc_type:
        must_conditions.append(
            FieldCondition(key="doc_type", match={"value": doc_type})
        )

    # Construct the filter
    filter_dict = {"should": should_conditions}
    if must_conditions:
        filter_dict["must"] = must_conditions

    # Scroll through matching points
    scroll_result = await client.scroll(
        settings.collection_name,
        scroll_filter=Filter(**filter_dict),
        limit=limit * 2,  # Fetch more to account for duplicates
        with_payload=True,
    )

    points, _ = scroll_result

    # Deduplicate by document_id and build results
    seen_docs: set[str] = set()
    results = []

    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("document_id", "")

        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)

        results.append({
            "document_id": doc_id,
            "filename": payload.get("filename", ""),
            "path": payload.get("path", ""),
            "doc_type": payload.get("doc_type", ""),
            "title": payload.get("title"),
            "tags": payload.get("tags", []),
        })

        if len(results) >= limit:
            break

    return results


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
