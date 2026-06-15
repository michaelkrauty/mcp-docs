"""Document root management operations.

Tools:
- add_document_root: Add a document root directory for scanning
- list_document_roots: List all registered document roots
- get_document_root: Get info about a specific document root
- remove_document_root: Remove a document root
- scan_document_root: Trigger a scan of a document root
- scan_all_roots: Trigger a scan of all enabled document roots
"""

from pathlib import Path

from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.singletons import (
    get_document_indexer,
    get_document_processor,
    get_document_scanner,
    get_document_store,
)
from mcp_docs.tools.documents import delete_document_artifacts


@mcp.tool()
async def add_document_root(
    path: str,
    name: str | None = None,
    recursive: bool = True,
    auto_scan: bool = True,
) -> dict:
    """
    Add a document root directory for scanning.

    Document roots are directories that will be monitored for new documents.
    All supported files in the directory will be registered and indexed.

    Args:
        path: Absolute path to the directory
        name: Optional friendly name for the root
        recursive: Whether to scan subdirectories (default True)
        auto_scan: Whether to scan immediately after adding (default True)

    Returns:
        Created root info or error dict
    """
    root_path = Path(path).resolve()

    if not root_path.exists():
        return error_response(ErrorCode.FILE_NOT_FOUND, f"Directory not found: {path}")

    if not root_path.is_dir():
        return error_response(ErrorCode.INVALID_INPUT, f"Not a directory: {path}")

    store = get_document_store()

    # Check if already registered
    existing = store.get_root(str(root_path))
    if existing:
        return {
            **existing.to_dict(),
            "already_registered": True,
        }

    # Add the root
    root = store.add_root(
        path=str(root_path),
        name=name,
        recursive=recursive,
    )

    result = root.to_dict()

    # Optionally scan immediately
    if auto_scan:
        try:
            scanner = await get_document_scanner()
            processor = await get_document_processor()
            indexer = await get_document_indexer()

            async def enqueue_doc(doc_id, file_path):
                await processor.enqueue(doc_id, file_path)

            async def delete_doc_index(doc_id):
                await indexer.delete_document_index(doc_id)

            async def relocate_doc_index(doc_id, old_path, new_path):
                await indexer.update_document_path_in_index(doc_id, new_path)

            scan_result = await scanner.scan_root(
                root,
                enqueue_callback=enqueue_doc,
                delete_callback=delete_doc_index,
                relocate_callback=relocate_doc_index,
            )
            result["scan_result"] = scan_result.to_dict()
        except Exception as e:
            result["scan_error"] = str(e)

    return result


@mcp.tool()
async def list_document_roots() -> list[dict]:
    """
    List all registered document roots.

    Returns:
        List of document root info dicts
    """
    store = get_document_store()
    roots = store.list_roots()
    return [r.to_dict() for r in roots]


@mcp.tool()
async def get_document_root(path: str) -> dict:
    """
    Get info about a specific document root.

    Args:
        path: Path to the document root

    Returns:
        Root info dict or error
    """
    root_path = Path(path).resolve()
    store = get_document_store()

    root = store.get_root(str(root_path))
    if root is None:
        return error_response(ErrorCode.NOT_FOUND, f"Root not found: {path}")

    return root.to_dict()


@mcp.tool()
async def remove_document_root(
    path: str,
    delete_documents: bool = False,
) -> dict:
    """
    Remove a document root.

    By default, keeps registered documents but they won't be monitored.

    Args:
        path: Path to the document root
        delete_documents: If True, also delete all documents from this root

    Returns:
        Success status or error
    """
    root_path = Path(path).resolve()
    store = get_document_store()

    # Check if exists
    root = store.get_root(str(root_path))
    if root is None:
        return error_response(ErrorCode.NOT_FOUND, f"Root not found: {path}")

    # Optionally delete documents. Remove each document fully — its
    # vector-index points and fact-source links, not just the registry row —
    # so deleting a root never orphans searchable points in Qdrant.
    deleted_count = 0
    sources_marked = 0
    if delete_documents:
        docs = store.list_summaries(document_root=str(root_path), limit=10000)
        for doc in docs:
            sources_marked += await delete_document_artifacts(
                store, doc.id, doc.content_hash
            )
            deleted_count += 1

    # Remove the root
    store.remove_root(str(root_path))

    return {
        "success": True,
        "removed_path": str(root_path),
        "documents_deleted": deleted_count if delete_documents else None,
        "sources_marked_deleted": sources_marked if delete_documents else None,
    }


@mcp.tool()
async def scan_document_root(path: str) -> dict:
    """
    Trigger a scan of a document root.

    Finds new files, detects modified files, and marks deleted files.

    Args:
        path: Path to the document root to scan

    Returns:
        Scan result with statistics
    """
    root_path = Path(path).resolve()
    store = get_document_store()

    root = store.get_root(str(root_path))
    if root is None:
        return error_response(ErrorCode.NOT_FOUND, f"Root not found: {path}")

    scanner = await get_document_scanner()
    processor = await get_document_processor()
    indexer = await get_document_indexer()

    async def enqueue_doc(doc_id, file_path):
        await processor.enqueue(doc_id, file_path)

    async def delete_doc_index(doc_id):
        await indexer.delete_document_index(doc_id)

    async def relocate_doc_index(doc_id, old_path, new_path):
        await indexer.update_document_path_in_index(doc_id, new_path)

    result = await scanner.scan_root(
        root,
        enqueue_callback=enqueue_doc,
        delete_callback=delete_doc_index,
        relocate_callback=relocate_doc_index,
    )
    return result.to_dict()


@mcp.tool()
async def scan_all_roots() -> list[dict]:
    """
    Trigger a scan of all enabled document roots.

    Returns:
        List of scan results for each root
    """
    scanner = await get_document_scanner()
    processor = await get_document_processor()
    indexer = await get_document_indexer()

    async def enqueue_doc(doc_id, file_path):
        await processor.enqueue(doc_id, file_path)

    async def delete_doc_index(doc_id):
        await indexer.delete_document_index(doc_id)

    async def relocate_doc_index(doc_id, old_path, new_path):
        await indexer.update_document_path_in_index(doc_id, new_path)

    results = await scanner.scan_all_roots(
        enqueue_callback=enqueue_doc,
        delete_callback=delete_doc_index,
        relocate_callback=relocate_doc_index,
    )
    return [r.to_dict() for r in results]
