"""MCP server for document management with vector search."""

import atexit
import logging

from vector_core import sync_cleanup_wrapper, verify_tools_registered

from mcp_docs.settings import validate_collection_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import mcp instance from app module (shared across tool modules)
from mcp_docs.app import mcp  # noqa: E402

# Import singletons module for cleanup access and backward-compatible re-exports
from mcp_docs.singletons import (  # noqa: E402, F401
    _document_indexer,
    _document_processor,
    _document_scanner,
    _document_store,
    _glossary_helper,
    _glossary_indexer,
    _glossary_store,
    _integrity_manager,
    _search_engine,
    cleanup_async_resources,
    # Re-export singleton getters for backward compatibility
    get_document_store,
    get_document_processor,
    get_document_scanner,
    get_search_engine,
    get_document_indexer,
    get_glossary_store,
    get_glossary_indexer,
    get_glossary_helper,
    get_integrity_manager,
)

# Re-export settings for backward compatibility
from mcp_docs.settings import settings  # noqa: E402, F401

# Import tool modules to register tools with mcp instance
from mcp_docs import tools  # noqa: E402, F401

# Expected tools for verification (catches silent import failures)
EXPECTED_TOOLS = [
    # Documents
    "register_document",
    "get_document",
    "get_document_by_hash",
    "update_document_tags",
    "delete_document",
    "list_documents",
    # Processing
    "get_processing_status",
    "list_queued_documents",
    "wait_for_document",
    "cancel_processing",
    # Search
    "search_documents",
    "find_similar_documents",
    "get_document_chunks",
    # Indexing
    "index_document",
    "index_all_documents",
    # Roots
    "add_document_root",
    "list_document_roots",
    "get_document_root",
    "remove_document_root",
    "scan_document_root",
    "scan_all_roots",
    # Hashes
    "lookup_hash",
    "verify_document_reference",
    "batch_verify_references",
    # Glossary
    "add_glossary_entry",
    "lookup_term",
    "search_glossary",
    "list_glossary",
    "update_glossary_entry",
    "delete_glossary_entry",
]

# Re-export tools for backward compatibility with tests
from mcp_docs.tools.documents import (  # noqa: E402, F401
    register_document,
    get_document,
    get_document_by_hash,
    update_document_tags,
    delete_document,
    list_documents,
)
from mcp_docs.tools.processing import (  # noqa: E402, F401
    get_processing_status,
    list_queued_documents,
    wait_for_document,
    cancel_processing,
)
from mcp_docs.tools.search import (  # noqa: E402, F401
    search_documents,
    find_similar_documents,
    get_document_chunks,
)
from mcp_docs.tools.indexing import (  # noqa: E402, F401
    index_document,
    index_all_documents,
)
from mcp_docs.tools.roots import (  # noqa: E402, F401
    add_document_root,
    list_document_roots,
    get_document_root,
    remove_document_root,
    scan_document_root,
    scan_all_roots,
)
from mcp_docs.tools.hashes import (  # noqa: E402, F401
    lookup_hash,
    verify_document_reference,
    batch_verify_references,
)
from mcp_docs.tools.glossary import (  # noqa: E402, F401
    add_glossary_entry,
    lookup_term,
    search_glossary,
    list_glossary,
    update_glossary_entry,
    delete_glossary_entry,
)


# ============= Cleanup =============


def _cleanup_sync_resources() -> None:
    """Clean up sync singletons (database connections, etc.)."""
    # Close document store
    store = _document_store.get_if_initialized()
    if store is not None:
        try:
            store.close()
        except Exception:
            pass

    # Close glossary store
    gstore = _glossary_store.get_if_initialized()
    if gstore is not None:
        try:
            gstore.close()
        except Exception:
            pass

    # Note: SourceIntegrityManager's FactStore is managed by vector-core


def _sync_cleanup() -> None:
    """Sync wrapper for cleanup, called on exit."""
    # Check if there's anything to clean up first
    async_singletons = [
        _document_processor,
        _document_scanner,
        _document_indexer,
        _search_engine,
        _glossary_indexer,
        _glossary_helper,
    ]
    sync_singletons = [_document_store, _glossary_store, _integrity_manager]

    has_async = any(s.is_initialized for s in async_singletons)
    has_sync = any(s.is_initialized for s in sync_singletons)

    if not has_async and not has_sync:
        return

    # Clean up async resources using sync_cleanup_wrapper
    if has_async:
        sync_cleanup_wrapper(cleanup_async_resources, async_singletons)

    # Clean up sync resources
    if has_sync:
        _cleanup_sync_resources()


# Register cleanup handler
atexit.register(_sync_cleanup)


# ============= Main =============


def main() -> None:
    """Run the MCP server."""
    # Validate collection name at startup
    validate_collection_name()

    # Verify all expected tools are registered (catches silent import failures)
    verify_tools_registered(mcp, EXPECTED_TOOLS, "mcp-docs")

    # Run the server - cleanup is handled by atexit.register(_sync_cleanup)
    mcp.run()


if __name__ == "__main__":
    main()
