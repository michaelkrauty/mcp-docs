"""MCP Docs tool modules.

Modularized tool implementations for the mcp-docs server.
Each module handles a specific category of tools:

- documents.py: Core document CRUD operations
- processing.py: Document processing queue operations
- search.py: Search and similarity operations
- indexing.py: Document indexing operations
- roots.py: Document root management
- hashes.py: Hash verification for mcp-notes integration
- glossary.py: Glossary tools (using vector-core GlossaryToolHelper)
- filesystem.py: Filesystem management tools (move, create, rename, delete)

Tools are registered via @mcp.tool() decorator when modules are imported.
Import all modules in server.py to register all tools.
"""

# Import all tool modules to register their tools with mcp
from mcp_docs.tools import (
    documents,
    filesystem,  # noqa: F401
    glossary,
    hashes,
    indexing,
    processing,
    roots,
    search,
)

__all__ = [
    "documents",
    "processing",
    "search",
    "indexing",
    "roots",
    "hashes",
    "glossary",
    "filesystem",
]
