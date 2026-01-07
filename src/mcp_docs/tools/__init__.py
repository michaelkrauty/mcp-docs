"""MCP Docs tool modules.

Modularized tool implementations for the mcp-docs server.
Each module handles a specific category of tools:

- documents.py: Core document CRUD operations
- processing.py: Document processing queue operations
- search.py: Search and similarity operations
- indexing.py: Document indexing operations
- roots.py: Document root management
- hashes.py: Hash verification for mcp-facts integration
- glossary.py: Glossary tools (using vector-core GlossaryToolHelper)

Tools are registered via @mcp.tool() decorator when modules are imported.
Import all modules in server.py to register all tools.
"""

# Import all tool modules to register their tools with mcp
from mcp_docs.tools import documents
from mcp_docs.tools import processing
from mcp_docs.tools import search
from mcp_docs.tools import indexing
from mcp_docs.tools import roots
from mcp_docs.tools import hashes
from mcp_docs.tools import glossary

__all__ = [
    "documents",
    "processing",
    "search",
    "indexing",
    "roots",
    "hashes",
    "glossary",
]
