"""MCP Docs - Document management with vector search."""

__version__ = "1.1.22"


def main() -> None:
    """Run the MCP server."""
    from mcp_docs.server import main as _main

    _main()


__all__ = ["main", "__version__"]
