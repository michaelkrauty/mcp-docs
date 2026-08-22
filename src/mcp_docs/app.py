"""MCPServer application instance for mcp-docs."""

from mcp.server import MCPServer

from mcp_docs import __version__

mcp = MCPServer("mcp-docs", version=__version__)
