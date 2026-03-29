"""FastMCP application instance for mcp-docs."""

from mcp.server.fastmcp import FastMCP

from mcp_docs import __version__

mcp = FastMCP("mcp-docs")
mcp._mcp_server.version = __version__
