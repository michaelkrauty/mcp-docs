"""Document indexing module."""

from mcp_docs.indexing.chunker import DocumentChunker, chunk_document
from mcp_docs.indexing.indexer import DocumentIndexer

__all__ = [
    "DocumentChunker",
    "DocumentIndexer",
    "chunk_document",
]
