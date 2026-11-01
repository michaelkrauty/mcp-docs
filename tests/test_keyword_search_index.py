"""keyword_search against a real Qdrant full-text index.

Qdrant's in-memory mode ignores payload indexes, so case-insensitive word and
phrase matching can only be observed on a server. Opt in with
MCP_DOCS_TEST_QDRANT_URL; the test owns a throwaway collection.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import PointStruct, VectorParams
from vector_core import QdrantStorage

from mcp_docs.search.engine import DocumentSearchEngine
from mcp_docs.tools.search import keyword_search

from .conftest import requires_qdrant

DOCUMENTS = {
    "a": ("Annual-Report-2024.pdf", "The Auditor reviewed the Annual Report with finance."),
    "b": ("notes.md", "Send the annual report to the auditors on Monday."),
    "c": ("YEAR_END_REPORT.md", "REPORT OF THE AUDITOR, filed in March."),
    "d": ("unrelated.txt", "Reports were handled by the audit team."),
}


@requires_qdrant
@pytest.mark.asyncio
async def test_keyword_search_ignores_case_and_matches_whole_words(test_collection_name):
    storage = QdrantStorage()
    client = await storage.get_client()
    await client.create_collection(
        test_collection_name, vectors_config=VectorParams(size=1, distance="Cosine")
    )
    try:
        await client.upsert(
            test_collection_name,
            [
                PointStruct(
                    id=index,
                    vector=[1.0],
                    payload={
                        "type": "doc_chunk",
                        "document_id": doc_id,
                        "filename": filename,
                        "path": f"/{filename}",
                        "doc_type": filename.rsplit(".", 1)[1],
                        "content": content,
                    },
                )
                for index, (doc_id, (filename, content)) in enumerate(DOCUMENTS.items())
            ],
            wait=True,
        )
        engine = DocumentSearchEngine(
            storage=storage,
            embedder=MagicMock(),
            global_vocab=MagicMock(),
            collection_name=test_collection_name,
        )

        @asynccontextmanager
        async def unbound(*, write):
            yield

        engine.collection_operation = unbound  # type: ignore[method-assign]

        async def found(keyword, **options):
            with patch(
                "mcp_docs.tools.search.get_search_engine",
                new=AsyncMock(return_value=engine),
            ):
                result = await keyword_search(keyword=keyword, **options)
            assert isinstance(result, list), result
            return {item["document_id"] for item in result}

        assert await found("auditor") == {"a", "c"}
        assert await found("AUDITOR") == {"a", "c"}
        assert await found("annual report") == {"a", "b"}
        # A phrase keeps its word order; whole words only.
        assert await found("report annual") == set()
        assert await found("aud") == set()
        assert await found("report", search_content=False) == {"a", "c"}
        assert await found("annual report", search_content=False) == {"a"}

        schema = (await client.get_collection(test_collection_name)).payload_schema
        assert {"content", "filename"} <= set(schema)
        assert all(schema[field].params.lowercase for field in ("content", "filename"))
    finally:
        await client.delete_collection(test_collection_name)
        await storage.close()
