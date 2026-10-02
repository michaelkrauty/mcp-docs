"""Bounded, complete similarity queries and released-core fragment coordinates."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, SparseVector, SparseVectorParams, VectorParams
from vector_core import HybridSearcher
from vector_core.storage.embedding_fragments import fragment_point

from mcp_docs.search.engine import DocumentSearchEngine


def source_points(start, count):
    return [
        SimpleNamespace(id=i, vector={"dense": [float(i)]}) for i in range(start, start + count)
    ]


def grouped_hit(document_id, point_id, score):
    return SimpleNamespace(
        groups=[
            SimpleNamespace(id=str(document_id), hits=[SimpleNamespace(id=point_id, score=score)])
        ]
    )


def fake_engine(client):
    storage = AsyncMock()
    storage.get_client.return_value = client
    return DocumentSearchEngine(storage, AsyncMock(), MagicMock(), "coverage")


@pytest.mark.usefixtures("embedding_generation")
async def test_similarity_overlaps_eight_queries_and_covers_every_page():
    source_id, target_id = uuid4(), uuid4()
    client = AsyncMock()
    client.scroll.side_effect = [
        (source_points(0, 19), 19),
        (source_points(19, 10), None),
    ]
    batch_sizes = iter([8, 8, 3, 8, 2])
    expected = next(batch_sizes)
    barrier = asyncio.Event()
    active = peak = started = 0
    queried = []

    async def query(collection, **kwargs):
        nonlocal active, peak, started, expected, barrier
        assert collection == "coverage"
        vector_id = int(kwargs["query"][0])
        queried.append(vector_id)
        active += 1
        peak = max(peak, active)
        started += 1
        gate = barrier
        if started == expected:
            gate.set()
        await gate.wait()
        active -= 1
        if active == 0:
            started = 0
            expected = next(batch_sizes, 0)
            barrier = asyncio.Event()
        return grouped_hit(target_id, 100 + vector_id, vector_id / 29)

    client.query_points_groups.side_effect = query
    client.retrieve.return_value = [
        SimpleNamespace(id=128, payload={"type": "doc_chunk", "content": "last-page winner"})
    ]
    results = await asyncio.wait_for(fake_engine(client).find_similar(source_id, limit=1), 2)
    assert peak == 8
    assert active == 0
    assert queried == list(range(29))
    assert client.scroll.await_args_list[1].kwargs["offset"] == 19
    for call in client.query_points_groups.await_args_list:
        assert call.kwargs["group_by"] == "document_id"
        assert call.kwargs["group_size"] == 1
        assert call.kwargs["query_filter"].must_not[0].match.value == str(source_id)
        assert call.kwargs["with_payload"] is False
    client.retrieve.assert_awaited_once_with(
        "coverage", ids=[128], with_payload=True, with_vectors=False
    )
    assert results[0].document_id == target_id
    assert results[0].content == "last-page winner"
    assert results[0].score == pytest.approx(28 / 29)


@pytest.mark.usefixtures("embedding_generation")
async def test_query_failure_cancels_and_joins_siblings_without_hydrating_partial_results():
    source_id, target_id = uuid4(), uuid4()
    client = AsyncMock()
    client.scroll.return_value = (source_points(0, 24), None)
    barrier = asyncio.Event()
    started, cancelled, drained = [], [], []
    tasks = []

    async def query(collection, **kwargs):
        vector_id = int(kwargs["query"][0])
        if vector_id < 8:
            return grouped_hit(target_id, 100, 0.9)
        tasks.append(asyncio.current_task())
        started.append(vector_id)
        if len(started) == 8:
            barrier.set()
        try:
            await barrier.wait()
            if vector_id == 8:
                raise RuntimeError("query failed")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(vector_id)
            raise
        finally:
            # Joining must wait for asynchronous cleanup, not merely send cancellation.
            await asyncio.sleep(0)
            drained.append(vector_id)

    client.query_points_groups.side_effect = query
    with pytest.raises(RuntimeError, match="query failed"):
        await asyncio.wait_for(fake_engine(client).find_similar(source_id), 2)
    assert started == list(range(8, 16))
    assert sorted(cancelled) == list(range(9, 16))
    assert sorted(drained) == list(range(8, 16))
    assert all(task.done() for task in tasks)
    assert client.query_points_groups.await_count == 16
    client.retrieve.assert_not_awaited()


class FragmentEmbedder:
    @staticmethod
    def split_text(text, *, role):
        return [
            SimpleNamespace(start=0, end=7, text=text[:7]),
            SimpleNamespace(start=7, end=len(text), text=text[7:]),
        ]

    @staticmethod
    async def embed_all(texts, *, role):
        return [[1.0, 0.0], [0.0, 1.0]]

    @staticmethod
    async def embed_single_cached(text, *, role):
        return [0.0, 1.0]


@pytest.mark.usefixtures("embedding_generation")
async def test_real_fragment_hit_offsets_are_composed_once_from_nonzero_parent():
    original = "unindexed prefix: " + "head   " + "derived exact evidence\n終"
    parent_start = len("unindexed prefix: ")
    body = original[parent_start:]
    doc_id = uuid4()
    embedder = FragmentEmbedder()
    empty_sparse = SparseVector(indices=[], values=[])
    points = await fragment_point(
        embedder,
        point_id=42,
        payload={
            "document_id": str(doc_id),
            "type": "doc_chunk",
            "content": body,
            "embedding_text_field": "content",
            "char_start": parent_start,
            "char_end": len(original),
        },
        sparse=empty_sparse,
        vectorize=lambda _: empty_sparse,
    )
    child = points[1].payload
    assert child["char_start"] == parent_start + 7
    assert child["char_end"] == len(original)
    client = AsyncQdrantClient(":memory:")
    storage = MagicMock()
    storage.get_client = AsyncMock(return_value=client)
    storage._get_client = AsyncMock(return_value=client)
    vocab = MagicMock()
    vocab.vectorize_query.return_value = empty_sparse
    engine = DocumentSearchEngine(storage, embedder, vocab, "coverage")
    engine._searcher = HybridSearcher(storage, dense_weight=1.0, sparse_weight=0.0)
    try:
        await client.create_collection(
            "coverage",
            vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        await client.upsert("coverage", points=points)
        result = (await engine.search("derived", limit=1))[0]
        assert result.document_id == doc_id
        assert result.embedding_span == {"start": 7, "end": len(body)}
        assert result.char_start == parent_start + 7
        assert result.char_end == len(original)
        assert result.content == original[result.char_start : result.char_end]
        assert result.content == "derived exact evidence\n終"
    finally:
        await client.close()
