"""Full retained-source coverage, exercised without external services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams
from vector_core import HybridSearcher, QdrantStorage
from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.embedding_fragments import is_derived_fragment, source_hash

from mcp_docs.indexing.chunker import DocumentChunker
from mcp_docs.indexing.indexer import SOURCE_LAYOUT_VERSION, DocumentIndexer
from mcp_docs.models import ExtractedContent, ExtractionStatus
from mcp_docs.search.engine import DocumentSearchEngine, _matched_payload


@given(st.text(min_size=1, max_size=1500), st.integers(min_value=1, max_value=100))
@settings(max_examples=100, deadline=None)
def test_passages_cover_exact_source(text, budget):
    chunks = DocumentChunker(max_chars=budget).chunk(uuid4(), text).chunks
    if not text.strip():
        assert chunks == []
        return
    covered = 0
    for chunk in chunks:
        assert chunk.content == text[chunk.char_start : chunk.char_end]
        assert chunk.char_start <= covered
        assert chunk.char_end > covered
        assert len(chunk.content) <= budget
        covered = chunk.char_end
    assert covered == len(text)


@pytest.mark.parametrize("section", [False, True])
def test_giant_unit_headings_whitespace_and_tail_survive(section):
    source = ("# RETAINED HEADING\n\t " if section else "\t ") + "a" * 240_000
    source += "\n\n# DISTINCTIVE TAIL\n  最終 evidence\t\n"
    chunks = DocumentChunker(max_chars=8192, overlap_chars=0).chunk(uuid4(), source).chunks
    assert "".join(chunk.content for chunk in chunks) == source
    assert all(len(chunk.content) <= 8192 for chunk in chunks)


def test_search_evidence_slices_canonical_without_mutating_retained_body():
    body = "prefix " + "retained evidence " * 1000
    payload = {
        "content": body,
        "embedding_text_field": "content",
        "char_start": 500,
        "char_end": 500 + len(body),
        "embedding_fragment": {
            "schema": 1,
            "parent_id": 42,
            "index": 0,
            "count": 2,
            "start": 0,
            "end": 7,
            "source_hash": source_hash(body),
        },
    }
    matched = _matched_payload(payload)
    assert matched["content"] == "prefix "
    assert (matched["char_start"], matched["char_end"]) == (500, 507)
    assert payload["content"] == body
    assert payload["char_end"] == 500 + len(body)


def test_canonical_sparse_tail_excerpt_is_distinct_from_dense_span():
    text = "orchard material " * 1000 + "stellar plasma evidence"
    payload = {
        "content": text,
        "embedding_text_field": "content",
        "char_start": 300,
        "embedding_fragment": {
            "schema": 1,
            "parent_id": 42,
            "index": 0,
            "count": 2,
            "start": 0,
            "end": 1024,
            "source_hash": source_hash(text),
        },
    }
    evidence = _matched_payload(payload, "stellar")
    assert evidence["evidence_kind"] == "keyword_excerpt"
    assert "stellar plasma" in evidence["content"]
    assert evidence["embedding_span"] == {"start": 0, "end": 1024}
    span = evidence["evidence_span"]
    assert span["start"] > 1024
    assert evidence["content"] == text[span["start"] : span["end"]]
    assert evidence["char_start"] == 300 + span["start"]
    assert payload["content"] == text


class PassageEmbedder:
    """Deterministic semantic stand-in: the tail has an orthogonal dense signal."""

    @staticmethod
    def split_text(text, *, role="document", context_prefix=""):
        return [
            SimpleNamespace(start=i, end=min(i + 1024, len(text)), text=text[i : i + 1024])
            for i in range(0, len(text), 1024)
        ]

    @staticmethod
    async def embed_all(texts, *, role="document"):
        assert all(len(text) <= 1024 for text in texts)
        return [[0.0, 1.0] if "stellar" in text else [1.0, 0.0] for text in texts]

    async def embed_single_cached(self, text, *, role="query"):
        return (await self.embed_all([text], role=role))[0]


@pytest.mark.usefixtures("embedding_generation")
async def test_dense_only_tail_retrieval_and_document_grouping(document_store, sample_text):
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "coverage",
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    storage = MagicMock()
    storage.get_client = AsyncMock(return_value=client)
    storage._get_client = AsyncMock(return_value=client)
    vocab = MagicMock()
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    vocab.vectorize_query.side_effect = lambda _: SparseVector(indices=[], values=[])
    embedder = PassageEmbedder()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "coverage")
    document = document_store.register(sample_text)
    source = "ordinary orchard material " * 4000 + "stellar evidence at the very end"
    summary, chunks = indexer._split_document(document, source)
    points = await indexer._build_points(document, summary, chunks)
    await client.upsert("coverage", points=points)
    engine = DocumentSearchEngine(storage, embedder, vocab, "coverage")
    engine._searcher = HybridSearcher(storage, dense_weight=1.0, sparse_weight=0.0)
    try:
        for include_chunks in (True, False):
            results = await engine.search("stellar", limit=1, include_chunks=include_chunks)
            assert len(results) == 1
            assert results[0].document_id == document.id
            assert "stellar evidence" in results[0].content
            assert len(results[0].content) <= 1024
        neighbor = document.model_copy(update={"id": uuid4(), "filename": "neighbor.txt"})
        summary, chunks = indexer._split_document(neighbor, "stellar evidence in another document")
        await client.upsert(
            "coverage", points=await indexer._build_points(neighbor, summary, chunks)
        )
        similar = await engine.find_similar(document.id, limit=1)
        assert similar[0].document_id == neighbor.id
        assert similar[0].score == pytest.approx(1.0)
        assert similar[0].content == "stellar evidence in another document"
    finally:
        await client.close()


@pytest.mark.usefixtures("embedding_generation")
async def test_similarity_reads_every_source_page_and_filters_self_before_grouping():
    source_id, target_id = uuid4(), uuid4()
    client = AsyncMock()
    client.scroll.side_effect = [
        ([SimpleNamespace(id=1, vector={"dense": [1.0, 0.0]})], 99),
        ([SimpleNamespace(id=2, vector={"dense": [0.0, 1.0]})], None),
    ]
    tail = SimpleNamespace(
        id=20,
        score=0.99,
        payload={
            "document_id": str(target_id),
            "type": "doc_chunk",
            "content": "tail evidence",
        },
    )
    client.query_points_groups.side_effect = [
        SimpleNamespace(
            groups=[
                SimpleNamespace(
                    id=str(uuid4()),
                    hits=[SimpleNamespace(id=10, score=0.1)],
                )
            ]
        ),
        SimpleNamespace(groups=[SimpleNamespace(id=str(target_id), hits=[tail])]),
    ]
    client.retrieve.return_value = [tail]
    storage = AsyncMock()
    storage.get_client.return_value = client
    engine = DocumentSearchEngine(storage, AsyncMock(), MagicMock(), "coverage")
    results = await engine.find_similar(source_id, limit=1)
    assert results[0].document_id == target_id
    assert client.scroll.await_args_list[1].kwargs["offset"] == 99
    assert [call.kwargs["query"] for call in client.query_points_groups.await_args_list] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    for call in client.query_points_groups.await_args_list:
        assert call.kwargs["group_by"] == "document_id"
        assert call.kwargs["query_filter"].must_not[0].match.value == str(source_id)
        assert call.kwargs["with_payload"] is False
    client.retrieve.assert_awaited_once_with(
        "coverage",
        ids=[20],
        with_payload=True,
        with_vectors=False,
    )


@pytest.mark.usefixtures("embedding_generation")
async def test_chunk_reads_keep_full_canonical_bodies_and_exclude_children():
    doc_id = uuid4()
    raw = "source " * 1000
    canonical = {"document_id": str(doc_id), "type": "doc_chunk", "content": raw}
    child = {
        **canonical,
        "content": "source",
        "embedding_fragment": {
            "schema": 1,
            "index": 1,
            "count": 2,
            "parent_id": 42,
            "start": 7,
            "end": 14,
            "source_hash": source_hash(raw),
        },
    }
    storage = AsyncMock()
    storage.scroll_points.return_value = [canonical, child]
    engine = DocumentSearchEngine(storage, AsyncMock(), MagicMock(), "coverage")
    chunks = await engine.get_document_chunks(doc_id)
    assert [chunk.content for chunk in chunks] == [raw]
    assert storage.scroll_points.await_args.kwargs["max_results"] == 0


@pytest.mark.parametrize("include_chunks", [True, False])
@pytest.mark.usefixtures("embedding_generation")
async def test_document_family_filtered_before_retrieval(include_chunks):
    engine = DocumentSearchEngine(AsyncMock(), AsyncMock(), MagicMock(), "coverage")
    engine._searcher = AsyncMock()
    engine._searcher.search.return_value = []
    await engine.search("evidence", limit=3, include_chunks=include_chunks)
    kwargs = engine._searcher.search.await_args.kwargs
    family = next(condition for condition in kwargs["filter_conditions"] if condition.key == "type")
    assert set(family.match.any) == {"document", "doc_chunk"}
    assert kwargs["group_by"] == (None if include_chunks else "document_id")


async def test_blank_query_rejected_before_initializing_collection(monkeypatch):
    from mcp_docs.tools.search import search_documents

    initialize = AsyncMock()
    monkeypatch.setattr("mcp_docs.tools.search.get_search_engine", initialize)
    result = await search_documents(" \n\t ")
    assert isinstance(result, dict)
    assert "error_code" in result
    initialize.assert_not_awaited()


async def test_keyword_search_does_not_hide_matches_after_many_fragments(monkeypatch):
    from mcp_docs.tools.search import keyword_search

    first, second = str(uuid4()), str(uuid4())
    client = AsyncMock()
    client.scroll.side_effect = [
        ([SimpleNamespace(payload={"document_id": first})] * 512, offset) for offset in range(1, 6)
    ] + [([SimpleNamespace(payload={"document_id": second})], None)]
    engine = AsyncMock()
    engine.storage.get_client.return_value = client
    engine.collection_operation = MagicMock(return_value=AsyncMock())
    monkeypatch.setattr("mcp_docs.tools.search.get_search_engine", AsyncMock(return_value=engine))
    results = await keyword_search("evidence", limit=2)
    assert [result["document_id"] for result in results] == [first, second]
    assert client.scroll.await_count == 6


@pytest.mark.usefixtures("embedding_generation")
async def test_source_layout_repair_once_and_missing_original_preservation(
    document_store,
    sample_text,
    monkeypatch,
):
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(":memory:")
    await storage.create_collection("coverage", dense_dim=2)
    vocab = MagicMock()
    vocab.tokenize.return_value = []
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(document_store, storage, PassageEmbedder(), vocab, "coverage")
    document = document_store.register(sample_text)
    document = document_store.update(document.id, extraction_status=ExtractionStatus.INDEXED)
    source = "# RESTORED HEADING\n\n" + "ordinary source " * 200 + "stellar tail"
    extraction = MagicMock(
        return_value=ExtractedContent(
            text=source,
            title=None,
            page_count=None,
            word_count=len(source.split()),
        )
    )
    monkeypatch.setattr("mcp_docs.indexing.indexer.extract_content", extraction)
    legacy = [
        indexer._create_point("document", document, document.filename, [1.0, 0.0]),
        indexer._create_point(
            "doc_chunk", document, "old body without heading", [1.0, 0.0], chunk_index=0
        ),
    ]
    await storage.upsert_batch("coverage", legacy)
    try:
        # A metadata refresh must not certify that an old body was repaired.
        await indexer.update_document_tags_in_index(document)
        assert await indexer._get_indexed_hashes() == set()
        repaired = await indexer.index_all()
        assert repaired["indexed"] == 1
        assert extraction.call_count == 1
        assert await indexer._get_indexed_hashes() == {indexer._doc_hash(document)}
        assert (await indexer.index_all())["skipped"] is True
        assert extraction.call_count == 1
        records = await storage.scroll_points("coverage", max_results=0)
        assert any(is_derived_fragment(record) for record in records)
        canonical = [record for record in records if not is_derived_fragment(record)]
        assert all(record["source_layout"] == SOURCE_LAYOUT_VERSION for record in canonical)
        assert any("RESTORED HEADING" in record["content"] for record in canonical)

        # Shrinking a source removes obsolete derived children as well as chunks.
        await indexer.index_document(document.id, "short body")
        records = await storage.scroll_points("coverage", max_results=0)
        assert len(records) == 2
        assert not any(is_derived_fragment(record) for record in records)
        assert any(record["content"] == "short body" for record in records)

        # Missing historical originals are reported and their retained points survive.
        await storage.upsert_batch("coverage", legacy)
        sample_text.unlink()
        before, _ = await storage._client.scroll("coverage", with_vectors=True)
        skipped = await indexer.index_all()
        assert skipped["indexed"] == 0
        assert skipped["unavailable_sources"] == [
            {
                "document_id": str(document.id),
                "path": str(sample_text),
                "reason": "Source unavailable; retained index left unchanged",
            }
        ]
        assert extraction.call_count == 1
        assert (await storage._client.scroll("coverage", with_vectors=True))[0] == before
    finally:
        await storage.close()


@pytest.mark.usefixtures("embedding_generation")
async def test_same_content_current_document_cannot_hide_legacy_document(
    document_store,
    sample_text,
    monkeypatch,
):
    from tests.test_vocabulary_accounting import FakeEmbedder, FakeStorage

    original = document_store.register(sample_text)
    # The store may coalesce identical files; clone metadata to model two retained
    # document identities with the same content hash, title and tags.
    legacy = original.model_copy(update={"id": uuid4()})
    original = original.model_copy(update={"extraction_status": ExtractionStatus.INDEXED})
    legacy = legacy.model_copy(update={"extraction_status": ExtractionStatus.INDEXED})
    store = MagicMock()
    store.iter_all.return_value = iter([original, legacy])
    store.count.return_value = 2
    storage = FakeStorage()
    vocab = MagicMock()
    vocab.tokenize.return_value = []
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(store, storage, FakeEmbedder(), vocab, "coverage")
    monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())
    current = indexer._create_point(
        "document",
        original,
        original.filename,
        [1.0, 0.0],
        source_layout=SOURCE_LAYOUT_VERSION,
    )
    old = indexer._create_point("document", legacy, legacy.filename, [1.0, 0.0])
    await storage.upsert_batch("coverage", [current, old])
    extraction = MagicMock(
        return_value=ExtractedContent(
            text="# Restored heading\nbody",
            title=None,
            page_count=None,
            word_count=4,
        )
    )
    monkeypatch.setattr("mcp_docs.indexing.indexer.extract_content", extraction)
    result = await indexer.index_all()
    assert result["indexed"] == 1
    assert extraction.call_count == 1
    store.update.assert_called_once_with(
        legacy.id, title=legacy.title, word_count=4, extraction_status=ExtractionStatus.INDEXED
    )


@pytest.mark.usefixtures("embedding_generation")
async def test_normal_indexing_bounds_serialized_writes_before_pruning(
    document_store,
    sample_text,
    monkeypatch,
):
    from qdrant_client.models import PointsList
    from vector_core.storage import embedding_migration

    from mcp_docs.settings import settings as docs_settings

    monkeypatch.setattr(embedding_migration, "_MAX_UPSERT_BYTES", 10_000)
    monkeypatch.setattr(docs_settings, "max_chunk_chars", 3000)
    monkeypatch.setattr(docs_settings, "chunk_overlap_chars", 0)
    client = AsyncQdrantClient(":memory:")
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    storage._client = client
    await storage.create_collection("coverage", dense_dim=2)
    vocab = MagicMock()
    vocab.tokenize.return_value = []
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(document_store, storage, PassageEmbedder(), vocab, "coverage")
    document = document_store.register(sample_text)
    stale = indexer._create_point("doc_chunk", document, "old tail", [1.0, 0.0], chunk_index=999)
    await storage.upsert_batch("coverage", [stale])
    source = 'é"\\\n' * 10_000 + "stellar tail"
    sizes = []
    operations = []
    actual_upsert, actual_delete = client.upsert, client.delete

    async def bounded_upsert(collection, points, **kwargs):
        size = len(
            PointsList(points=points)
            .model_dump_json(
                by_alias=True,
                exclude_none=True,
                exclude_unset=True,
            )
            .encode()
        )
        assert size <= 10_000
        sizes.append(size)
        operations.append("write")
        return await actual_upsert(collection, points, **kwargs)

    async def recorded_delete(collection, points_selector, **kwargs):
        if stale.id in points_selector.points:
            operations.append("prune old tail")
        return await actual_delete(collection, points_selector=points_selector, **kwargs)

    monkeypatch.setattr(client, "upsert", bounded_upsert)
    monkeypatch.setattr(client, "delete", recorded_delete)
    try:
        count = await indexer.index_document(document.id, source)
        assert count > 30
        assert sum(sizes) > 10_000
        assert operations[-2:] == ["prune old tail", "write"]
        chunks = await DocumentSearchEngine(
            storage, PassageEmbedder(), vocab, "coverage"
        ).get_document_chunks(document.id)
        assert "".join(chunk.content for chunk in chunks) == source
        assert await client.retrieve("coverage", ids=[stale.id]) == []
    finally:
        await storage.close()


@pytest.mark.usefixtures("embedding_generation")
async def test_failed_body_group_keeps_old_tail_and_does_not_publish_layout_key(
    document_store,
    sample_text,
    monkeypatch,
):
    from tests.test_vocabulary_accounting import FakeEmbedder, FakeStorage

    document = document_store.register(sample_text)
    storage = FakeStorage()
    vocab = MagicMock()
    vocab.tokenize.return_value = []
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(document_store, storage, FakeEmbedder(), vocab, "coverage")
    monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())
    old = indexer._create_point(
        "doc_chunk", document, "retained old tail", [1.0, 0.0], chunk_index=999
    )
    await storage.upsert_batch("coverage", [old])
    original_upsert = storage.upsert
    writes = 0

    async def fail_body(collection, points, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("second body write failed")
        return await original_upsert(collection, points, **kwargs)

    monkeypatch.setattr(storage, "upsert", fail_body)
    with pytest.raises(RuntimeError, match="second body write failed"):
        await indexer.index_document(document.id, "source body " * 20_000)
    assert storage.points[old.id].payload["content"] == "retained old tail"
    assert not any(point.payload["type"] == "document" for point in storage.points.values())
