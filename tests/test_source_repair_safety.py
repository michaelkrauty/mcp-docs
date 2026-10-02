"""Failed cleanup and changed files must never certify a completed layout repair."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import MatchAny
from vector_core.embeddings.sparse import SparseVector

from mcp_docs.indexing.indexer import SOURCE_LAYOUT_VERSION, DocumentIndexer
from mcp_docs.models import ExtractedContent, ExtractionStatus
from tests.test_vocabulary_accounting import FakeEmbedder, FakeStorage

pytestmark = pytest.mark.usefixtures("embedding_generation")


async def test_registered_source_hash_does_not_block_event_loop(
    document_store, sample_text, monkeypatch
):
    document = document_store.register(sample_text)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def slow_hash(path):
        assert path == sample_text
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2), "Event loop could not release the hash worker"
        return document.content_hash

    monkeypatch.setattr("mcp_docs.indexing.indexer.compute_file_hash", slow_hash)
    verification = asyncio.create_task(DocumentIndexer._verify_registered_source(document))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not verification.done()
    finally:
        release.set()
        await verification


def prepared(document_store, sample_text, monkeypatch, layout=None):
    document = document_store.register(sample_text)
    document = document_store.update(document.id, extraction_status=ExtractionStatus.INDEXED)
    storage = FakeStorage()
    vocab = MagicMock()
    vocab.tokenize.return_value = []
    vocab.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(document_store, storage, FakeEmbedder(), vocab, "repair")
    monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())
    summary = indexer._create_point(
        "document",
        document,
        document.filename,
        [1.0, 0.0],
        source_layout=layout,
    )
    old = indexer._create_point(
        "doc_chunk",
        document,
        "retained historical evidence",
        [1.0, 0.0],
        chunk_index=999,
    )
    storage.points.update({point.id: point for point in (summary, old)})
    return indexer, document, storage, old


@pytest.mark.parametrize("failure", ["scroll", "delete"])
@pytest.mark.parametrize("layout", [None, SOURCE_LAYOUT_VERSION])
async def test_cleanup_failure_invalidates_layout_and_incremental_retry_repairs(
    failure,
    layout,
    document_store,
    sample_text,
    monkeypatch,
):
    indexer, document, storage, old = prepared(document_store, sample_text, monkeypatch, layout)
    original_scroll, original_delete = storage.scroll, storage.delete
    failed = False

    async def scroll(collection, scroll_filter=None, **kwargs):
        nonlocal failed
        is_document_prune = any(
            condition.key == "type" and isinstance(condition.match, MatchAny)
            for condition in (scroll_filter.must if scroll_filter else [])
        )
        if failure == "scroll" and is_document_prune and not failed:
            failed = True
            raise RuntimeError("stale scroll unavailable")
        return await original_scroll(collection, scroll_filter=scroll_filter, **kwargs)

    async def delete(collection, points_selector, **kwargs):
        nonlocal failed
        if failure == "delete" and old.id in points_selector.points and not failed:
            failed = True
            raise RuntimeError("stale delete unavailable")
        await original_delete(collection, points_selector, **kwargs)

    monkeypatch.setattr(storage, "scroll", scroll)
    monkeypatch.setattr(storage, "delete", delete)
    with pytest.raises(RuntimeError, match=f"stale {failure} unavailable"):
        await indexer.index_document(document.id, "replacement evidence")
    assert failed
    assert old.id in storage.points
    assert await indexer._get_indexed_hashes() == set()

    extraction = MagicMock(
        return_value=ExtractedContent(
            text="replacement evidence",
            title=None,
            page_count=None,
            word_count=2,
        )
    )
    monkeypatch.setattr("mcp_docs.indexing.indexer.extract_content", extraction)
    result = await indexer.index_all()
    assert result["indexed"] == 1
    assert extraction.call_count == 1
    assert old.id not in storage.points
    assert await indexer._get_indexed_hashes() == {indexer._doc_hash(document)}
    assert (await indexer.index_all())["skipped"] is True
    assert extraction.call_count == 1


@pytest.mark.parametrize("change_at", ["before", "during_extraction", "during_embedding"])
async def test_changed_registered_source_is_not_published_under_old_hash(
    change_at,
    document_store,
    sample_text,
    monkeypatch,
):
    indexer, document, storage, _ = prepared(document_store, sample_text, monkeypatch)
    before = {point_id: point.model_copy(deep=True) for point_id, point in storage.points.items()}
    replacement = "changed source at the same registered path"
    if change_at == "before":
        sample_text.write_text(replacement)

    def extract(*args):
        if change_at == "during_extraction":
            sample_text.write_text(replacement)
        return ExtractedContent(text="extracted source", title=None, page_count=None, word_count=2)

    extraction = MagicMock(side_effect=extract)
    monkeypatch.setattr("mcp_docs.indexing.indexer.extract_content", extraction)
    build = indexer._build_points

    async def build_points(*args):
        points = await build(*args)
        if change_at == "during_embedding":
            sample_text.write_text(replacement)
        return points

    monkeypatch.setattr(indexer, "_build_points", build_points)
    result = await indexer.index_all()
    assert result["indexed"] == 0
    assert any(
        "Source hash differs from registration; rescan" in error for error in result["errors"]
    )
    assert storage.points == before
    assert document_store.read(document.id).content_hash == document.content_hash
    assert extraction.call_count == (0 if change_at == "before" else 1)
