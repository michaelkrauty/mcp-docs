"""Indexing must not stall the MCP event loop (issue #83).

Extraction (markitdown, Office, OCR) is synchronous and can run for minutes on
a scanned PDF. Every test here blocks a fake extractor on a threading.Event
that only the event loop can release, so the test fails if extraction runs on
the loop thread instead of a worker thread.
"""

import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from vector_core.errors import ErrorCode, is_error_response

from mcp_docs.extraction import ocr as ocr_mod
from mcp_docs.models import ExtractedContent
from mcp_docs.settings import settings
from mcp_docs.storage.database import compute_file_hash
from mcp_docs.tools.indexing import index_document
from tests.test_source_repair_safety import prepared

pytestmark = pytest.mark.usefixtures("embedding_generation")


def _blocking_extract(started: asyncio.Event, release: threading.Event):
    """A fake extractor that signals the loop and waits for it to answer."""
    loop = asyncio.get_running_loop()

    def extract(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2), "Event loop could not release the extraction worker"
        return ExtractedContent(text="extracted body", title=None, page_count=None, word_count=2)

    return extract


async def test_index_all_extracts_off_the_event_loop(document_store, sample_text, monkeypatch):
    indexer, _, _, _ = prepared(document_store, sample_text, monkeypatch)
    started, release = asyncio.Event(), threading.Event()
    monkeypatch.setattr(
        "mcp_docs.indexing.indexer.extract_content", _blocking_extract(started, release)
    )

    task = asyncio.create_task(indexer.index_all())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not task.done()
    finally:
        release.set()
    result = await task

    assert result["indexed"] == 1
    assert result["errors"] is None


@contextmanager
def tool_dependencies(document_store, indexer, extract):
    with (
        patch("mcp_docs.tools.indexing.get_document_store", return_value=document_store),
        patch(
            "mcp_docs.tools.indexing.get_document_indexer",
            new=AsyncMock(return_value=indexer),
        ),
        patch("mcp_docs.extraction.extractor.extract_content", new=extract),
    ):
        yield


async def run_index_document_tool(document_store, indexer, document, extract):
    with tool_dependencies(document_store, indexer, extract):
        return await index_document(str(document.id))


async def test_index_document_tool_extracts_off_the_event_loop(
    document_store, sample_text, monkeypatch
):
    indexer, document, storage, old = prepared(document_store, sample_text, monkeypatch)
    started, release = asyncio.Event(), threading.Event()

    task = asyncio.create_task(
        run_index_document_tool(
            document_store, indexer, document, _blocking_extract(started, release)
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not task.done()
    finally:
        release.set()
    result = await task

    assert result["success"] is True
    assert result["points_indexed"] > 0
    assert old.id not in storage.points
    assert any(
        "extracted body" in str(point.payload.get("content", ""))
        for point in storage.points.values()
    )


async def test_index_document_tool_bounds_concurrent_extractions(
    document_store, sample_text, monkeypatch
):
    indexer, document, _, _ = prepared(document_store, sample_text, monkeypatch)
    limit = settings.max_workers
    loop = asyncio.get_running_loop()
    counter = threading.Lock()
    running = peak = 0
    limit_reached, release = asyncio.Event(), threading.Event()

    def extract(*args):
        nonlocal running, peak
        with counter:
            running += 1
            peak = max(peak, running)
            if running == limit:
                loop.call_soon_threadsafe(limit_reached.set)
        try:
            assert release.wait(2), "Event loop could not release the extraction workers"
        finally:
            with counter:
                running -= 1
        return ExtractedContent(text="extracted body", title=None, page_count=None, word_count=2)

    with tool_dependencies(document_store, indexer, extract):
        tasks = [asyncio.create_task(index_document(str(document.id))) for _ in range(limit + 1)]
        try:
            await asyncio.wait_for(limit_reached.wait(), 2)
            # Give an unbounded pool time to start the extra extraction.
            await asyncio.sleep(0.2)
            assert peak == limit
        finally:
            release.set()
        results = await asyncio.gather(*tasks)

    assert all(result["success"] for result in results)


def _changes_during_extraction(change):
    def extract(*args):
        change()
        return ExtractedContent(text="extracted body", title=None, page_count=None, word_count=2)

    return extract


@pytest.mark.parametrize("change", ["registration", "source_file", "deleted"])
async def test_index_document_tool_does_not_publish_text_from_a_changed_source(
    change, document_store, sample_text, monkeypatch
):
    indexer, document, storage, _ = prepared(document_store, sample_text, monkeypatch)
    before = {point_id: point.model_copy(deep=True) for point_id, point in storage.points.items()}

    def mutate():
        if change == "deleted":
            # A concurrent delete_document removed the registration.
            document_store.delete(document.id)
            return
        sample_text.write_text("edited while it was being extracted")
        if change == "registration":
            # A concurrent rescan registered the edited file, so the file and
            # the registration agree again, but not with the extracted text.
            document_store.update(document.id, content_hash=compute_file_hash(sample_text))

    result = await run_index_document_tool(
        document_store, indexer, document, _changes_during_extraction(mutate)
    )

    assert is_error_response(result)
    assert result["error_code"] == ErrorCode.CONFLICT.value
    assert storage.points == before


async def test_ocr_sync_wrapper_runs_on_a_worker_thread(monkeypatch):
    content = ExtractedContent(text="ocr text", title=None, page_count=1, word_count=2)
    loop_thread = threading.get_ident()
    ocr_threads = []

    async def extract(path):
        ocr_threads.append(threading.get_ident())
        return content

    monkeypatch.setattr(ocr_mod, "_extract_with_cache", extract)

    result = await asyncio.to_thread(ocr_mod.extract_scanned_pdf_sync, Path("scan.pdf"))

    assert result is content
    assert ocr_threads and ocr_threads[0] != loop_thread


async def test_ocr_sync_wrapper_keeps_ocr_errors_inside_a_running_loop(monkeypatch):
    async def extract(path):
        raise RuntimeError("vision backend failed")

    monkeypatch.setattr(ocr_mod, "_extract_with_cache", extract)

    with pytest.raises(RuntimeError, match="vision backend failed"):
        ocr_mod.extract_scanned_pdf_sync(Path("scan.pdf"))
