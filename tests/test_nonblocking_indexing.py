"""Indexing must not stall the MCP event loop (issue #83).

Extraction (markitdown, Office, OCR) is synchronous and can run for minutes on
a scanned PDF. Every test here blocks a fake extractor on a threading.Event
that only the event loop can release, so the test fails if extraction runs on
the loop thread instead of a worker thread.
"""

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_docs.extraction import ocr as ocr_mod
from mcp_docs.models import ExtractedContent, ExtractionStatus
from mcp_docs.processing import DocumentProcessor
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


async def test_index_document_tool_extracts_off_the_event_loop(document_store, sample_text):
    document = document_store.register(sample_text)
    document_store.update(document.id, extraction_status=ExtractionStatus.EXTRACTED)
    indexer = MagicMock()
    indexer.index_document = AsyncMock(return_value=3)
    started, release = asyncio.Event(), threading.Event()

    with (
        patch("mcp_docs.tools.indexing.get_document_store", return_value=document_store),
        patch(
            "mcp_docs.tools.indexing.get_document_indexer",
            new=AsyncMock(return_value=indexer),
        ),
        patch(
            "mcp_docs.extraction.extractor.extract_content",
            new=_blocking_extract(started, release),
        ),
    ):
        task = asyncio.create_task(index_document(str(document.id)))
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert not task.done()
        finally:
            release.set()
        result = await task

    assert result["success"] is True
    assert result["points_indexed"] == 3
    indexer.index_document.assert_awaited_once_with(document.id, "extracted body")


def _pending_startup_index(document_store, sample_text):
    """A processor whose startup index_all blocks until the test releases it."""
    document = document_store.register(sample_text)
    document_store.update(document.id, extraction_status=ExtractionStatus.EXTRACTED)
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled = []

    async def index_all(force=False):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return {"indexed": 1}

    indexer = MagicMock()
    indexer.index_all = AsyncMock(side_effect=index_all)
    processor = DocumentProcessor(document_store, max_workers=1, indexer=indexer)
    return processor, indexer, entered, release, cancelled


async def test_start_returns_while_startup_indexing_runs(document_store, sample_text):
    processor, indexer, entered, release, _ = _pending_startup_index(document_store, sample_text)

    await asyncio.wait_for(processor.start(), 2)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        indexer.index_all.assert_awaited_once_with(force=False)
        release.set()
        assert await asyncio.wait_for(processor._startup_indexing, 2) is None
    finally:
        await processor.stop()


async def test_stop_cancels_unfinished_startup_indexing(document_store, sample_text):
    processor, _, entered, _, cancelled = _pending_startup_index(document_store, sample_text)

    await asyncio.wait_for(processor.start(), 2)
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(processor.stop(), 2)

    assert cancelled == [True]
    assert processor._startup_indexing is None


async def test_startup_indexing_failure_is_logged_not_raised(document_store, sample_text, caplog):
    processor, indexer, _, _, _ = _pending_startup_index(document_store, sample_text)
    indexer.index_all = AsyncMock(side_effect=RuntimeError("index unavailable"))

    await asyncio.wait_for(processor.start(), 2)
    try:
        assert await asyncio.wait_for(processor._startup_indexing, 2) is None
    finally:
        await processor.stop()

    assert "Failed to index extracted documents: index unavailable" in caplog.text


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
