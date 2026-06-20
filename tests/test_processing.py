"""Tests for document processing queue."""

import asyncio
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_docs.models import ExtractionStatus
from mcp_docs.processing import (
    DocumentProcessor,
    ProcessingResult,
    ProcessingStatus,
    ProcessingTask,
)
from mcp_docs.storage.database import DocumentStore, compute_file_hash


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir: Path):
    """Create a temporary DocumentStore."""
    db_path = temp_dir / "test_docs.db"
    store = DocumentStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def sample_file(temp_dir: Path) -> Path:
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Hello, World! This is a test document for processing.")
    return file_path


@pytest.fixture
def sample_markdown(temp_dir: Path) -> Path:
    """Create a sample markdown file."""
    file_path = temp_dir / "sample.md"
    file_path.write_text("# Test Document\n\nThis is the content of the test.")
    return file_path


class TestProcessingTask:
    """Tests for ProcessingTask dataclass."""

    def test_task_ordering_by_priority(self, temp_dir: Path) -> None:
        """Higher priority (lower number) tasks come first."""
        from uuid import uuid4

        task_low = ProcessingTask(
            document_id=uuid4(),
            path=temp_dir / "low.txt",
            priority=10,
        )
        task_high = ProcessingTask(
            document_id=uuid4(),
            path=temp_dir / "high.txt",
            priority=1,
        )

        assert task_high < task_low  # High priority comes first


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    @pytest.mark.asyncio
    async def test_start_and_stop(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Can start and stop processor."""
        processor = DocumentProcessor(store, max_workers=1)

        await processor.start()
        assert processor.is_running

        await processor.stop()
        assert not processor.is_running

    @pytest.mark.asyncio
    async def test_enqueue_and_process(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Can enqueue and process a document."""
        # Register document first
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        processor = DocumentProcessor(store, max_workers=1)
        await processor.start()

        try:
            # Enqueue
            await processor.enqueue(doc.id, sample_file)

            # Wait for processing
            result = await processor.wait_for(doc.id, timeout=10.0)

            assert result is not None
            assert result.status == ProcessingStatus.COMPLETED
            assert result.document_id == doc.id
            assert result.word_count > 0

            # Verify document was updated
            updated = store.read(doc.id)
            assert updated.word_count is not None
            assert updated.word_count > 0

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_process_markdown_extracts_title(
        self, store: DocumentStore, sample_markdown: Path
    ) -> None:
        """Processing markdown extracts title from H1."""
        content_hash = compute_file_hash(sample_markdown)
        doc = store.register(sample_markdown, content_hash)

        processor = DocumentProcessor(store, max_workers=1)
        await processor.start()

        try:
            await processor.enqueue(doc.id, sample_markdown)
            result = await processor.wait_for(doc.id, timeout=10.0)

            assert result is not None
            assert result.status == ProcessingStatus.COMPLETED
            assert result.title == "Test Document"

            # Verify in store
            updated = store.read(doc.id)
            assert updated.title == "Test Document"

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_get_status_queued(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """get_status returns queued status for unprocessed docs."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        # Don't start workers - document stays queued
        processor = DocumentProcessor(store, max_workers=0)

        await processor.queue.put(
            ProcessingTask(document_id=doc.id, path=sample_file)
        )

        status = processor.get_status(doc.id)
        assert status["status"] == ProcessingStatus.QUEUED.value

    @pytest.mark.asyncio
    async def test_get_status_completed(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """get_status returns completed status after processing."""
        content_hash = compute_file_hash(sample_file)
        doc = store.register(sample_file, content_hash)

        processor = DocumentProcessor(store, max_workers=1)
        await processor.start()

        try:
            await processor.enqueue(doc.id, sample_file)
            await processor.wait_for(doc.id, timeout=10.0)

            status = processor.get_status(doc.id)
            assert status["status"] == ProcessingStatus.COMPLETED.value

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_list_queued(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """list_queued shows documents in pipeline."""
        # Create multiple files
        files = []
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)
            doc = store.register(f, compute_file_hash(f))

        processor = DocumentProcessor(store, max_workers=1)
        await processor.start()

        try:
            # Enqueue all
            for f in files:
                doc = store.get_by_hash(compute_file_hash(f))
                await processor.enqueue(doc.id, f)

            # Check queue status
            queued = processor.list_queued()
            # Should have at least queue size info
            assert len(queued) >= 1

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_wait_for_timeout(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """wait_for returns None on timeout."""
        from uuid import uuid4

        processor = DocumentProcessor(store, max_workers=0)  # No workers

        # Wait for non-existent doc
        result = await processor.wait_for(uuid4(), timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_failure(
        self, store: DocumentStore, temp_dir: Path
    ) -> None:
        """Processing failure is handled gracefully."""
        # Register a file that will be deleted
        file_path = temp_dir / "will_delete.txt"
        file_path.write_text("Temporary content")
        content_hash = compute_file_hash(file_path)
        doc = store.register(file_path, content_hash)

        # Delete the file before processing
        file_path.unlink()

        processor = DocumentProcessor(store, max_workers=1)
        await processor.start()

        try:
            await processor.enqueue(doc.id, file_path)
            result = await processor.wait_for(doc.id, timeout=10.0)

            assert result is not None
            assert result.status == ProcessingStatus.FAILED
            assert result.error is not None

        finally:
            await processor.stop()


class TestWaitForDocumentsTerminal:
    """wait_for_documents(require_completed=False) lets directory operations
    proceed once a document is in a terminal state, even if that state is
    FAILED or CANCELLED rather than COMPLETED. A genuinely still-processing
    document still blocks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status", [ExtractionStatus.FAILED, ExtractionStatus.CANCELLED]
    )
    async def test_terminal_non_completed_accepted_when_not_requiring_completion(
        self, store: DocumentStore, sample_file: Path, status: ExtractionStatus
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=status)
        processor = DocumentProcessor(store, max_workers=0)

        # Default (require_completed=True) treats a terminal-but-not-completed
        # document as "not done", which wrongly blocks moves/renames.
        assert await processor.wait_for_documents([doc.id], timeout=1.0) is False
        # require_completed=False accepts any terminal state.
        assert (
            await processor.wait_for_documents(
                [doc.id], timeout=1.0, require_completed=False
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_still_queued_document_times_out_regardless(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        # Registered but never processed (no workers): genuinely still pending.
        doc = store.register(sample_file, compute_file_hash(sample_file))
        processor = DocumentProcessor(store, max_workers=0)

        assert (
            await processor.wait_for_documents(
                [doc.id], timeout=0.1, require_completed=False
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_completed_document_accepted_either_way(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)
        processor = DocumentProcessor(store, max_workers=0)

        assert await processor.wait_for_documents([doc.id], timeout=1.0) is True
        assert (
            await processor.wait_for_documents(
                [doc.id], timeout=1.0, require_completed=False
            )
            is True
        )


class TestCancel:
    """cancel() must only cancel queued documents and record a CANCELLED
    status, never clobbering completed work or marking a cancellation as a
    failure (which startup recovery would re-enqueue)."""

    @pytest.mark.asyncio
    async def test_cancel_queued_records_cancelled_status(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        assert store.read(doc.id).extraction_status == ExtractionStatus.QUEUED

        processor = DocumentProcessor(store, max_workers=0)
        assert processor.cancel(doc.id) is True

        updated = store.read(doc.id)
        assert updated.extraction_status == ExtractionStatus.CANCELLED
        assert updated.extraction_error is None
        assert doc.id in processor.cancelled

    @pytest.mark.asyncio
    async def test_cancel_does_not_clobber_indexed(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """An already-indexed document is never flipped to a terminal
        cancelled/failed state by a stray cancel call."""
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)

        processor = DocumentProcessor(store, max_workers=0)
        assert processor.cancel(doc.id) is False
        assert store.read(doc.id).extraction_status == ExtractionStatus.INDEXED

    @pytest.mark.asyncio
    async def test_cancel_does_not_clobber_extracted(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.EXTRACTED)

        processor = DocumentProcessor(store, max_workers=0)
        assert processor.cancel(doc.id) is False
        assert store.read(doc.id).extraction_status == ExtractionStatus.EXTRACTED

    @pytest.mark.asyncio
    async def test_cancel_refuses_in_progress(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        processor = DocumentProcessor(store, max_workers=0)
        processor.in_progress[doc.id] = ProcessingTask(
            document_id=doc.id, path=sample_file
        )

        assert processor.cancel(doc.id) is False
        # Still queued, not clobbered.
        assert store.read(doc.id).extraction_status == ExtractionStatus.QUEUED

    @pytest.mark.asyncio
    async def test_cancel_missing_document_returns_false(
        self, store: DocumentStore
    ) -> None:
        from uuid import uuid4

        processor = DocumentProcessor(store, max_workers=0)
        assert processor.cancel(uuid4()) is False

    @pytest.mark.asyncio
    async def test_get_status_reports_cancelled(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        processor = DocumentProcessor(store, max_workers=0)
        assert processor.cancel(doc.id) is True

        status = processor.get_status(doc.id)
        assert status["status"] == ProcessingStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_recovery_does_not_reenqueue_cancelled(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Startup recovery must not re-enqueue cancelled documents."""
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.CANCELLED)

        processor = DocumentProcessor(store, max_workers=0)
        await processor._reenqueue_orphaned_documents()

        assert processor.queue.qsize() == 0
        assert store.read(doc.id).extraction_status == ExtractionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_recovery_migrates_legacy_cancellation(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """Documents cancelled by the old version (stored as FAILED with a
        'Processing cancelled' error) are migrated to CANCELLED on recovery and
        not re-enqueued."""
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(
            doc.id,
            extraction_status=ExtractionStatus.FAILED,
            extraction_error="Processing cancelled",
        )

        processor = DocumentProcessor(store, max_workers=0)
        await processor._reenqueue_orphaned_documents()

        assert processor.queue.qsize() == 0
        migrated = store.read(doc.id)
        assert migrated.extraction_status == ExtractionStatus.CANCELLED
        assert migrated.extraction_error is None


class TestWaitForTerminal:
    """wait_for must resolve immediately for a document already in a terminal
    state in the database (e.g. one processed in a previous session, before the
    in-memory completed cache existed), instead of blocking until timeout for a
    worker event that will never fire."""

    @pytest.mark.asyncio
    async def test_returns_completed_for_indexed(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)

        processor = DocumentProcessor(store, max_workers=0)
        result = await processor.wait_for(doc.id, timeout=1.0)

        assert result is not None
        assert result.status == ProcessingStatus.COMPLETED
        # No stale wait event was created.
        assert doc.id not in processor.waiting

    @pytest.mark.asyncio
    async def test_returns_failed_with_error(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(
            doc.id,
            extraction_status=ExtractionStatus.FAILED,
            extraction_error="boom",
        )

        processor = DocumentProcessor(store, max_workers=0)
        result = await processor.wait_for(doc.id, timeout=1.0)

        assert result is not None
        assert result.status == ProcessingStatus.FAILED
        assert result.error == "boom"

    @pytest.mark.asyncio
    async def test_returns_cancelled(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.CANCELLED)

        processor = DocumentProcessor(store, max_workers=0)
        result = await processor.wait_for(doc.id, timeout=1.0)

        assert result is not None
        assert result.status == ProcessingStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_queued_document_still_waits_and_times_out(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """A queued (non-terminal) document is not short-circuited; with no
        worker running it still waits and times out."""
        doc = store.register(sample_file, compute_file_hash(sample_file))  # QUEUED

        processor = DocumentProcessor(store, max_workers=0)
        result = await processor.wait_for(doc.id, timeout=0.3)

        assert result is None

    @pytest.mark.asyncio
    async def test_in_progress_document_is_not_short_circuited(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        """A worker sets EXTRACTED before it finishes auto-indexing, so an
        in-progress document must stay on the event path and not resolve early
        from the DB."""
        doc = store.register(sample_file, compute_file_hash(sample_file))
        store.update(doc.id, extraction_status=ExtractionStatus.EXTRACTED)

        processor = DocumentProcessor(store, max_workers=0)
        processor.in_progress[doc.id] = ProcessingTask(
            document_id=doc.id, path=sample_file
        )

        # No worker will signal (max_workers=0), so it must wait and time out
        # rather than reporting the in-progress EXTRACTED row as completed.
        result = await processor.wait_for(doc.id, timeout=0.3)
        assert result is None


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_to_dict(self) -> None:
        """ProcessingResult.to_dict works correctly."""
        from datetime import UTC, datetime
        from uuid import uuid4

        result = ProcessingResult(
            document_id=uuid4(),
            status=ProcessingStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            title="Test",
            word_count=100,
        )

        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["title"] == "Test"
        assert d["word_count"] == 100


class TestConcurrentWaitFor:
    """wait_for supports concurrent waiters on the same document (they share one
    event). One waiter's timeout must not orphan the others, and a timed-out
    waiter must still recover a result that completed during the wait."""

    @staticmethod
    def _result(doc_id):
        now = datetime.now(UTC)
        return ProcessingResult(
            document_id=doc_id,
            status=ProcessingStatus.COMPLETED,
            started_at=now,
            completed_at=now,
        )

    @pytest.mark.asyncio
    async def test_timeout_recovers_result_completed_during_wait(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        proc = DocumentProcessor(store, max_workers=0)
        proc.in_progress[doc.id] = ProcessingTask(document_id=doc.id, path=sample_file)
        result = self._result(doc.id)

        async def populate():
            await asyncio.sleep(0.05)
            # Result cached but the event is NOT set (e.g. orphaned by a peer).
            proc.completed[doc.id] = result

        asyncio.create_task(populate())
        got = await proc.wait_for(doc.id, timeout=0.3)
        assert got is result

    @pytest.mark.asyncio
    async def test_short_waiter_timeout_does_not_orphan_long_waiter(
        self, store: DocumentStore, sample_file: Path
    ) -> None:
        doc = store.register(sample_file, compute_file_hash(sample_file))
        proc = DocumentProcessor(store, max_workers=0)
        proc.in_progress[doc.id] = ProcessingTask(document_id=doc.id, path=sample_file)
        result = self._result(doc.id)

        async def short():
            return await proc.wait_for(doc.id, timeout=0.1)

        async def long():
            return await proc.wait_for(doc.id, timeout=2.0)

        async def complete():
            await asyncio.sleep(0.3)  # after the short waiter has timed out
            proc.completed[doc.id] = result
            # Mirror the worker's completion signal.
            if doc.id in proc.waiting:
                event, _ = proc.waiting[doc.id]
                event.set()

        t0 = time.monotonic()
        a, b, _ = await asyncio.gather(short(), long(), complete())
        elapsed = time.monotonic() - t0

        assert a is None  # short waiter timed out
        assert b is result  # long waiter was not orphaned and recovered the result
        assert elapsed < 1.5  # long waiter woke on the signal, not after its 2.0s timeout
