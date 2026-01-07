"""Document processing queue and processor."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import UUID

from mcp_docs.extraction import ContentExtractor
from mcp_docs.models import ExtractionStatus
from mcp_docs.settings import settings
from mcp_docs.storage.database import DocumentStore

if TYPE_CHECKING:
    from mcp_docs.indexing import DocumentIndexer

logger = logging.getLogger(__name__)

# Maximum number of completed results to keep in memory
# Prevents unbounded memory growth during long server lifetime
DEFAULT_MAX_COMPLETED_CACHE = 1000

# Maximum number of documents that can be queued for processing
# Prevents unbounded memory growth if documents are queued faster than processed
DEFAULT_MAX_QUEUE_SIZE = 10000


class ProcessingStatus(str, Enum):
    """Processing task status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """A document processing task."""

    document_id: UUID
    path: Path
    priority: int = 0
    queued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0

    def __lt__(self, other: "ProcessingTask") -> bool:
        """Higher priority (lower number) comes first, then earlier queue time."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.queued_at < other.queued_at


@dataclass
class ProcessingResult:
    """Result of document processing."""

    document_id: UUID
    status: ProcessingStatus
    started_at: datetime
    completed_at: datetime
    title: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "document_id": str(self.document_id),
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "title": self.title,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "error": self.error,
        }


class DocumentProcessor:
    """
    Background processor for document extraction and indexing.

    Queue behavior:
    - Queue is ephemeral (in-memory during server lifetime)
    - On restart, documents with extraction_status=QUEUED remain in DB
    - Re-scan of document roots will re-enqueue pending documents

    Usage:
        processor = DocumentProcessor(document_store)
        await processor.start()  # Start worker tasks

        # Enqueue document
        await processor.enqueue(document_id, path)

        # Wait for completion (optional)
        result = await processor.wait_for(document_id, timeout=60)

        # Shutdown
        await processor.stop()
    """

    def __init__(
        self,
        document_store: DocumentStore,
        max_workers: int | None = None,
        extractor: ContentExtractor | None = None,
        on_complete: Callable[[ProcessingResult], None] | None = None,
        max_completed_cache: int = DEFAULT_MAX_COMPLETED_CACHE,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        indexer: DocumentIndexer | None = None,
    ):
        """
        Initialize processor.

        Args:
            document_store: DocumentStore for updating document status
            max_workers: Number of concurrent workers (default from settings)
            extractor: ContentExtractor instance (created if not provided)
            on_complete: Optional callback when processing completes
            max_completed_cache: Max completed results to keep in memory (LRU eviction)
            max_queue_size: Max documents in queue (prevents OOM during bulk imports)
            indexer: Optional DocumentIndexer for automatic indexing after extraction
        """
        self.document_store = document_store
        self.max_workers = max_workers or settings.max_workers
        self.extractor = extractor or ContentExtractor()
        self.on_complete = on_complete
        self.max_completed_cache = max_completed_cache
        self.max_queue_size = max_queue_size
        self.indexer = indexer

        # Priority queue with bounded size to prevent unbounded memory growth
        self.queue: asyncio.PriorityQueue[ProcessingTask] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        # Track state
        self.in_progress: dict[UUID, ProcessingTask] = {}
        self.completed: dict[UUID, ProcessingResult] = {}  # LRU: oldest first
        # Waiting events with creation time for stale cleanup: (event, created_at)
        self.waiting: dict[UUID, tuple[asyncio.Event, datetime]] = {}
        self._waiting_timeout_seconds = 3600  # 1 hour max wait before cleanup
        self.cancelled: set[UUID] = set()  # Track cancelled document IDs

        # Lock for wait_for() event creation to prevent race conditions
        # where concurrent wait_for() calls could create duplicate events
        self._wait_lock = asyncio.Lock()

        # Worker management
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Thread pool for blocking extraction I/O
        # Using a shared pool prevents thread exhaustion under load
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="doc-extract-",
        )

    async def start(self) -> None:
        """Start background worker tasks and re-enqueue orphaned documents."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)

        logger.info(f"Started {self.max_workers} document processing workers")

        # Re-enqueue orphaned documents from previous runs
        await self._reenqueue_orphaned_documents()

        # Index any extracted-but-not-indexed documents
        # Awaited (not background) to ensure indexing completes on startup
        if self.indexer is not None:
            await self._index_extracted_documents()

    async def _reenqueue_orphaned_documents(self) -> None:
        """
        Re-enqueue documents left in QUEUED or PROCESSING state from a previous run.

        This handles two cases where the server restarted:
        1. Documents waiting in queue (QUEUED) - never started processing
        2. Documents mid-processing (PROCESSING) - worker was interrupted

        Note: We don't check if files exist here - let the extraction worker handle
        that. Path.exists() can fail spuriously at startup due to filesystem timing.
        """
        try:
            # Query for documents with extraction_status=QUEUED or PROCESSING
            orphaned = []
            for status in (ExtractionStatus.QUEUED, ExtractionStatus.PROCESSING):
                docs = self.document_store.query(
                    extraction_status=status,
                    limit=self.max_queue_size,
                )
                orphaned.extend(docs)

            if not orphaned:
                return

            logger.info(f"Re-enqueuing {len(orphaned)} orphaned documents from previous run")

            for doc in orphaned:
                path = Path(doc.path)

                # Reset to QUEUED if it was mid-processing
                self.document_store.update(
                    doc.id,
                    extraction_status=ExtractionStatus.QUEUED,
                )

                task = ProcessingTask(
                    document_id=doc.id,
                    path=path,
                    priority=1,  # Lower priority than new documents
                )
                try:
                    self.queue.put_nowait(task)
                    logger.debug(f"Re-enqueued orphaned document {doc.id}")
                except asyncio.QueueFull:
                    logger.warning(f"Queue full, stopping re-enqueue at {doc.id}")
                    break

        except Exception as e:
            logger.error(f"Failed to re-enqueue orphaned documents: {e}")

    async def _index_extracted_documents(self) -> None:
        """
        Index documents that were extracted but not indexed.

        This handles the case where documents were extracted before
        auto-indexing was enabled, or if indexing failed previously.
        Uses index_all() for proper two-pass vocabulary training.
        Runs in background on startup so it doesn't block initialization.
        """
        if self.indexer is None:
            return

        try:
            # Check if there are any EXTRACTED (not INDEXED) documents
            docs = self.document_store.query(
                extraction_status=ExtractionStatus.EXTRACTED,
                limit=1,
            )

            if not docs:
                return

            logger.info("Found extracted documents pending indexing, running index_all")
            result = await self.indexer.index_all(force=False)
            logger.info(f"Background index_all completed: {result}")

        except Exception as e:
            logger.error(f"Failed to index extracted documents: {e}")

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop all workers gracefully.

        Logs any in-progress or queued documents that will be abandoned.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        # Log abandoned work for debugging/recovery
        if self.in_progress:
            for doc_id in self.in_progress:
                logger.warning(f"Abandoning in-progress document: {doc_id}")

        queue_size = self.queue.qsize()
        if queue_size > 0:
            logger.warning(f"{queue_size} documents still queued, will be abandoned")

        # Cancel workers after timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Worker shutdown timed out, cancelling")
            for worker in self._workers:
                worker.cancel()

        self._workers.clear()

        # Shutdown thread pool
        self._executor.shutdown(wait=False)
        logger.info("Document processing workers stopped")

    async def enqueue(
        self,
        document_id: UUID,
        path: Path,
        priority: int = 0,
        timeout: float = 30.0,
    ) -> bool:
        """
        Add document to processing queue.

        Args:
            document_id: Document UUID
            path: Path to document file
            priority: Lower number = higher priority (default 0)
            timeout: Max time to wait if queue is full (seconds)

        Returns:
            True if enqueued, False if queue is full after timeout

        Raises:
            asyncio.TimeoutError: If timeout expires while waiting for queue space
        """
        task = ProcessingTask(
            document_id=document_id,
            path=path,
            priority=priority,
        )

        # Update database status
        self.document_store.update(
            document_id,
            extraction_status=ExtractionStatus.QUEUED,
        )

        try:
            # Wait up to timeout for queue space
            await asyncio.wait_for(self.queue.put(task), timeout=timeout)
            logger.debug(f"Enqueued document {document_id} for processing")
            return True
        except asyncio.TimeoutError:
            # Queue full - update status to reflect this
            logger.warning(
                f"Queue full ({self.queue.qsize()}/{self.max_queue_size}), "
                f"failed to enqueue {document_id}"
            )
            self.document_store.update(
                document_id,
                extraction_status=ExtractionStatus.FAILED,
                extraction_error="Processing queue full - try again later",
            )
            return False

    async def wait_for(
        self,
        document_id: UUID,
        timeout: float = 300.0,
    ) -> ProcessingResult | None:
        """
        Wait for document processing to complete.

        Uses a lock to prevent race conditions when multiple callers
        wait for the same document concurrently.

        Args:
            document_id: Document UUID
            timeout: Maximum time to wait in seconds

        Returns:
            ProcessingResult if completed, None if timeout
        """
        # Use lock for atomic check-and-create of wait event
        async with self._wait_lock:
            # Already completed?
            if document_id in self.completed:
                return self.completed[document_id]

            # Create wait event if needed (atomic with completed check)
            if document_id not in self.waiting:
                self.waiting[document_id] = (asyncio.Event(), datetime.now(UTC))

            event, _ = self.waiting[document_id]

        # Wait outside the lock to allow concurrent waits on the same event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self.completed.get(document_id)
        except asyncio.TimeoutError:
            # Clean up stale waiting event on timeout to prevent unbounded growth
            self.waiting.pop(document_id, None)
            logger.warning(f"Timeout waiting for document {document_id}")
            return None

    async def wait_for_documents(self, document_ids: list[UUID], timeout: float = 60.0) -> bool:
        """
        Wait for multiple documents to finish processing. Returns True if all done.

        Args:
            document_ids: List of document UUIDs to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if all documents are completed, False if timeout or any document failed
        """
        if not document_ids:
            return True

        # Wait for all documents concurrently
        tasks = [
            self.wait_for(doc_id, timeout=timeout)
            for doc_id in document_ids
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if all documents completed successfully
            all_completed = True
            for i, result in enumerate(results):
                doc_id = document_ids[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error waiting for document {doc_id}: {result}")
                    all_completed = False
                elif result is None:
                    logger.warning(f"Timeout waiting for document {doc_id}")
                    all_completed = False
                elif result.status != ProcessingStatus.COMPLETED:
                    logger.warning(f"Document {doc_id} processing failed: {result.error}")
                    all_completed = False

            return all_completed

        except Exception as e:
            logger.error(f"Error waiting for documents: {e}")
            return False

    def get_status(self, document_id: UUID) -> dict:
        """
        Get processing status for a document.

        Args:
            document_id: Document UUID

        Returns:
            Status dict with position in queue, state, etc.
        """
        # Check completed
        if document_id in self.completed:
            result = self.completed[document_id]
            return {
                "document_id": str(document_id),
                "status": result.status.value,
                "completed_at": result.completed_at.isoformat(),
                "error": result.error,
            }

        # Check in-progress
        if document_id in self.in_progress:
            task = self.in_progress[document_id]
            return {
                "document_id": str(document_id),
                "status": ProcessingStatus.PROCESSING.value,
                "started_at": datetime.now(UTC).isoformat(),
                "retry_count": task.retry_count,
            }

        # Must be queued (can't easily check position in asyncio.PriorityQueue)
        return {
            "document_id": str(document_id),
            "status": ProcessingStatus.QUEUED.value,
            "queue_size": self.queue.qsize(),
        }

    def cancel(self, document_id: UUID) -> bool:
        """
        Cancel queued processing (cannot cancel in-progress).

        Args:
            document_id: Document UUID

        Returns:
            True if cancelled, False if not found or in-progress
        """
        # Can't cancel in-progress tasks
        if document_id in self.in_progress:
            return False

        # Track as cancelled so worker skips it when dequeued
        self.cancelled.add(document_id)

        # Mark as cancelled in DB
        try:
            self.document_store.update(
                document_id,
                extraction_status=ExtractionStatus.FAILED,
                extraction_error="Processing cancelled",
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel document {document_id}: {e}")
            # Still keep in cancelled set to skip processing
            return False

    def list_queued(self) -> list[dict]:
        """
        List documents currently in processing pipeline.

        Returns:
            List of status dicts for queued and in-progress documents
        """
        result = []

        # In-progress
        for doc_id, task in self.in_progress.items():
            result.append({
                "document_id": str(doc_id),
                "status": ProcessingStatus.PROCESSING.value,
                "path": str(task.path),
                "queued_at": task.queued_at.isoformat(),
                "priority": task.priority,
            })

        # Note: asyncio.PriorityQueue doesn't support iteration
        # We report queue size instead
        result.append({
            "_queue_size": self.queue.qsize(),
            "_workers_active": len(self.in_progress),
        })

        return result

    def _cache_result(self, doc_id: UUID, result: ProcessingResult) -> None:
        """
        Cache a processing result with LRU eviction.

        Evicts oldest entries when cache exceeds max size.
        Dict iteration order is insertion order (Python 3.7+).
        """
        self.completed[doc_id] = result

        # Evict oldest entries if over limit
        while len(self.completed) > self.max_completed_cache:
            oldest_id = next(iter(self.completed))
            del self.completed[oldest_id]
            # Also clean up any stale waiting events
            self.waiting.pop(oldest_id, None)

        # Periodically clean up stale waiting events (every 100 cache writes)
        if len(self.completed) % 100 == 0:
            self._cleanup_stale_waiting()

    def _cleanup_stale_waiting(self) -> None:
        """
        Remove stale waiting events that have exceeded the timeout.

        This prevents unbounded memory growth from abandoned wait_for() calls
        where the document was never processed (cancelled, worker crashed, etc.).
        """
        now = datetime.now(UTC)
        stale = [
            doc_id
            for doc_id, (_, created_at) in self.waiting.items()
            if (now - created_at).total_seconds() > self._waiting_timeout_seconds
        ]
        for doc_id in stale:
            self.waiting.pop(doc_id, None)
            logger.debug(f"Cleaned up stale waiting event for document {doc_id}")

    async def _worker(self, worker_id: int) -> None:
        """Background worker loop."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Wait for task with timeout to check shutdown
                try:
                    task = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Skip cancelled tasks
                if task.document_id in self.cancelled:
                    self.cancelled.discard(task.document_id)
                    self.queue.task_done()
                    logger.debug(f"Skipped cancelled task {task.document_id}")
                    continue

                # Process the document
                self.in_progress[task.document_id] = task
                try:
                    result = await self._process(task)
                    self._cache_result(task.document_id, result)

                    # Notify waiters and clean up event to prevent memory leak
                    if task.document_id in self.waiting:
                        event, _ = self.waiting[task.document_id]
                        event.set()
                        # Clean up the event after signaling to prevent unbounded growth
                        del self.waiting[task.document_id]

                    # Callback
                    if self.on_complete:
                        try:
                            self.on_complete(result)
                        except Exception as e:
                            logger.warning(f"on_complete callback error: {e}")

                finally:
                    del self.in_progress[task.document_id]
                    # Clean up cancelled set in case cancel was requested during processing
                    self.cancelled.discard(task.document_id)
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    async def _process(self, task: ProcessingTask) -> ProcessingResult:
        """
        Process a single document.

        Args:
            task: Processing task

        Returns:
            ProcessingResult
        """
        started_at = datetime.now(UTC)
        logger.info(f"Processing document {task.document_id}: {task.path}")

        # Update status to processing
        self.document_store.update(
            task.document_id,
            extraction_status=ExtractionStatus.PROCESSING,
        )

        try:
            # Run extraction in shared thread pool (blocking I/O)
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                self._executor,
                self.extractor.extract,
                task.path,
            )

            # Update document with extracted content
            self.document_store.update(
                task.document_id,
                title=content.title,
                page_count=content.page_count,
                word_count=content.word_count,
                extraction_status=ExtractionStatus.EXTRACTED,
            )

            # Automatically index if indexer is configured
            if self.indexer is not None:
                try:
                    points = await self.indexer.index_document(task.document_id, content.text)
                    logger.debug(f"Indexed document {task.document_id}: {points} points")
                except Exception as e:
                    # Log but don't fail - document is extracted, indexing can be retried
                    logger.warning(f"Auto-indexing failed for {task.document_id}: {e}")

            completed_at = datetime.now(UTC)
            logger.info(
                f"Completed processing {task.document_id} in "
                f"{(completed_at - started_at).total_seconds():.2f}s"
            )

            return ProcessingResult(
                document_id=task.document_id,
                status=ProcessingStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                title=content.title,
                page_count=content.page_count,
                word_count=content.word_count,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Processing failed for {task.document_id}: {error_msg}")

            # Update document with error
            self.document_store.update(
                task.document_id,
                extraction_status=ExtractionStatus.FAILED,
                extraction_error=error_msg,
            )

            return ProcessingResult(
                document_id=task.document_id,
                status=ProcessingStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                error=error_msg,
            )

    @property
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
