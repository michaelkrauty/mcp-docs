"""Background directory scanner for document roots."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from uuid import UUID
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mcp_docs.extraction import ContentExtractor
from mcp_docs.models import DocumentRoot, DocumentStatus
from mcp_docs.storage.database import DocumentStore, compute_file_hash

logger = logging.getLogger(__name__)


# Supported document extensions (whitelist for security)
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".rtf",
    ".html",
    ".htm",
    ".xlsx",
    ".xls",
    ".csv",
    ".epub",
    ".xml",
}

# Maximum number of errors to collect (prevents memory exhaustion)
MAX_ERRORS = 1000

# Maximum number of files to scan per root (prevents DoS)
MAX_FILES_PER_ROOT = 100000


@dataclass
class ScanResult:
    """Result of scanning a document root."""

    root_path: str
    scanned_at: datetime
    files_found: int = 0
    files_new: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_relocated: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "root_path": self.root_path,
            "scanned_at": self.scanned_at.isoformat(),
            "files_found": self.files_found,
            "files_new": self.files_new,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "files_relocated": self.files_relocated,
            "files_skipped": self.files_skipped,
            "errors": self.errors,
        }


class DocumentScanner:
    """
    Scans document roots for new and modified files.

    Tracks file changes via content hash and registers/updates documents
    accordingly.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        extractor: ContentExtractor | None = None,
        recursive: bool = True,
    ):
        """
        Initialize scanner.

        Args:
            document_store: DocumentStore for registration
            extractor: ContentExtractor for file type detection (created if not provided)
            recursive: Whether to scan subdirectories
        """
        self.document_store = document_store
        self.extractor = extractor or ContentExtractor()
        self.recursive = recursive
        self._running = False
        self._task: asyncio.Task | None = None

    def is_supported(self, path: Path) -> bool:
        """Check if file extension is supported."""
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _is_safe_path(self, file_path: Path, root_path: Path) -> bool:
        """
        Check if file path is safe (within root, not a symlink escape).

        Prevents path traversal attacks via symlinks or directory escapes.
        """
        try:
            # Resolve to absolute path (follows symlinks)
            resolved = file_path.resolve()
            root_resolved = root_path.resolve()

            # Check if resolved path is within root
            # Use os.path.commonpath for robust comparison
            try:
                common = Path(os.path.commonpath([resolved, root_resolved]))
                return common == root_resolved
            except ValueError:
                # Different drives on Windows, definitely not within root
                return False
        except (OSError, RuntimeError):
            # Permission denied, too many symlinks, etc.
            return False

    async def scan_root(
        self,
        root: DocumentRoot,
        enqueue_callback: Callable | None = None,
        delete_callback: Callable[[UUID], Awaitable[None]] | None = None,
        relocate_callback: Callable[[UUID, str, str], Awaitable[None]] | None = None,
    ) -> ScanResult:
        """
        Scan a document root for changes.

        Args:
            root: DocumentRoot to scan
            enqueue_callback: Optional async callback to enqueue new/modified docs
            delete_callback: Optional async callback when doc is marked deleted (for index cleanup)
            relocate_callback: Optional async callback when doc is moved (doc_id, old_path, new_path)

        Returns:
            ScanResult with statistics
        """
        result = ScanResult(
            root_path=root.path,
            scanned_at=datetime.now(UTC),
        )

        root_path = Path(root.path)
        if not root_path.exists():
            result.errors.append(f"Root path does not exist: {root.path}")
            return result

        if not root_path.is_dir():
            result.errors.append(f"Root path is not a directory: {root.path}")
            return result

        # Get existing documents in this root
        existing_docs = {
            doc.path: doc
            for doc in self.document_store.list_summaries(
                document_root=root.path, limit=10000
            )
        }

        # Track which paths we've seen
        seen_paths: set[str] = set()

        # Scan directory
        try:
            if self.recursive:
                files = root_path.rglob("*")
            else:
                files = root_path.glob("*")

            files_processed = 0
            for file_path in files:
                # Enforce file limit to prevent DoS
                if files_processed >= MAX_FILES_PER_ROOT:
                    if len(result.errors) < MAX_ERRORS:
                        result.errors.append(
                            f"File limit reached ({MAX_FILES_PER_ROOT}), stopping scan"
                        )
                    break

                # Skip directories
                if not file_path.is_file():
                    continue

                # Skip symlinks (security: prevent symlink traversal attacks)
                if file_path.is_symlink():
                    result.files_skipped += 1
                    continue

                # Skip unsupported files
                if not self.is_supported(file_path):
                    result.files_skipped += 1
                    continue

                # Skip hidden files
                if file_path.name.startswith("."):
                    result.files_skipped += 1
                    continue

                # Skip Office temp/lock files (e.g., ~$document.docx)
                if file_path.name.startswith("~$"):
                    result.files_skipped += 1
                    continue

                # Security: validate path is within root (prevents path traversal)
                if not self._is_safe_path(file_path, root_path):
                    result.files_skipped += 1
                    if len(result.errors) < MAX_ERRORS:
                        result.errors.append(
                            f"Skipped unsafe path: {file_path}"
                        )
                    continue

                files_processed += 1
                result.files_found += 1
                path_str = str(file_path.resolve())
                seen_paths.add(path_str)

                try:
                    content_hash = compute_file_hash(file_path)

                    if path_str in existing_docs:
                        # Check if modified
                        existing = existing_docs[path_str]
                        existing_full = self.document_store.read(existing.id)

                        if existing_full and existing_full.content_hash != content_hash:
                            # File modified
                            self.document_store.update(
                                existing.id,
                                content_hash=content_hash,
                                status=DocumentStatus.MODIFIED,
                            )
                            result.files_modified += 1

                            if enqueue_callback:
                                await enqueue_callback(existing.id, file_path)
                        elif existing_full and existing_full.status != DocumentStatus.ACTIVE:
                            # File unchanged but was marked deleted/etc - restore to active
                            self.document_store.update(
                                existing.id,
                                status=DocumentStatus.ACTIVE,
                            )
                    else:
                        # New file - check if hash exists elsewhere
                        by_hash = self.document_store.get_by_hash(content_hash)
                        if by_hash:
                            # Same content, different location - file was moved
                            old_path = by_hash.path
                            self.document_store.update(
                                by_hash.id,
                                path=path_str,
                                document_root=root.path,
                                status=DocumentStatus.ACTIVE,
                            )
                            result.files_relocated += 1

                            # Mark old path as "seen" to prevent deletion
                            seen_paths.add(old_path)

                            # Update vector index with new path
                            if relocate_callback:
                                try:
                                    await relocate_callback(by_hash.id, old_path, path_str)
                                except Exception as e:
                                    if len(result.errors) < MAX_ERRORS:
                                        result.errors.append(
                                            f"Failed to update index for relocated {path_str}: {e}"
                                        )
                        else:
                            # Truly new file
                            doc = self.document_store.register(
                                path=file_path,
                                content_hash=content_hash,
                                document_root=root.path,
                            )
                            result.files_new += 1

                            if enqueue_callback:
                                await enqueue_callback(doc.id, file_path)

                except Exception as e:
                    if len(result.errors) < MAX_ERRORS:
                        result.errors.append(f"Error processing {file_path}: {e}")

        except Exception as e:
            if len(result.errors) < MAX_ERRORS:
                result.errors.append(f"Error scanning directory: {e}")

        # Mark deleted files and clean up index
        for path_str, doc in existing_docs.items():
            if path_str not in seen_paths:
                self.document_store.update(
                    doc.id,
                    status=DocumentStatus.DELETED,
                )
                result.files_deleted += 1

                # Clean up vector index for deleted document
                if delete_callback:
                    try:
                        await delete_callback(doc.id)
                    except Exception as e:
                        if len(result.errors) < MAX_ERRORS:
                            result.errors.append(f"Failed to delete index for {path_str}: {e}")

        # Update last scan time
        self.document_store.update_root_scan(root.path, result.files_found)

        return result

    async def scan_all_roots(
        self,
        enqueue_callback: Callable | None = None,
        delete_callback: Callable[[UUID], Awaitable[None]] | None = None,
        relocate_callback: Callable[[UUID, str, str], Awaitable[None]] | None = None,
    ) -> list[ScanResult]:
        """
        Scan all registered document roots.

        Args:
            enqueue_callback: Optional async callback to enqueue new/modified docs
            delete_callback: Optional async callback when doc is marked deleted (for index cleanup)
            relocate_callback: Optional async callback when doc is moved (doc_id, old_path, new_path)

        Returns:
            List of ScanResults for each root
        """
        roots = self.document_store.list_roots()
        results = []

        for root in roots:
            if not root.enabled:
                continue

            try:
                result = await self.scan_root(
                    root, enqueue_callback, delete_callback, relocate_callback
                )
                results.append(result)
            except Exception as e:
                results.append(
                    ScanResult(
                        root_path=root.path,
                        scanned_at=datetime.now(UTC),
                        errors=[str(e)],
                    )
                )

        return results

    async def start_background_scanning(
        self,
        interval_seconds: int = 300,  # 5 minutes
        enqueue_callback: Callable | None = None,
    ) -> None:
        """
        Start background scanning loop.

        Args:
            interval_seconds: Time between scans
            enqueue_callback: Optional async callback to enqueue new/modified docs
        """
        if self._running:
            logger.warning("Background scanning already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._scan_loop(interval_seconds, enqueue_callback)
        )
        logger.info(f"Started background scanning (interval: {interval_seconds}s)")

    async def stop_background_scanning(self, timeout: float = 10.0) -> None:
        """
        Stop background scanning.

        Args:
            timeout: Maximum time to wait for current scan to complete
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._task = None

        logger.info("Stopped background scanning")

    async def _scan_loop(
        self,
        interval_seconds: int,
        enqueue_callback: Callable | None,
    ) -> None:
        """Background scanning loop."""
        while self._running:
            try:
                results = await self.scan_all_roots(enqueue_callback)
                total_new = sum(r.files_new for r in results)
                total_modified = sum(r.files_modified for r in results)

                if total_new > 0 or total_modified > 0:
                    logger.info(
                        f"Background scan: {total_new} new, {total_modified} modified"
                    )

            except Exception as e:
                logger.error(f"Background scan error: {e}")

            # Wait for next scan
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

    @property
    def is_running(self) -> bool:
        """Check if background scanning is running."""
        return self._running
