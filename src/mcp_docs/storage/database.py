"""SQLite-based document storage with thread-safe operations."""

import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from vector_core.utils.hashing import compute_file_hash
from vector_core.utils.sentinel import UNSET, UnsetType, is_set
from vector_core.utils.sqlite import SQLiteConfig, ThreadSafeSQLiteStore

from mcp_docs.models import (
    Document,
    DocumentNotFoundError,
    DocumentRoot,
    DocumentStatus,
    DocumentSummary,
    DocumentType,
    ExtractionStatus,
)
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


class DocumentStore(ThreadSafeSQLiteStore):
    """
    SQLite-based document storage with thread-safe operations.

    Uses vector-core's shared_data_dir for storage location.

    Thread-safety:
    - Uses per-thread connections (via ThreadSafeSQLiteStore)
    - WAL mode for concurrent readers + single writer
    - 5-second timeout for lock acquisition

    Tables:
    - documents: Core document metadata
    - document_tags: Tags (one-to-many with documents)
    - document_roots: Scanned root directories
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize document store.

        Args:
            db_path: Path to SQLite database. Default: shared_data_dir/documents.db
        """
        db_path = db_path or settings.docs_db_path
        super().__init__(
            db_path,
            config=SQLiteConfig(
                foreign_keys=True,  # Enable FK constraints for cascading deletes
                busy_timeout_ms=5000,
                connect_timeout=5.0,
            ),
        )
        self._ensure_parent_dir()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()

        # Core documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content_hash TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                title TEXT,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                document_root TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                extraction_status TEXT NOT NULL DEFAULT 'queued',
                extraction_error TEXT,
                indexed_at TEXT NOT NULL,
                last_verified TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Document tags (one-to-many with documents)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                PRIMARY KEY (document_id, tag)
            )
        """)

        # Document roots for scanning
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_roots (
                path TEXT PRIMARY KEY,
                name TEXT,
                recursive INTEGER DEFAULT 1,
                enabled INTEGER DEFAULT 1,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_scanned TEXT,
                file_count INTEGER DEFAULT 0
            )
        """)

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash "
            "ON documents(content_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status "
            "ON documents(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_extraction "
            "ON documents(extraction_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_type "
            "ON documents(doc_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_root "
            "ON documents(document_root)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags_document "
            "ON document_tags(document_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags_tag "
            "ON document_tags(tag)"
        )

        conn.commit()

    def register(
        self,
        path: Path,
        content_hash: str | None = None,
        document_root: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        """
        Register a document from file path.

        If document with same content hash exists, returns existing document.

        Args:
            path: Path to document file
            content_hash: Pre-computed SHA-256 hash (computed if not provided)
            document_root: Root directory this document belongs to
            tags: Initial tags

        Returns:
            Document (existing if duplicate, new otherwise)

        Raises:
            FileNotFoundError: If file doesn't exist
            DuplicateDocumentError: If document already registered (with existing doc attached)
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Compute content hash if not provided
        if content_hash is None:
            content_hash = compute_file_hash(path)

        # Use INSERT OR IGNORE to handle concurrent registration atomically
        # (content_hash has UNIQUE constraint)
        conn = self._get_conn()
        now = datetime.now(UTC)
        doc_id = uuid4()
        doc_type = DocumentType.from_extension(path.suffix)
        size_bytes = path.stat().st_size

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO documents
            (id, content_hash, path, filename, doc_type, size_bytes,
             document_root, status, extraction_status, indexed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(doc_id),
                content_hash,
                str(path),
                path.name,
                doc_type.value,
                size_bytes,
                document_root,
                DocumentStatus.ACTIVE.value,
                ExtractionStatus.QUEUED.value,
                now.isoformat(),
                now.isoformat(),
            ),
        )

        # Check if insert succeeded or was ignored due to duplicate
        if cursor.rowcount == 0:
            # Duplicate - return existing document (update path if moved)
            conn.commit()  # Release any locks
            existing = self.get_by_hash(content_hash)
            if existing and existing.path != str(path):
                self._update_path(existing.id, str(path))
                existing = self.read(existing.id)
            return existing

        # New document - add tags
        if tags:
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO document_tags (document_id, tag) VALUES (?, ?)",
                    (str(doc_id), tag.lower().strip()),
                )

        conn.commit()
        return self.read(doc_id)

    def read(self, document_id: UUID) -> Document | None:
        """
        Read a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, content_hash, path, filename, doc_type, title,
                   size_bytes, page_count, word_count, document_root,
                   status, extraction_status, extraction_error,
                   indexed_at, last_verified, created_at
            FROM documents WHERE id = ?
            """,
            (str(document_id),),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_document(row, conn)

    def get_by_hash(self, content_hash: str) -> Document | None:
        """
        Get document by content hash.

        Args:
            content_hash: SHA-256 hash

        Returns:
            Document if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id FROM documents WHERE content_hash = ?",
            (content_hash,),
        )
        row = cursor.fetchone()
        if row:
            return self.read(UUID(row[0]))
        return None

    def update(
        self,
        document_id: UUID,
        title: str | None = None,
        page_count: int | None = None,
        word_count: int | None = None,
        extraction_status: ExtractionStatus | None = None,
        extraction_error: str | None | UnsetType = UNSET,
        status: DocumentStatus | None = None,
        path: str | None = None,
        filename: str | None = None,
        content_hash: str | None = None,
        document_root: str | None = None,
    ) -> Document:
        """
        Update document metadata.

        Args:
            document_id: Document UUID
            title: New title
            page_count: New page count
            word_count: New word count
            extraction_status: New extraction status
            extraction_error: Extraction error message (pass None to clear
                a stale error, e.g. when re-queueing or after a successful
                re-extraction; leave UNSET to keep the current value)
            status: New document status
            path: New path (for relocations)
            filename: New basename (when a relocation also renames the file)
            content_hash: New content hash (when file modified)
            document_root: New document root

        Returns:
            Updated Document

        Raises:
            DocumentNotFoundError: If not found
        """
        # Verify exists
        if self.read(document_id) is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        conn = self._get_conn()
        updates = []
        params: list[Any] = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)

        if word_count is not None:
            updates.append("word_count = ?")
            params.append(word_count)

        if extraction_status is not None:
            updates.append("extraction_status = ?")
            params.append(extraction_status.value)

        if is_set(extraction_error):
            updates.append("extraction_error = ?")
            params.append(extraction_error)

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)

        if path is not None:
            updates.append("path = ?")
            params.append(path)

        if filename is not None:
            updates.append("filename = ?")
            params.append(filename)

        if content_hash is not None:
            # Check for hash collision before updating — another document
            # may already have this hash (e.g., duplicate file content).
            existing_with_hash = self.get_by_hash(content_hash)
            if existing_with_hash and existing_with_hash.id != document_id:
                # Another document already has this content hash.
                # This means the file was modified to match an existing document.
                # Skip the hash update to avoid UNIQUE constraint violation.
                logger.debug(
                    f"Skipping content_hash update for {document_id}: "
                    f"hash {content_hash[:16]}... already belongs to "
                    f"{existing_with_hash.id}"
                )
            else:
                updates.append("content_hash = ?")
                params.append(content_hash)

        if document_root is not None:
            updates.append("document_root = ?")
            params.append(document_root)

        if updates:
            updates.append("indexed_at = ?")
            params.append(datetime.now(UTC).isoformat())
            params.append(str(document_id))

            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        return self.read(document_id)

    def update_tags(
        self,
        document_id: UUID,
        tags: list[str],
    ) -> Document:
        """
        Replace document tags.

        Args:
            document_id: Document UUID
            tags: New tag list

        Returns:
            Updated Document

        Raises:
            DocumentNotFoundError: If not found
        """
        # Verify exists
        if self.read(document_id) is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        conn = self._get_conn()

        # Delete existing tags
        conn.execute(
            "DELETE FROM document_tags WHERE document_id = ?",
            (str(document_id),),
        )

        # Add new tags
        for tag in tags:
            normalized = tag.lower().strip()
            if normalized:
                conn.execute(
                    "INSERT OR IGNORE INTO document_tags (document_id, tag) VALUES (?, ?)",
                    (str(document_id), normalized),
                )

        conn.commit()
        return self.read(document_id)

    def delete(self, document_id: UUID) -> None:
        """
        Delete a document.

        Args:
            document_id: Document UUID

        Raises:
            DocumentNotFoundError: If not found
        """
        # Verify exists
        if self.read(document_id) is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        conn = self._get_conn()
        conn.execute("DELETE FROM documents WHERE id = ?", (str(document_id),))
        conn.commit()

    def query(
        self,
        doc_type: DocumentType | None = None,
        status: DocumentStatus | None = None,
        extraction_status: ExtractionStatus | None = None,
        document_root: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Document]:
        """
        Query documents by criteria.

        Args:
            doc_type: Filter by document type
            status: Filter by status
            extraction_status: Filter by extraction status
            document_root: Filter by root directory
            tags: Filter by tags (all must match)
            limit: Maximum results

        Returns:
            List of matching Documents
        """
        conn = self._get_conn()

        query = "SELECT id FROM documents WHERE 1=1"
        params: list[Any] = []

        if doc_type:
            query += " AND doc_type = ?"
            params.append(doc_type.value)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if extraction_status:
            query += " AND extraction_status = ?"
            params.append(extraction_status.value)

        if document_root:
            query += " AND document_root = ?"
            params.append(document_root)

        if tags:
            # All tags must match
            for tag in tags:
                query += """
                    AND id IN (
                        SELECT document_id FROM document_tags WHERE tag = ?
                    )
                """
                params.append(tag.lower().strip())

        query += " ORDER BY indexed_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [self.read(UUID(row[0])) for row in cursor.fetchall()]

    def list_summaries(
        self,
        doc_type: str | DocumentType | None = None,
        status: DocumentStatus | None = None,
        extraction_status: ExtractionStatus | None = None,
        document_root: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[DocumentSummary]:
        """
        List documents as lightweight summaries.

        Args:
            doc_type: Filter by document type (string or enum)
            status: Filter by status
            extraction_status: Filter by extraction status
            document_root: Filter by root directory
            tags: Filter by tags (all must match)
            limit: Maximum results

        Returns:
            List of DocumentSummary objects
        """
        conn = self._get_conn()

        query = """
            SELECT id, path, content_hash, filename, doc_type, title,
                   size_bytes, status, extraction_status, indexed_at
            FROM documents WHERE 1=1
        """
        params: list[Any] = []

        if doc_type:
            doc_type_value = doc_type.value if isinstance(doc_type, DocumentType) else doc_type
            query += " AND doc_type = ?"
            params.append(doc_type_value)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if extraction_status:
            query += " AND extraction_status = ?"
            params.append(extraction_status.value)

        if document_root:
            query += " AND document_root = ?"
            params.append(document_root)

        if tags:
            for tag in tags:
                query += """
                    AND id IN (
                        SELECT document_id FROM document_tags WHERE tag = ?
                    )
                """
                params.append(tag.lower().strip())

        query += " ORDER BY indexed_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        summaries = []

        for row in cursor.fetchall():
            doc_id = UUID(row[0])
            tags = self._get_tags(conn, doc_id)

            summaries.append(
                DocumentSummary(
                    id=doc_id,
                    path=row[1],
                    content_hash=row[2],
                    filename=row[3],
                    doc_type=DocumentType(row[4]),
                    title=row[5],
                    size_bytes=row[6],
                    tags=tags,
                    status=DocumentStatus(row[7]),
                    extraction_status=ExtractionStatus(row[8]),
                    indexed_at=datetime.fromisoformat(row[9]),
                )
            )

        return summaries

    def _row_to_document(self, row: tuple, conn: sqlite3.Connection) -> Document:
        """Convert database row to Document."""
        doc_id = UUID(row[0])
        tags = self._get_tags(conn, doc_id)

        return Document(
            id=doc_id,
            content_hash=row[1],
            path=row[2],
            filename=row[3],
            doc_type=DocumentType(row[4]),
            title=row[5],
            size_bytes=row[6],
            page_count=row[7],
            word_count=row[8],
            document_root=row[9],
            status=DocumentStatus(row[10]),
            extraction_status=ExtractionStatus(row[11]),
            extraction_error=row[12],
            indexed_at=datetime.fromisoformat(row[13]),
            last_verified=datetime.fromisoformat(row[14]) if row[14] else None,
            created_at=datetime.fromisoformat(row[15]),
            tags=tags,
        )

    def _get_tags(self, conn: sqlite3.Connection, document_id: UUID) -> list[str]:
        """Get tags for a document."""
        cursor = conn.execute(
            "SELECT tag FROM document_tags WHERE document_id = ? ORDER BY tag",
            (str(document_id),),
        )
        return [row[0] for row in cursor.fetchall()]

    def _update_path(self, document_id: UUID, new_path: str) -> None:
        """Update document path and filename (for relocations).

        The filename always follows the path's basename, so they are updated
        together to keep the registry consistent when the same content is
        re-registered at a renamed location.
        """
        conn = self._get_conn()
        conn.execute(
            "UPDATE documents SET path = ?, filename = ?, indexed_at = ? WHERE id = ?",
            (
                new_path,
                Path(new_path).name,
                datetime.now(UTC).isoformat(),
                str(document_id),
            ),
        )
        conn.commit()

    # Document roots management

    def add_root(
        self,
        path: str,
        name: str | None = None,
        recursive: bool = True,
        enabled: bool = True,
    ) -> DocumentRoot:
        """
        Add a document root directory.

        Args:
            path: Directory path
            name: Optional friendly name
            recursive: Whether to scan subdirectories
            enabled: Whether scanning is enabled

        Returns:
            DocumentRoot
        """
        conn = self._get_conn()
        now = datetime.now(UTC)

        conn.execute(
            """
            INSERT OR REPLACE INTO document_roots (path, name, recursive, enabled, added_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (path, name, 1 if recursive else 0, 1 if enabled else 0, now.isoformat()),
        )
        conn.commit()

        return DocumentRoot(
            path=path,
            added_at=now,
            last_scanned=None,
            file_count=0,
            name=name,
            recursive=recursive,
            enabled=enabled,
        )

    def get_root(self, path: str) -> DocumentRoot | None:
        """
        Get a document root by path.

        Args:
            path: Directory path

        Returns:
            DocumentRoot if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT path, name, recursive, enabled, added_at, last_scanned, file_count
               FROM document_roots WHERE path = ?""",
            (path,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return DocumentRoot(
            path=row[0],
            name=row[1],
            recursive=bool(row[2]),
            enabled=bool(row[3]),
            added_at=datetime.fromisoformat(row[4]),
            last_scanned=datetime.fromisoformat(row[5]) if row[5] else None,
            file_count=row[6] or 0,
        )

    def list_roots(self) -> list[DocumentRoot]:
        """List all document roots."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT path, name, recursive, enabled, added_at, last_scanned, file_count
               FROM document_roots ORDER BY path"""
        )

        return [
            DocumentRoot(
                path=row[0],
                name=row[1],
                recursive=bool(row[2]),
                enabled=bool(row[3]),
                added_at=datetime.fromisoformat(row[4]),
                last_scanned=datetime.fromisoformat(row[5]) if row[5] else None,
                file_count=row[6] or 0,
            )
            for row in cursor.fetchall()
        ]

    def update_root_scan(self, path: str, file_count: int) -> None:
        """
        Update root after scanning.

        Args:
            path: Root path
            file_count: Number of files found
        """
        conn = self._get_conn()
        conn.execute(
            "UPDATE document_roots SET last_scanned = ?, file_count = ? WHERE path = ?",
            (datetime.now(UTC).isoformat(), file_count, path),
        )
        conn.commit()

    def remove_root(self, path: str) -> None:
        """
        Remove a document root.

        Args:
            path: Directory path
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM document_roots WHERE path = ?", (path,))
        conn.commit()

    def count(self) -> int:
        """Get total document count."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def iter_all(
        self, extraction_status: ExtractionStatus | None = None
    ) -> Iterator[Document]:
        """
        Iterate over every document, optionally filtered by extraction status.

        Unlike ``query()`` and ``list_summaries()`` (which default to a 50-row
        limit), this yields the entire matching corpus, so callers that must
        process all documents (index_all in particular) are never silently
        capped.

        The id list is snapshotted up front and each document is read lazily.
        A document that cannot be loaded (deleted by another writer between the
        snapshot and its read, or a malformed stored row) is skipped rather
        than aborting iteration over the rest. A failure of the initial id
        query itself is not swallowed: it propagates, so a systemic read
        failure such as a locked database stays loud instead of masquerading
        as an empty corpus.

        Args:
            extraction_status: If given, only yield documents in this state.

        Yields:
            Document for each readable document.
        """
        conn = self._get_conn()
        if extraction_status is not None:
            cursor = conn.execute(
                "SELECT id FROM documents WHERE extraction_status = ? "
                "ORDER BY indexed_at DESC",
                (extraction_status.value,),
            )
        else:
            cursor = conn.execute("SELECT id FROM documents ORDER BY indexed_at DESC")

        for row in cursor.fetchall():
            try:
                document = self.read(UUID(row[0]))
            except sqlite3.Error:
                # Systemic DB failure (e.g. database is locked). Must NOT be
                # swallowed as a single bad row: that would yield a partial
                # corpus and let a force reindex delete points for everything
                # that failed to load. Fail loud.
                raise
            except (ValueError, KeyError, TypeError):
                # Malformed stored row (bad uuid/enum/date). Skip just this
                # document rather than aborting iteration over the rest.
                logger.warning(
                    "Skipping unreadable document %s during iter_all",
                    row[0],
                    exc_info=True,
                )
                continue
            if document is None:
                # Deleted between the id snapshot and this read; skip it.
                logger.debug("Document %s vanished during iter_all, skipping", row[0])
                continue
            yield document

    def get_by_path(self, path: str) -> Document | None:
        """
        Get document by exact file path.

        Args:
            path: Exact file path

        Returns:
            Document if found, None otherwise
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT id FROM documents WHERE path = ?", (path,))
        row = cursor.fetchone()
        if row:
            return self.read(UUID(row[0]))
        return None

    def query_by_path_prefix(self, path_prefix: str, limit: int = 10000) -> list[DocumentSummary]:
        """
        Query documents strictly under a directory (for directory ops).

        Only documents under the directory (path_prefix + "/") are returned.
        The match is an exact, case-sensitive prefix comparison anchored at
        the path-separator boundary - LIKE is unsuitable here because "%"/"_"
        are wildcards AND SQLite LIKE is case-insensitive for ASCII, any of
        which lets a query for "/data/my_docs/" wrongly return documents under
        siblings like "/data/myXdocs/" (the "_" matched "X") or "/data/My_Docs/"
        (case). This mirrors update_paths_batch / update_document_roots_batch.

        Args:
            path_prefix: Directory path (with or without trailing "/")
            limit: Maximum results

        Returns:
            List of DocumentSummary objects
        """
        conn = self._get_conn()
        old_dir = path_prefix.rstrip("/")
        cursor = conn.execute(
            """
            SELECT id, path, content_hash, filename, doc_type, title,
                   size_bytes, status, extraction_status, indexed_at
            FROM documents WHERE SUBSTR(path, 1, ?) = ? ORDER BY path LIMIT ?
            """,
            (len(old_dir) + 1, old_dir + "/", limit),
        )

        summaries = []
        for row in cursor.fetchall():
            doc_id = UUID(row[0])
            tags = self._get_tags(conn, doc_id)

            summaries.append(
                DocumentSummary(
                    id=doc_id,
                    path=row[1],
                    content_hash=row[2],
                    filename=row[3],
                    doc_type=DocumentType(row[4]),
                    title=row[5],
                    size_bytes=row[6],
                    tags=tags,
                    status=DocumentStatus(row[7]),
                    extraction_status=ExtractionStatus(row[8]),
                    indexed_at=datetime.fromisoformat(row[9]),
                )
            )

        return summaries

    def update_paths_batch(self, old_prefix: str, new_prefix: str) -> int:
        """
        Batch update document paths by replacing a directory prefix.

        Only documents strictly under the directory (old_prefix + "/") are
        updated. The match is an exact, case-sensitive prefix comparison
        anchored at the path-separator boundary — LIKE is unsuitable here
        because "%"/"_" are wildcards AND SQLite LIKE is case-insensitive
        for ASCII, any of which lets renaming "/data/docs" silently
        corrupt siblings like "/data/docs2", "/data/my_dir"-vs-"myxdir",
        or "/data/Docs".

        Args:
            old_prefix: Old directory path (with or without trailing "/")
            new_prefix: New directory path

        Returns:
            Number of documents updated
        """
        conn = self._get_conn()
        old_dir = old_prefix.rstrip("/")
        new_dir = new_prefix.rstrip("/")

        cursor = conn.execute(
            """
            UPDATE documents
            SET path = ? || SUBSTR(path, ? + 1), indexed_at = ?
            WHERE SUBSTR(path, 1, ?) = ?
            """,
            (
                new_dir,
                len(old_dir),
                datetime.now(UTC).isoformat(),
                len(old_dir) + 1,
                old_dir + "/",
            ),
        )

        count = cursor.rowcount
        conn.commit()
        return count

    def update_document_roots_batch(self, old_prefix: str, new_root: str) -> int:
        """
        Batch update document_root for documents under a directory.

        Matching is the same anchored, case-sensitive prefix comparison as
        update_paths_batch.

        Args:
            old_prefix: Directory path to match (with or without trailing "/")
            new_root: New document root

        Returns:
            Number of documents updated
        """
        conn = self._get_conn()
        old_dir = old_prefix.rstrip("/")

        cursor = conn.execute(
            """
            UPDATE documents
            SET document_root = ?, indexed_at = ?
            WHERE SUBSTR(path, 1, ?) = ?
            """,
            (
                new_root,
                datetime.now(UTC).isoformat(),
                len(old_dir) + 1,
                old_dir + "/",
            ),
        )

        count = cursor.rowcount
        conn.commit()
        return count
