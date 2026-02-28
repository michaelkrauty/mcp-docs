"""Data models for mcp-docs."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    RTF = "rtf"
    ODT = "odt"
    XLSX = "xlsx"
    CSV = "csv"
    EPUB = "epub"
    XML = "xml"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "DocumentType":
        """Get DocumentType from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "doc": cls.DOC,
            "pptx": cls.PPTX,
            "ppt": cls.PPT,
            "txt": cls.TXT,
            "md": cls.MD,
            "markdown": cls.MD,
            "html": cls.HTML,
            "htm": cls.HTML,
            "rtf": cls.RTF,
            "odt": cls.ODT,
            "xlsx": cls.XLSX,
            "xls": cls.XLSX,
            "csv": cls.CSV,
            "epub": cls.EPUB,
            "xml": cls.XML,
        }
        return mapping.get(ext, cls.UNKNOWN)


class DocumentStatus(str, Enum):
    """Document status relative to filesystem."""

    ACTIVE = "active"  # Path exists AND hash matches
    MODIFIED = "modified"  # Path exists BUT hash differs
    RELOCATED = "relocated"  # Found at different path (same hash)
    DELETED = "deleted"  # Not found anywhere


class ExtractionStatus(str, Enum):
    """Document extraction/processing status."""

    QUEUED = "queued"  # Waiting for processing
    PROCESSING = "processing"  # Currently being processed
    EXTRACTED = "extracted"  # Successfully extracted
    INDEXED = "indexed"  # Extracted and indexed
    SKIPPED_NO_OCR = "skipped_no_ocr"  # Scanned PDF, OCR disabled
    SKIPPED_TOO_LARGE = "skipped_too_large"  # File too large
    FAILED = "failed"  # Extraction failed


class DocumentError(Exception):
    """Base exception for document operations."""

    pass


class DocumentNotFoundError(DocumentError):
    """Document not found by ID or hash."""

    pass


class DuplicateDocumentError(DocumentError):
    """Document with same content hash already exists."""

    pass


class ExtractionError(DocumentError):
    """Document content extraction failed."""

    pass


class Document(BaseModel):
    """
    Registered document with metadata.

    Primary identifier is content_hash (SHA-256), not path.
    Path is a hint that may change (document relocated).
    """

    model_config = {"use_enum_values": True}

    id: UUID
    content_hash: str  # SHA-256 of file content
    path: str  # Current/last known path
    filename: str
    doc_type: DocumentType
    title: str | None
    size_bytes: int
    page_count: int | None
    word_count: int | None
    tags: list[str]
    document_root: str | None  # Root path this was scanned from
    status: DocumentStatus
    extraction_status: ExtractionStatus
    extraction_error: str | None
    indexed_at: datetime
    last_verified: datetime | None
    created_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (backward compatibility)."""
        return self.model_dump(mode="json")


class DocumentSummary(BaseModel):
    """Lightweight document summary for listings."""

    model_config = {"use_enum_values": True}

    id: UUID
    path: str
    content_hash: str
    filename: str
    doc_type: DocumentType
    title: str | None
    size_bytes: int
    tags: list[str]
    status: DocumentStatus
    extraction_status: ExtractionStatus
    indexed_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (backward compatibility)."""
        return self.model_dump(mode="json")


class DocumentChunk(BaseModel):
    """A chunk of extracted document content for indexing."""

    document_id: UUID
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    char_start: int
    char_end: int


class DocumentRoot(BaseModel):
    """A root directory for document scanning."""

    path: str
    added_at: datetime
    last_scanned: datetime | None
    file_count: int
    name: str | None = None
    recursive: bool = True
    enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (backward compatibility)."""
        return self.model_dump(mode="json")


class ExtractedContent(BaseModel):
    """Result of document content extraction."""

    text: str
    title: str | None
    page_count: int | None
    word_count: int
    metadata: dict = Field(default_factory=dict)
