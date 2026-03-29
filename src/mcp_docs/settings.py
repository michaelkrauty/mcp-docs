"""Configuration for mcp-docs via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from vector_core.settings import VectorCoreSettingsMixin
from vector_core.settings import settings as vector_settings


class DocsSettings(VectorCoreSettingsMixin, BaseSettings):
    """Docs-specific settings.

    Inherits vector-core settings (embedding_url, qdrant_url, etc.) via mixin.
    """

    model_config = SettingsConfigDict(env_prefix="DOCS_")

    # OCR settings (for scanned/image-based PDFs)
    # OCR is attempted automatically when markitdown extraction yields insufficient text.
    # Set ocr_vision_url to an OpenAI-compatible vision endpoint to enable OCR.
    # Leave empty to disable OCR (scanned PDFs will return empty/minimal text).
    ocr_vision_url: str = ""  # OpenAI-compatible vision endpoint (e.g. http://localhost:8080)
    ocr_vision_model: str = ""  # Vision model name (empty = let endpoint decide)
    ocr_dpi: int = 300  # DPI for PDF page rendering
    ocr_timeout: int = 180  # Per-page timeout in seconds
    ocr_max_pages: int = 200  # Maximum pages to OCR per document
    ocr_image_max_dimension: int = 1536  # Max image width/height for vision model
    ocr_image_format: str = "jpeg"  # Image format: jpeg (smaller) or png (lossless)
    ocr_jpeg_quality: int = 90  # JPEG quality (1-100) if using jpeg format
    ocr_cache_enabled: bool = True  # Cache OCR results by file metadata
    ocr_concurrency: int = 4  # Max concurrent OCR page requests

    # Chunking thresholds
    max_chunk_chars: int = 80000  # ~20k tokens
    chunk_overlap_chars: int = 500  # Overlap between chunks

    # Processing
    max_workers: int = 2  # Background processing workers

    # Tags
    max_tags_per_document: int = 20
    max_tag_length: int = 50

    # Derived paths (not in vector-core mixin)
    @property
    def docs_db_path(self) -> Path:
        """Path to documents database."""
        return vector_settings.shared_data_dir / "documents.db"

    @property
    def facts_db_path(self) -> Path:
        """Path to facts database (shared with mcp-notes)."""
        return vector_settings.shared_data_dir / "facts.db"


settings = DocsSettings()


def validate_collection_name() -> None:
    """
    Validate that VECTOR_COLLECTION_NAME is set.

    Unlike mcp-notes (which can generate from ~/notes path),
    mcp-docs has no single path and requires explicit collection name.

    Raises:
        RuntimeError: If collection name not set
    """
    if not settings.collection_name:
        raise RuntimeError(
            "VECTOR_COLLECTION_NAME environment variable must be set for mcp-docs. "
            "Example: VECTOR_COLLECTION_NAME=my_documents\n"
            "This is required because mcp-docs may have multiple document roots."
        )
