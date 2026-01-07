"""Configuration for mcp-docs via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from vector_core.settings import VectorCoreSettingsMixin, settings as vector_settings


class DocsSettings(VectorCoreSettingsMixin, BaseSettings):
    """Docs-specific settings.

    Inherits vector-core settings (embedding_url, qdrant_url, etc.) via mixin.
    """

    model_config = SettingsConfigDict(env_prefix="DOCS_")

    # Extraction settings
    max_file_size_mb: int = 100  # Max file size to process
    ocr_enabled: bool = False  # OCR for scanned PDFs
    ocr_language: str = "eng"  # Tesseract language (future fallback)

    # Vision LLM OCR settings
    ocr_vision_url: str = "http://localhost:22222"  # OpenAI-compatible endpoint
    ocr_vision_model: str = "olmocr-2-7b-1025"  # Vision model for OCR
    ocr_dpi: int = 300  # DPI for PDF rendering (300 is good balance)
    ocr_timeout: int = 180  # Per-page timeout in seconds
    ocr_max_pages: int = 200  # Maximum pages to OCR per document
    ocr_image_max_dimension: int = 1536  # Max image width/height for vision model
    ocr_image_format: str = "jpeg"  # Image format: jpeg (smaller) or png (lossless)
    ocr_jpeg_quality: int = 90  # JPEG quality (1-100) if using jpeg format
    ocr_cache_enabled: bool = True  # Cache OCR results by content hash

    # Chunking thresholds
    max_chunk_chars: int = 80000  # ~20k tokens for Qwen3
    chunk_overlap_chars: int = 500  # Overlap between chunks

    # Processing
    max_workers: int = 2  # Background processing workers
    processing_timeout_seconds: int = 300  # Per-document timeout

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
            "Example: VECTOR_COLLECTION_NAME=unified_knowledge\n"
            "This is required because mcp-docs may have multiple document roots."
        )
