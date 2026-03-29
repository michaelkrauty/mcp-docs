"""PDF content extraction using markitdown for text and pypdf for metadata."""

import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.models import ExtractedContent, ExtractionError

logger = logging.getLogger(__name__)


def extract_pdf(path: Path) -> ExtractedContent:
    """
    Extract text content from a PDF file.

    Uses markitdown (pdfplumber + pdfminer) for text extraction and
    pypdf for metadata. This handles all text-layer PDFs including
    those with heavy imagery alongside text.

    Args:
        path: Path to the PDF file

    Returns:
        ExtractedContent with text, title, page count, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        reader = PdfReader(path)

        # Extract text via markitdown (preserves structure, tables, headings)
        text = extract_text_markitdown(path)

        # Get metadata
        metadata: dict[str, Any] = dict(reader.metadata) if reader.metadata else {}
        title = metadata.get("/Title") or metadata.get("Title")
        if isinstance(title, bytes):
            title = title.decode("utf-8", errors="replace")

        # Calculate word count
        word_count = len(text.split()) if text else 0

        # Build metadata dict
        meta = {}
        if metadata.get("/Author"):
            meta["author"] = str(metadata.get("/Author"))
        if metadata.get("/Subject"):
            meta["subject"] = str(metadata.get("/Subject"))
        if metadata.get("/Creator"):
            meta["creator"] = str(metadata.get("/Creator"))

        return ExtractedContent(
            text=text,
            title=title if title else None,
            page_count=len(reader.pages),
            word_count=word_count,
            metadata=meta,
        )

    except Exception as e:
        raise ExtractionError(f"Failed to extract PDF content: {e}") from e
