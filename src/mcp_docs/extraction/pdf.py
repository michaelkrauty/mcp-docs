"""PDF content extraction using pypdf."""

import logging
from pathlib import Path

from pypdf import PdfReader

from mcp_docs.models import ExtractedContent, ExtractionError

logger = logging.getLogger(__name__)


def extract_pdf(path: Path) -> ExtractedContent:
    """
    Extract text content from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        ExtractedContent with text, title, page count, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        reader = PdfReader(path)

        # Extract text from all pages
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        text = "\n\n".join(text_parts)

        # Get metadata
        metadata = reader.metadata or {}
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


def is_scanned_pdf(path: Path) -> bool:
    """
    Check if a PDF appears to be scanned (image-based with no text).

    Args:
        path: Path to the PDF file

    Returns:
        True if PDF appears to be scanned (no extractable text)
    """
    try:
        reader = PdfReader(path)

        # Check first few pages for text
        pages_to_check = min(3, len(reader.pages))
        total_text = ""

        for i in range(pages_to_check):
            page_text = reader.pages[i].extract_text()
            if page_text:
                total_text += page_text

        # If we found very little text, likely scanned
        # Threshold: less than 50 chars per page checked
        return len(total_text.strip()) < (pages_to_check * 50)

    except Exception as e:
        logger.debug(f"OCR check failed for {path}, assuming no OCR needed: {e}")
        return False
