"""Plain text, markdown, RTF, HTML, CSV, EPUB, and XML extraction."""

from pathlib import Path

from striprtf.striprtf import rtf_to_text

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.models import ExtractedContent, ExtractionError


def extract_text(path: Path) -> ExtractedContent:
    """
    Extract content from a plain text file.

    Args:
        path: Path to the text file

    Returns:
        ExtractedContent with text and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract text content: {e}") from e


def extract_markdown(path: Path) -> ExtractedContent:
    """
    Extract content from a Markdown file.

    Attempts to extract a title from the first H1 heading.

    Args:
        path: Path to the markdown file

    Returns:
        ExtractedContent with text, optional title, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)

        # Try to extract title from first H1
        title = None
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                break

        word_count = len(text.split()) if text else 0

        return ExtractedContent(
            text=text,
            title=title,
            page_count=None,
            word_count=word_count,
            metadata={},
        )

    except Exception as e:
        raise ExtractionError(f"Failed to extract markdown content: {e}") from e


def extract_rtf(path: Path) -> ExtractedContent:
    """
    Extract content from an RTF file.

    Uses striprtf since markitdown has no RTF converter.

    Args:
        path: Path to the RTF file

    Returns:
        ExtractedContent with text and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        # Try common encodings for RTF files
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        raw_content = None

        for encoding in encodings:
            try:
                raw_content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if raw_content is None:
            raw_content = path.read_text(encoding="utf-8", errors="replace")

        text = rtf_to_text(raw_content)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract RTF content: {e}") from e


def extract_html(path: Path) -> ExtractedContent:
    """
    Extract content from an HTML file.

    Args:
        path: Path to the HTML file

    Returns:
        ExtractedContent with markdown-converted text and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract HTML content: {e}") from e


def extract_csv(path: Path) -> ExtractedContent:
    """
    Extract content from a CSV file as a markdown table.

    Args:
        path: Path to the CSV file

    Returns:
        ExtractedContent with markdown table representation

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract CSV content: {e}") from e


def extract_epub(path: Path) -> ExtractedContent:
    """
    Extract content from an EPUB file.

    Args:
        path: Path to the EPUB file

    Returns:
        ExtractedContent with text and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract EPUB content: {e}") from e


def extract_xml(path: Path) -> ExtractedContent:
    """
    Extract content from an XML file.

    Args:
        path: Path to the XML file

    Returns:
        ExtractedContent with text and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        text = extract_text_markitdown(path)
        word_count = len(text.split()) if text else 0
        return ExtractedContent(
            text=text,
            title=None,
            page_count=None,
            word_count=word_count,
            metadata={},
        )
    except Exception as e:
        raise ExtractionError(f"Failed to extract XML content: {e}") from e
