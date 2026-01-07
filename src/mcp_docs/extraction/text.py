"""Plain text and markdown extraction."""

from pathlib import Path

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
        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        text = None

        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            # Fallback with error replacement
            text = path.read_text(encoding="utf-8", errors="replace")

        # Calculate word count
        word_count = len(text.split()) if text else 0

        return ExtractedContent(
            text=text,
            title=None,  # Plain text has no metadata
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
        # Try common encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        text = None

        for encoding in encodings:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            text = path.read_text(encoding="utf-8", errors="replace")

        # Try to extract title from first H1
        title = None
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                break

        # Calculate word count
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
