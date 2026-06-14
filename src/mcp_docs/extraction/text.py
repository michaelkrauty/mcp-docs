"""Plain text, markdown, RTF, HTML, CSV, EPUB, and XML extraction."""

import csv
import io
from pathlib import Path

from striprtf.striprtf import rtf_to_text

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.models import ExtractedContent, ExtractionError

# Encodings tried, in order, when reading a text file. utf-8-sig comes first so
# a UTF-8 BOM is stripped rather than left as a stray ﻿ in the first cell
# (it decodes BOM-less UTF-8 identically); latin-1/cp1252 cover the common
# non-UTF-8 real-world files (accented names, smart quotes).
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


def _read_text_with_encoding_fallback(path: Path) -> str:
    """Read a text file, trying common encodings before a lossy UTF-8 fallback.

    A UTF-16/UTF-32 byte-order mark is honored first: those encodings (e.g.
    Excel "Unicode Text" CSV exports) interleave NUL bytes that latin-1 below
    would happily decode into gibberish, since latin-1 never raises. After that
    we try UTF-8 (BOM-aware) then the common single-byte encodings, decoding
    lossily only as a last resort so extraction never hard-fails on encoding.
    """
    data = path.read_bytes()
    # Check 4-byte UTF-32 BOMs before the 2-byte UTF-16 ones (a UTF-32LE BOM
    # starts with the UTF-16LE BOM).
    for bom, encoding in (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if data.startswith(bom):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                break
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _csv_to_markdown_table(raw: str) -> str:
    """Render CSV text as a GitHub-flavored markdown table.

    Uses the ``csv`` module so quoted fields, embedded commas, and embedded
    newlines parse correctly. Ragged rows are padded to the widest row; pipes
    are escaped and embedded newlines flattened so each record stays one row.
    """
    rows = list(csv.reader(io.StringIO(raw)))
    # Drop trailing fully-blank rows (common trailing-newline artifact).
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    if not rows:
        return ""

    width = max(len(row) for row in rows)

    def _cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()

    def _row(cells: list[str]) -> str:
        padded = list(cells) + [""] * (width - len(cells))
        return "| " + " | ".join(_cell(c) for c in padded) + " |"

    lines = [_row(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(_row(row) for row in rows[1:])
    return "\n".join(lines)


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
        raw_content = _read_text_with_encoding_fallback(path)
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

    Reads with an encoding fallback and renders the table directly with the
    ``csv`` module rather than delegating to markitdown, whose CSV converter
    decodes with the locale default (often ASCII) and raises a
    ``UnicodeDecodeError`` on any non-ASCII content — e.g. accented merchant
    names in a financial export.

    Args:
        path: Path to the CSV file

    Returns:
        ExtractedContent with markdown table representation

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        raw = _read_text_with_encoding_fallback(path)
        text = _csv_to_markdown_table(raw)
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
