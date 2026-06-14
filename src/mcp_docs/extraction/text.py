"""Plain text, markdown, RTF, HTML, CSV, EPUB, and XML extraction."""

import csv
import io
from pathlib import Path

from striprtf.striprtf import rtf_to_text

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.models import ExtractedContent, ExtractionError

# Encodings tried in order after any BOM check. utf-8-sig precedes utf-8 so a
# UTF-8 BOM is stripped (it decodes BOM-less UTF-8 identically). cp1252 precedes
# latin-1 because latin-1 never raises and would otherwise map Windows-1252
# smart quotes/euro to C1 control chars and shadow cp1252; latin-1 stays last as
# the never-fail catch-all (cp1252 falls through to it on its few undefined
# bytes). Deterministic order is deliberate: heuristic charset detection
# (charset-normalizer) mis-identifies short Western files and would corrupt the
# overwhelmingly-common UTF-8/Windows-1252/latin-1 case it is meant to help.
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _read_text_with_encoding_fallback(path: Path) -> str:
    """Read a text file robustly, regardless of encoding or locale.

    A UTF-16/UTF-32 byte-order mark is honored first: those encodings (e.g.
    Excel "Unicode Text" CSV exports) interleave NUL bytes that latin-1 below
    would happily decode into gibberish, since latin-1 never raises. After that
    the encodings in ``_TEXT_ENCODINGS`` are tried in order, with a lossy UTF-8
    decode only as a last resort so extraction never hard-fails on encoding.
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
    # BOM-less UTF-16 (e.g. some Windows exports): a high density of NUL bytes
    # is the signature — ASCII text in UTF-16 is ~half NULs. UTF-8 below would
    # otherwise "succeed" (NUL is valid UTF-8) and return a string riddled with
    # embedded NULs. Pick the endianness from NUL position (UTF-16LE ASCII has
    # NULs at odd indices, UTF-16BE at even) and require a clean (NUL-free)
    # decode, so UTF-16BE input is not mis-accepted as mojibake'd UTF-16LE.
    if data and data.count(0) / len(data) > 0.25:
        nul_odd = sum(1 for i in range(1, len(data), 2) if data[i] == 0)
        nul_even = sum(1 for i in range(0, len(data), 2) if data[i] == 0)
        candidates = (
            ("utf-16-le", "utf-16-be")
            if nul_odd >= nul_even
            else ("utf-16-be", "utf-16-le")
        )
        for encoding in candidates:
            try:
                decoded = data.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "\x00" not in decoded:
                return decoded
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
    # newline="" lets the csv module handle CR/LF/CR-only record separators and
    # newlines embedded in quoted fields, instead of StringIO pre-translating
    # them (CR-only input otherwise raises "new-line character seen in
    # unquoted field").
    rows = list(csv.reader(io.StringIO(raw, newline="")))
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
    ``csv`` module instead of delegating to markitdown. markitdown's CSV
    converter is unreliable for the encodings this corpus actually contains: it
    raises ``UnicodeDecodeError`` on non-ASCII Windows-1252/latin-1 exports
    (locale-default ASCII decoding), mojibakes cp1252 smart quotes/euro by
    decoding them as latin-1, and leaves a UTF-8 BOM as a stray character in the
    first cell — all verified. The direct path handles UTF-8 (BOM-aware),
    Windows-1252, latin-1, and BOM/BOM-less UTF-16 correctly and deterministically
    (independent of locale). The trade-off is that genuinely single-byte
    non-Western encodings (e.g. Shift-JIS), which markitdown can charset-detect,
    fall back to best-effort latin-1; those do not occur in this corpus and
    markitdown's CSV path mishandles the encodings that do.

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
