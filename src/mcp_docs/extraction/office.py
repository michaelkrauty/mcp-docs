"""Office document extraction using markitdown for text, python-docx/python-pptx for metadata."""

from pathlib import Path

from docx import Document as DocxDocument
from pptx import Presentation

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.extraction.text import extract_rtf
from mcp_docs.models import ExtractedContent, ExtractionError


def extract_docx(path: Path) -> ExtractedContent:
    """
    Extract text content from a DOCX file.

    Args:
        path: Path to the DOCX file

    Returns:
        ExtractedContent with text, title, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        doc = DocxDocument(path)

        # Extract text via markitdown (preserves tables, headings, lists)
        text = extract_text_markitdown(path)

        # Get metadata from core properties
        title = None
        meta = {}
        core_props = doc.core_properties

        if core_props.title:
            title = core_props.title
        if core_props.author:
            meta["author"] = core_props.author
        if core_props.subject:
            meta["subject"] = core_props.subject
        if core_props.keywords:
            meta["keywords"] = core_props.keywords

        # Calculate word count
        word_count = len(text.split()) if text else 0

        return ExtractedContent(
            text=text,
            title=title,
            page_count=None,  # DOCX doesn't have fixed pages
            word_count=word_count,
            metadata=meta,
        )

    except Exception as e:
        raise ExtractionError(f"Failed to extract DOCX content: {e}") from e


def extract_doc(path: Path) -> ExtractedContent:
    """
    Extract text content from a DOC file (legacy Word format).

    Detects if the file is actually RTF format (common mislabeling)
    and routes to the RTF extractor. Otherwise raises an error.

    Args:
        path: Path to the DOC file

    Returns:
        ExtractedContent

    Raises:
        ExtractionError: If DOC format is not supported
    """
    # Check if this is actually an RTF file disguised as .doc
    # RTF files start with {\rtf
    try:
        with open(path, "rb") as f:
            magic = f.read(5)
        if magic.startswith(b"{\\rtf"):
            # RTF content — route to striprtf-based extractor
            return extract_rtf(path)
    except OSError:
        pass  # Fall through to error

    raise ExtractionError(
        f"Legacy DOC format not directly supported: {path.name}. "
        "Please convert to DOCX format."
    )


def extract_pptx(path: Path) -> ExtractedContent:
    """
    Extract text content from a PPTX file.

    Args:
        path: Path to the PPTX file

    Returns:
        ExtractedContent with text, title, slide count, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    try:
        prs = Presentation(path)

        # Extract text via markitdown (preserves slide structure)
        text = extract_text_markitdown(path)

        # Get metadata from core properties
        title = None
        meta = {}
        core_props = prs.core_properties

        if core_props.title:
            title = core_props.title
        if core_props.author:
            meta["author"] = core_props.author
        if core_props.subject:
            meta["subject"] = core_props.subject
        if core_props.keywords:
            meta["keywords"] = core_props.keywords

        # Calculate word count
        word_count = len(text.split()) if text else 0

        return ExtractedContent(
            text=text,
            title=title,
            page_count=len(prs.slides),  # Slide count as "pages"
            word_count=word_count,
            metadata=meta,
        )

    except Exception as e:
        raise ExtractionError(f"Failed to extract PPTX content: {e}") from e


def extract_ppt(path: Path) -> ExtractedContent:
    """
    Extract text content from a PPT file (legacy PowerPoint format).

    Tries markitdown first; falls back to an error since python-pptx
    doesn't handle legacy .ppt.

    Args:
        path: Path to the PPT file

    Returns:
        ExtractedContent

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
        raise ExtractionError(
            f"Legacy PPT format not directly supported: {path.name}. "
            "Please convert to PPTX format."
        ) from e


def extract_xlsx(path: Path) -> ExtractedContent:
    """
    Extract content from an XLSX (or legacy XLS) spreadsheet.

    Args:
        path: Path to the spreadsheet file

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
        raise ExtractionError(f"Failed to extract spreadsheet content: {e}") from e
