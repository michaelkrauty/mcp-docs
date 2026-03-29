"""Office document extraction using markitdown for text, python-docx/python-pptx for metadata."""

import logging
from pathlib import Path

from docx import Document as DocxDocument
from pptx import Presentation

from mcp_docs.extraction.markitdown_extractor import extract_text_markitdown
from mcp_docs.extraction.text import extract_rtf
from mcp_docs.models import ExtractedContent, ExtractionError

logger = logging.getLogger(__name__)

# OLE2 Compound Document magic bytes: D0 CF 11 E0 A1 B1 1A E1
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0"


def _is_ole2(path: Path) -> bool:
    """Check if a file is OLE2/CDF format rather than OOXML (ZIP)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _OLE2_MAGIC
    except OSError:
        return False


def _is_drm_protected(path: Path) -> bool:
    """
    Check if an OLE2 file contains DRM/IRM encryption markers.

    Microsoft IRM (Information Rights Management) uses a specific OLE2
    stream structure with "DRMEncrypted" markers. These files cannot be
    decrypted without the original DRM license server.

    Scans up to 512KB of the file to find DRM markers — OLE2 directory
    entries storing stream names can be beyond 64KB in large files.
    """
    try:
        marker = b"D\x00R\x00M\x00E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d"
        with open(path, "rb") as f:
            # Read in chunks to find marker without loading entire file.
            # OLE2 directory entries are typically within the first 512KB,
            # but we overlap reads to avoid missing markers split across chunks.
            chunk_size = 524288  # 512KB
            overlap = len(marker)
            prev_tail = b""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return False
                search_buf = prev_tail + chunk
                if marker in search_buf:
                    return True
                prev_tail = chunk[-overlap:]
    except OSError:
        return False


def _extract_ole2_as_markitdown(path: Path, ext: str) -> ExtractedContent:
    """
    Try to extract content from an OLE2 file using markitdown.

    OLE2 files with modern extensions (.docx/.pptx) are typically either
    encrypted Office documents or legacy binary format files with wrong
    extensions. markitdown may handle some of these via its own converters.

    Args:
        path: Path to the OLE2 file
        ext: Original extension for context in error messages

    Returns:
        ExtractedContent

    Raises:
        ExtractionError: If extraction fails
    """
    # Check for DRM first — these are unextractable without a license server
    if _is_drm_protected(path):
        raise ExtractionError(
            f"{path.name} is DRM-protected (Microsoft IRM). "
            "Cannot be decrypted without the original DRM license server. "
            "To extract, open on a machine with DRM access and re-save "
            "without IRM protection, or export to PDF."
        )

    # Try markitdown — it handles many OLE2 formats
    try:
        text = extract_text_markitdown(path)
        if text and text.strip():
            word_count = len(text.split())
            logger.info(
                f"Extracted {word_count} words from OLE2 file "
                f"via markitdown: {path.name}"
            )
            return ExtractedContent(
                text=text,
                title=None,
                page_count=None,
                word_count=word_count,
                metadata={"format_note": "OLE2/CDF format with modern extension"},
            )
    except Exception as e:
        logger.debug(f"markitdown failed on OLE2 file {path.name}: {e}")

    raise ExtractionError(
        f"{path.name} has a .{ext} extension but is in OLE2/CDF format "
        "(possibly encrypted or from a legacy system). "
        "Try converting with LibreOffice or decrypting first."
    )


def extract_docx(path: Path) -> ExtractedContent:
    """
    Extract text content from a DOCX file.

    Handles OLE2/CDF files with .docx extension (common in government/military
    environments) by falling back to markitdown.

    Args:
        path: Path to the DOCX file

    Returns:
        ExtractedContent with text, title, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    if _is_ole2(path):
        return _extract_ole2_as_markitdown(path, "docx")

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
    and routes to the RTF extractor. Otherwise tries markitdown.

    Args:
        path: Path to the DOC file

    Returns:
        ExtractedContent

    Raises:
        ExtractionError: If extraction fails
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
        pass

    # Try markitdown for OLE2/binary DOC files
    try:
        text = extract_text_markitdown(path)
        if text and text.strip():
            word_count = len(text.split())
            return ExtractedContent(
                text=text,
                title=None,
                page_count=None,
                word_count=word_count,
                metadata={},
            )
    except Exception as e:
        logger.debug(f"markitdown failed on DOC file {path.name}: {e}")

    raise ExtractionError(
        f"Legacy DOC format not directly supported: {path.name}. "
        "Please convert to DOCX format."
    )


def extract_pptx(path: Path) -> ExtractedContent:
    """
    Extract text content from a PPTX file.

    Handles OLE2/CDF files with .pptx extension by falling back to markitdown.

    Args:
        path: Path to the PPTX file

    Returns:
        ExtractedContent with text, title, slide count, and word count

    Raises:
        ExtractionError: If extraction fails
    """
    if _is_ole2(path):
        return _extract_ole2_as_markitdown(path, "pptx")

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
            f"Failed to extract PPT content from {path.name}: {e}. "
            "If this is a legacy .ppt file, please convert to PPTX format."
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
