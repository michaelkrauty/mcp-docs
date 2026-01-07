"""Office document extraction using python-docx and python-pptx."""

from pathlib import Path

from docx import Document as DocxDocument
from pptx import Presentation

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

        # Extract text from paragraphs
        paragraphs: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        text = "\n\n".join(paragraphs)

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

    Note: This requires antiword or similar external tool.
    For now, we raise an error suggesting conversion to DOCX.

    Args:
        path: Path to the DOC file

    Returns:
        ExtractedContent

    Raises:
        ExtractionError: Always, as DOC format is not directly supported
    """
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

        # Extract text from all slides
        text_parts: list[str] = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_texts: list[str] = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())

                # Also extract from tables
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_texts.append(row_text)

            if slide_texts:
                text_parts.append(f"--- Slide {slide_num} ---\n" + "\n".join(slide_texts))

        text = "\n\n".join(text_parts)

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

    Note: PPT format is not directly supported.
    For now, we raise an error suggesting conversion to PPTX.

    Args:
        path: Path to the PPT file

    Returns:
        ExtractedContent

    Raises:
        ExtractionError: Always, as PPT format is not directly supported
    """
    raise ExtractionError(
        f"Legacy PPT format not directly supported: {path.name}. "
        "Please convert to PPTX format."
    )
