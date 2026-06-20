"""Content extraction dispatcher."""

import logging
from pathlib import Path

from mcp_docs.extraction.notebook import extract_ipynb
from mcp_docs.extraction.office import (
    extract_doc,
    extract_docx,
    extract_ppt,
    extract_pptx,
    extract_xlsx,
)
from mcp_docs.extraction.pdf import extract_pdf
from mcp_docs.extraction.text import (
    extract_csv,
    extract_epub,
    extract_html,
    extract_markdown,
    extract_rtf,
    extract_text,
    extract_xml,
)
from mcp_docs.models import DocumentType, ExtractedContent, ExtractionError
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)

# Minimum words per page to consider extraction sufficient.
# Below this threshold, the PDF is likely scanned/image-based and OCR is attempted.
_MIN_WORDS_PER_PAGE = 10


class ContentExtractor:
    """
    Extracts content from documents based on their type.

    For PDFs, tries markitdown extraction first (fast, local). If the result
    has insufficient text (likely a scanned/image-based PDF), falls back to
    vision LLM OCR when configured.
    """

    def extract(self, path: Path, doc_type: DocumentType | None = None) -> ExtractedContent:
        """
        Extract content from a document.

        Args:
            path: Path to the document
            doc_type: Document type (auto-detected if not provided)

        Returns:
            ExtractedContent with text, metadata, and statistics

        Raises:
            ExtractionError: If extraction fails or document is unsupported
            FileNotFoundError: If path does not exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        # Auto-detect type if not provided
        if doc_type is None:
            doc_type = DocumentType.from_extension(path.suffix)

        # Route to appropriate extractor
        if doc_type == DocumentType.PDF:
            return self._extract_pdf(path)
        elif doc_type == DocumentType.DOCX:
            return extract_docx(path)
        elif doc_type == DocumentType.DOC:
            return extract_doc(path)
        elif doc_type == DocumentType.PPTX:
            return extract_pptx(path)
        elif doc_type == DocumentType.PPT:
            return extract_ppt(path)
        elif doc_type == DocumentType.TXT:
            return extract_text(path)
        elif doc_type in (DocumentType.MD,):
            return extract_markdown(path)
        elif doc_type == DocumentType.RTF:
            return extract_rtf(path)
        elif doc_type == DocumentType.HTML:
            return extract_html(path)
        elif doc_type == DocumentType.XLSX:
            return extract_xlsx(path)
        elif doc_type == DocumentType.CSV:
            return extract_csv(path)
        elif doc_type == DocumentType.EPUB:
            return extract_epub(path)
        elif doc_type == DocumentType.XML:
            return extract_xml(path)
        elif doc_type == DocumentType.IPYNB:
            return extract_ipynb(path)
        elif doc_type == DocumentType.ODT:
            raise ExtractionError(
                f"ODT format not directly supported: {path.name}. "
                "Please convert to DOCX format."
            )
        elif doc_type == DocumentType.UNKNOWN:
            # Try as plain text
            return extract_text(path)
        else:
            raise ExtractionError(f"Unsupported document type: {doc_type.value}")

    def _extract_pdf(self, path: Path) -> ExtractedContent:
        """
        Extract content from PDF with automatic OCR fallback.

        Strategy:
        1. Try markitdown (pdfplumber + pdfminer) — fast, local, no deps
        2. If insufficient text extracted, attempt vision LLM OCR
        3. If OCR unavailable/fails, return whatever markitdown got

        Args:
            path: Path to the PDF file

        Returns:
            ExtractedContent
        """
        # 1. Always try markitdown first
        result = extract_pdf(path)

        # 2. Check if we got enough text
        if _has_sufficient_text(result):
            return result

        # 3. Insufficient text — likely scanned. Try OCR fallback.
        logger.info(
            f"Thin text extraction ({result.word_count} words from "
            f"{result.page_count} pages) for {path.name}, attempting OCR fallback"
        )

        if not settings.ocr_vision_url:
            if result.word_count > 0:
                logger.warning(
                    f"OCR not configured (DOCS_OCR_VISION_URL not set). "
                    f"Returning thin extraction for {path.name}."
                )
                return result
            raise ExtractionError(
                f"No text could be extracted from {path.name}. "
                "The PDF may be scanned/image-based. "
                "Set DOCS_OCR_VISION_URL to an OpenAI-compatible vision endpoint "
                "and install mcp-docs[ocr] to enable OCR for scanned PDFs."
            )

        try:
            from mcp_docs.extraction.ocr import (
                _all_pages_failed,
                extract_scanned_pdf_sync,
            )

            ocr_result = extract_scanned_pdf_sync(path)
        except Exception as e:
            logger.warning(f"OCR fallback failed for {path.name}: {e}")
            if result.word_count > 0:
                return result
            raise ExtractionError(
                f"No text could be extracted from {path.name} and OCR fallback failed: {e}. "
                "The PDF may be scanned/image-based. Check that your vision endpoint "
                f"({settings.ocr_vision_url}) is running and mcp-docs[ocr] is installed."
            ) from e

        # Discard an OCR result whose every page failed. Each failed page is a
        # "[OCR failed for page N]" placeholder, so the result's word_count is
        # non-zero and would otherwise beat a thin markitdown result and be
        # indexed as the document body. OCR ran without error here, so this is a
        # deliberate rejection, not a fallback failure.
        if _all_pages_failed(ocr_result):
            logger.warning(
                f"OCR failed for every page of {path.name}; discarding placeholder result"
            )
            if result.word_count > 0:
                return result
            raise ExtractionError(
                f"No text could be extracted from {path.name}: markitdown found "
                "no usable text and OCR failed for every page. The PDF may be "
                "scanned/image-based; check that the vision endpoint "
                f"({settings.ocr_vision_url}) is reachable and mcp-docs[ocr] is installed."
            )

        # Only use OCR result if it actually got more content
        if ocr_result.word_count > result.word_count:
            logger.info(
                f"OCR extracted {ocr_result.word_count} words "
                f"(vs {result.word_count} from markitdown)"
            )
            return ocr_result

        logger.info("OCR didn't improve extraction, using markitdown result")
        return result

    def can_extract(self, doc_type: DocumentType) -> bool:
        """
        Check if a document type is supported for extraction.

        Args:
            doc_type: The document type to check

        Returns:
            True if the type is supported
        """
        supported = {
            DocumentType.PDF,
            DocumentType.DOCX,
            DocumentType.DOC,
            DocumentType.PPTX,
            DocumentType.PPT,
            DocumentType.TXT,
            DocumentType.MD,
            DocumentType.RTF,
            DocumentType.HTML,
            DocumentType.XLSX,
            DocumentType.CSV,
            DocumentType.EPUB,
            DocumentType.XML,
            DocumentType.IPYNB,
        }
        return doc_type in supported


def _has_sufficient_text(result: ExtractedContent) -> bool:
    """Check if extraction result has enough text to be useful."""
    if not result.page_count:
        return bool(result.word_count)
    words_per_page = result.word_count / result.page_count
    return words_per_page >= _MIN_WORDS_PER_PAGE


def extract_content(path: Path, doc_type: DocumentType | None = None) -> ExtractedContent:
    """
    Convenience function to extract content from a document.

    Args:
        path: Path to the document
        doc_type: Document type (auto-detected if not provided)

    Returns:
        ExtractedContent with text, metadata, and statistics

    Raises:
        ExtractionError: If extraction fails
    """
    extractor = ContentExtractor()
    return extractor.extract(path, doc_type)
