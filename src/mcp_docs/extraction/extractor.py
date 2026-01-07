"""Content extraction dispatcher."""

from pathlib import Path

from mcp_docs.extraction.office import extract_doc, extract_docx, extract_ppt, extract_pptx
from mcp_docs.extraction.pdf import extract_pdf, is_scanned_pdf
from mcp_docs.extraction.text import extract_markdown, extract_text
from mcp_docs.models import DocumentType, ExtractedContent, ExtractionError
from mcp_docs.settings import settings


class ContentExtractor:
    """
    Extracts content from documents based on their type.

    Handles routing to appropriate extractors and manages
    extraction settings like OCR and file size limits.
    """

    def __init__(
        self,
        ocr_enabled: bool | None = None,
        max_file_size_mb: int | None = None,
    ):
        """
        Initialize the content extractor.

        Args:
            ocr_enabled: Whether to use OCR for scanned documents.
                        Defaults to settings.ocr_enabled.
            max_file_size_mb: Maximum file size to process in MB.
                             Defaults to settings.max_file_size_mb.
        """
        self.ocr_enabled = ocr_enabled if ocr_enabled is not None else settings.ocr_enabled
        self.max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else settings.max_file_size_mb

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

        # Check file size
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ExtractionError(
                f"File too large: {size_mb:.1f}MB exceeds limit of {self.max_file_size_mb}MB"
            )

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
        elif doc_type == DocumentType.UNKNOWN:
            # Try as plain text
            return extract_text(path)
        else:
            raise ExtractionError(f"Unsupported document type: {doc_type.value}")

    def _extract_pdf(self, path: Path) -> ExtractedContent:
        """
        Extract content from PDF, handling scanned documents.

        Args:
            path: Path to the PDF file

        Returns:
            ExtractedContent

        Raises:
            ExtractionError: If PDF is scanned and OCR is disabled
        """
        # Check if scanned
        if is_scanned_pdf(path):
            if not self.ocr_enabled:
                raise ExtractionError(
                    f"Scanned PDF detected and OCR is disabled: {path.name}. "
                    "Enable OCR with DOCS_OCR_ENABLED=true."
                )

            # Use vision LLM OCR
            import asyncio

            from mcp_docs.extraction.ocr import extract_scanned_pdf_cached

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop - create one
                return asyncio.run(extract_scanned_pdf_cached(path))
            else:
                # Already in async context - need to run in new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, extract_scanned_pdf_cached(path)
                    )
                    return future.result()

        return extract_pdf(path)

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
            DocumentType.PPTX,
            DocumentType.TXT,
            DocumentType.MD,
        }
        return doc_type in supported


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
