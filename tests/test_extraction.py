"""Tests for document extraction."""

import tempfile
from pathlib import Path

import pytest

from mcp_docs.extraction import ContentExtractor, extract_content
from mcp_docs.extraction.text import extract_markdown, extract_text
from mcp_docs.models import DocumentType, ExtractionError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTextExtraction:
    """Tests for plain text extraction."""

    def test_extract_text(self, temp_dir: Path) -> None:
        """Can extract content from text file."""
        text_file = temp_dir / "sample.txt"
        text_file.write_text("Hello, World! This is a test.")

        content = extract_text(text_file)

        assert content.text == "Hello, World! This is a test."
        assert content.word_count == 6
        assert content.title is None
        assert content.page_count is None

    def test_extract_text_utf8(self, temp_dir: Path) -> None:
        """Can handle UTF-8 content."""
        text_file = temp_dir / "unicode.txt"
        text_file.write_text("Héllo Wörld! 你好世界 🌍")

        content = extract_text(text_file)
        assert "Héllo Wörld" in content.text
        assert "你好世界" in content.text
        assert "🌍" in content.text

    def test_extract_empty_file(self, temp_dir: Path) -> None:
        """Can handle empty files."""
        text_file = temp_dir / "empty.txt"
        text_file.write_text("")

        content = extract_text(text_file)
        assert content.text == ""
        assert content.word_count == 0


class TestMarkdownExtraction:
    """Tests for markdown extraction."""

    def test_extract_markdown_with_title(self, temp_dir: Path) -> None:
        """Can extract title from H1 heading."""
        md_file = temp_dir / "doc.md"
        md_file.write_text("# My Document\n\nThis is the content.")

        content = extract_markdown(md_file)

        assert content.title == "My Document"
        assert "This is the content" in content.text

    def test_extract_markdown_no_title(self, temp_dir: Path) -> None:
        """Handles markdown without H1."""
        md_file = temp_dir / "no_title.md"
        md_file.write_text("## Section\n\nJust content here.")

        content = extract_markdown(md_file)

        assert content.title is None
        assert "Section" in content.text

    def test_extract_markdown_word_count(self, temp_dir: Path) -> None:
        """Counts words correctly in markdown."""
        md_file = temp_dir / "words.md"
        md_file.write_text("# Title\n\nOne two three four five.")

        content = extract_markdown(md_file)
        # "#", "Title", "One", "two", "three", "four", "five." = 7 words
        assert content.word_count == 7


class TestPptxExtraction:
    """Tests for PPTX extraction."""

    def test_extract_pptx(self, temp_dir: Path) -> None:
        """Can extract content from PPTX file."""
        from pptx import Presentation
        from pptx.util import Inches

        # Create a simple PPTX
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout

        # Add a text box
        txBox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
        tf = txBox.text_frame
        tf.text = "Hello from PowerPoint"

        pptx_file = temp_dir / "test.pptx"
        prs.save(pptx_file)

        # Extract
        from mcp_docs.extraction.office import extract_pptx

        content = extract_pptx(pptx_file)

        assert "Hello from PowerPoint" in content.text
        assert content.page_count == 1  # One slide
        assert content.word_count >= 3

    def test_extract_pptx_multiple_slides(self, temp_dir: Path) -> None:
        """Can extract from multiple slides."""
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()

        # Add two slides
        for i in range(2):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            txBox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
            txBox.text_frame.text = f"Slide {i + 1} content"

        pptx_file = temp_dir / "multi.pptx"
        prs.save(pptx_file)

        from mcp_docs.extraction.office import extract_pptx

        content = extract_pptx(pptx_file)

        assert "Slide 1 content" in content.text
        assert "Slide 2 content" in content.text
        assert content.page_count == 2

    def test_extract_pptx_with_metadata(self, temp_dir: Path) -> None:
        """Extracts metadata from PPTX."""
        from pptx import Presentation

        prs = Presentation()
        prs.core_properties.title = "My Presentation"
        prs.core_properties.author = "Test Author"

        pptx_file = temp_dir / "meta.pptx"
        prs.save(pptx_file)

        from mcp_docs.extraction.office import extract_pptx

        content = extract_pptx(pptx_file)

        assert content.title == "My Presentation"
        assert content.metadata.get("author") == "Test Author"


class TestContentExtractor:
    """Tests for the ContentExtractor class."""

    def test_auto_detect_txt(self, temp_dir: Path) -> None:
        """Auto-detects .txt files."""
        txt_file = temp_dir / "file.txt"
        txt_file.write_text("Plain text content")

        extractor = ContentExtractor()
        content = extractor.extract(txt_file)

        assert content.text == "Plain text content"

    def test_auto_detect_md(self, temp_dir: Path) -> None:
        """Auto-detects .md files."""
        md_file = temp_dir / "file.md"
        md_file.write_text("# Markdown Title\n\nContent here.")

        extractor = ContentExtractor()
        content = extractor.extract(md_file)

        assert content.title == "Markdown Title"

    def test_auto_detect_pptx(self, temp_dir: Path) -> None:
        """Auto-detects .pptx files."""
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        txBox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
        txBox.text_frame.text = "PowerPoint content"

        pptx_file = temp_dir / "file.pptx"
        prs.save(pptx_file)

        extractor = ContentExtractor()
        content = extractor.extract(pptx_file)

        assert "PowerPoint content" in content.text

    def test_explicit_type(self, temp_dir: Path) -> None:
        """Can specify explicit document type."""
        # File with .md extension but treated as plain text
        md_file = temp_dir / "file.md"
        md_file.write_text("# Not A Title\n\nJust text.")

        extractor = ContentExtractor()
        content = extractor.extract(md_file, doc_type=DocumentType.TXT)

        # As plain text, title should be None (no extraction)
        assert content.title is None

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Raises FileNotFoundError for missing files."""
        extractor = ContentExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract(temp_dir / "nonexistent.txt")

    def test_can_extract(self) -> None:
        """can_extract returns correct results for supported types."""
        extractor = ContentExtractor()

        assert extractor.can_extract(DocumentType.PDF) is True
        assert extractor.can_extract(DocumentType.DOCX) is True
        assert extractor.can_extract(DocumentType.PPTX) is True
        assert extractor.can_extract(DocumentType.TXT) is True
        assert extractor.can_extract(DocumentType.MD) is True
        # DOC/PPT supported via MarkItDown (best-effort for legacy formats)
        assert extractor.can_extract(DocumentType.DOC) is True
        assert extractor.can_extract(DocumentType.PPT) is True


class TestExtractContentFunction:
    """Tests for the extract_content convenience function."""

    def test_extract_content_text(self, temp_dir: Path) -> None:
        """Convenience function works for text files."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Test content for extraction.")

        content = extract_content(txt_file)

        assert content.text == "Test content for extraction."
        assert content.word_count == 4

    def test_extract_content_markdown(self, temp_dir: Path) -> None:
        """Convenience function works for markdown files."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Doc Title\n\nBody text here.")

        content = extract_content(md_file)

        assert content.title == "Doc Title"
