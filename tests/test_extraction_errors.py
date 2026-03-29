"""Tests for handling corrupted and malformed documents.

These tests verify the extraction system handles edge cases gracefully:
- Corrupted PDF/DOCX files
- Binary files with wrong extensions
- Encoding errors
- Truncated files
"""

import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from mcp_docs.extraction import ContentExtractor, extract_content
from mcp_docs.models import DocumentType, ExtractionError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCorruptedPDF:
    """Tests for corrupted PDF handling."""

    def test_invalid_pdf_magic_bytes(self, temp_dir: Path) -> None:
        """Rejects file with wrong magic bytes."""
        fake_pdf = temp_dir / "fake.pdf"
        fake_pdf.write_bytes(b"Not a PDF file at all")

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError) as exc_info:
            extractor.extract(fake_pdf)

        assert "Failed to extract" in str(exc_info.value)

    def test_truncated_pdf(self, temp_dir: Path) -> None:
        """Handles truncated PDF files."""
        truncated_pdf = temp_dir / "truncated.pdf"
        # Valid PDF header but truncated content
        truncated_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(truncated_pdf)

    def test_pdf_with_null_bytes(self, temp_dir: Path) -> None:
        """Handles PDF with embedded null bytes in metadata."""
        # Create a minimal but valid-ish PDF structure
        fake_pdf = temp_dir / "nulls.pdf"
        fake_pdf.write_bytes(
            b"%PDF-1.0\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"xref\n0 3\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000052 00000 n\n"
            b"trailer<</Size 3/Root 1 0 R>>\n"
            b"startxref\n100\n%%EOF"
        )

        extractor = ContentExtractor()
        # May succeed with empty content or fail - both are valid
        try:
            content = extractor.extract(fake_pdf)
            assert content.text is not None  # Whatever we get back
        except ExtractionError:
            pass  # Also acceptable


class TestCorruptedDocx:
    """Tests for corrupted DOCX handling."""

    def test_invalid_docx_not_zip(self, temp_dir: Path) -> None:
        """Rejects DOCX that isn't a valid ZIP."""
        fake_docx = temp_dir / "fake.docx"
        fake_docx.write_bytes(b"This is not a ZIP file")

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError) as exc_info:
            extractor.extract(fake_docx)

        assert "Failed to extract" in str(exc_info.value)

    def test_docx_empty_zip(self, temp_dir: Path) -> None:
        """Handles DOCX that is an empty ZIP."""
        empty_zip = temp_dir / "empty.docx"
        with ZipFile(empty_zip, 'w') as zf:
            pass  # Create empty ZIP

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(empty_zip)

    def test_docx_missing_document_xml(self, temp_dir: Path) -> None:
        """Handles DOCX without document.xml."""
        bad_docx = temp_dir / "missing_doc.docx"
        with ZipFile(bad_docx, 'w') as zf:
            # Add some content but not the required document.xml
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(bad_docx)

    def test_docx_malformed_xml(self, temp_dir: Path) -> None:
        """Handles DOCX with malformed XML content."""
        bad_docx = temp_dir / "bad_xml.docx"
        with ZipFile(bad_docx, 'w') as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", "<<<NOT VALID XML>>>")

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(bad_docx)


class TestCorruptedPptx:
    """Tests for corrupted PPTX handling."""

    def test_invalid_pptx_not_zip(self, temp_dir: Path) -> None:
        """Rejects PPTX that isn't a valid ZIP."""
        fake_pptx = temp_dir / "fake.pptx"
        fake_pptx.write_bytes(b"Not a PPTX file")

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(fake_pptx)

    def test_pptx_empty_zip(self, temp_dir: Path) -> None:
        """Handles PPTX that is an empty ZIP."""
        empty_zip = temp_dir / "empty.pptx"
        with ZipFile(empty_zip, 'w') as zf:
            pass

        extractor = ContentExtractor()
        with pytest.raises(ExtractionError):
            extractor.extract(empty_zip)


class TestBinaryMasquerading:
    """Tests for binary files with document extensions."""

    def test_binary_as_txt(self, temp_dir: Path) -> None:
        """Handles binary data in .txt file."""
        binary_txt = temp_dir / "binary.txt"
        binary_txt.write_bytes(bytes(range(256)))  # All byte values

        extractor = ContentExtractor()
        # Should either extract (with replacement chars) or raise
        try:
            content = extractor.extract(binary_txt)
            # Some content extracted (with possible replacement chars)
            assert content.text is not None
        except (ExtractionError, UnicodeDecodeError):
            pass  # Also acceptable

    def test_binary_as_md(self, temp_dir: Path) -> None:
        """Handles binary data in .md file."""
        binary_md = temp_dir / "binary.md"
        binary_md.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

        extractor = ContentExtractor()
        try:
            content = extractor.extract(binary_md)
            assert content.text is not None
        except (ExtractionError, UnicodeDecodeError):
            pass


class TestEncodingErrors:
    """Tests for encoding edge cases."""

    def test_latin1_in_utf8_file(self, temp_dir: Path) -> None:
        """Handles Latin-1 encoded content."""
        latin1_file = temp_dir / "latin1.txt"
        latin1_file.write_bytes("Café résumé naïve".encode("latin-1"))

        extractor = ContentExtractor()
        try:
            content = extractor.extract(latin1_file)
            # Should decode somehow (maybe with replacement chars)
            assert content.text is not None
        except ExtractionError:
            pass  # Also acceptable

    def test_mixed_encoding(self, temp_dir: Path) -> None:
        """Handles mixed encoding content."""
        mixed_file = temp_dir / "mixed.txt"
        # UTF-8 with some invalid sequences
        mixed_file.write_bytes(b"Hello \xff\xfe World \xc3\xa9")

        extractor = ContentExtractor()
        try:
            content = extractor.extract(mixed_file)
            assert "Hello" in content.text or "World" in content.text
        except ExtractionError:
            pass

    def test_utf16_without_bom(self, temp_dir: Path) -> None:
        """Handles UTF-16 content without BOM."""
        utf16_file = temp_dir / "utf16.txt"
        utf16_file.write_bytes("Hello World".encode("utf-16-le"))

        extractor = ContentExtractor()
        try:
            content = extractor.extract(utf16_file)
            # May decode incorrectly but shouldn't crash
            assert content.text is not None
        except ExtractionError:
            pass


class TestEdgeCases:
    """Tests for other edge cases."""

    def test_very_long_line(self, temp_dir: Path) -> None:
        """Handles files with extremely long lines."""
        long_line_file = temp_dir / "longline.txt"
        long_line_file.write_text("x" * 1_000_000)  # 1MB single line

        extractor = ContentExtractor()
        content = extractor.extract(long_line_file)

        assert len(content.text) == 1_000_000

    def test_many_short_lines(self, temp_dir: Path) -> None:
        """Handles files with many short lines."""
        many_lines_file = temp_dir / "manylines.txt"
        many_lines_file.write_text("\n".join(["line"] * 10000))

        extractor = ContentExtractor()
        content = extractor.extract(many_lines_file)

        assert content.word_count == 10000

    def test_only_whitespace(self, temp_dir: Path) -> None:
        """Handles files with only whitespace."""
        whitespace_file = temp_dir / "whitespace.txt"
        whitespace_file.write_text("   \n\t\n   \r\n   ")

        extractor = ContentExtractor()
        content = extractor.extract(whitespace_file)

        assert content.text.strip() == ""
        assert content.word_count == 0

    def test_null_terminated_content(self, temp_dir: Path) -> None:
        """Handles content with embedded null characters."""
        null_file = temp_dir / "nulls.txt"
        null_file.write_bytes(b"Hello\x00World\x00!")

        extractor = ContentExtractor()
        try:
            content = extractor.extract(null_file)
            assert content.text is not None
        except ExtractionError:
            pass

    def test_file_permissions_read_only(self, temp_dir: Path) -> None:
        """Can read read-only files."""
        import os

        readonly_file = temp_dir / "readonly.txt"
        readonly_file.write_text("Read-only content")
        os.chmod(readonly_file, 0o444)

        try:
            extractor = ContentExtractor()
            content = extractor.extract(readonly_file)
            assert content.text == "Read-only content"
        finally:
            os.chmod(readonly_file, 0o644)  # Restore for cleanup
