"""Regression tests for all-pages-failed OCR handling (issue #39).

When the vision OCR fallback runs on a scanned PDF and every page fails, each
page is rendered as a ``[OCR failed for page N]`` placeholder. Those placeholders
must not be accepted as the document body or written to the OCR cache. The
cache-level behaviour (what is cached, retried, or resumed) is covered in
``test_ocr_partial_retry.py``.

All tests are pure-python: the OCR boundary is monkeypatched, so nothing hits a
vision endpoint, Qdrant, or the network.
"""

from pathlib import Path

import pytest

from mcp_docs.extraction import extractor as extractor_mod
from mcp_docs.extraction import ocr as ocr_mod
from mcp_docs.extraction.extractor import ContentExtractor
from mcp_docs.models import ExtractedContent, ExtractionError


def _all_failed_ocr(page_count: int) -> ExtractedContent:
    """An OCR result where every page failed (body is only placeholders)."""
    text = "\n\n".join(
        f"--- Page {n} ---\n\n[OCR failed for page {n}]"
        for n in range(1, page_count + 1)
    )
    return ExtractedContent(
        text=text,
        title=None,
        page_count=page_count,
        word_count=len(text.split()),
        metadata={
            "ocr_method": "vision_llm",
            "ocr_failed_pages": list(range(1, page_count + 1)),
        },
    )


class TestAllPagesFailedOcr:
    def test_all_pages_failed_not_accepted_when_markitdown_empty(self, monkeypatch):
        """An all-failed OCR result over an empty markitdown extraction is
        treated as unextractable, not indexed as placeholder text."""
        empty = ExtractedContent(
            text="", title=None, page_count=2, word_count=0, metadata={}
        )
        monkeypatch.setattr(extractor_mod, "extract_pdf", lambda p: empty)
        monkeypatch.setattr(
            ocr_mod, "extract_scanned_pdf_sync", lambda p: _all_failed_ocr(2)
        )
        monkeypatch.setattr(
            extractor_mod.settings, "ocr_vision_url", "http://vision.invalid"
        )

        with pytest.raises(ExtractionError):
            ContentExtractor()._extract_pdf(Path("scanned.pdf"))

    def test_all_pages_failed_falls_back_to_thin_markitdown(self, monkeypatch):
        """When markitdown got some (thin) text, an all-failed OCR result is
        discarded in its favor rather than overwriting it with placeholders."""
        thin = ExtractedContent(
            text="a little text", title=None, page_count=2, word_count=3, metadata={}
        )
        monkeypatch.setattr(extractor_mod, "extract_pdf", lambda p: thin)
        monkeypatch.setattr(
            ocr_mod, "extract_scanned_pdf_sync", lambda p: _all_failed_ocr(2)
        )
        monkeypatch.setattr(
            extractor_mod.settings, "ocr_vision_url", "http://vision.invalid"
        )

        result = ContentExtractor()._extract_pdf(Path("scanned.pdf"))

        assert "[OCR failed for page" not in result.text
        assert result.text == "a little text"
