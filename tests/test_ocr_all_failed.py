"""Regression tests for all-pages-failed OCR handling (issue #39).

When the vision OCR fallback runs on a scanned PDF and every page fails, each
page is rendered as a ``[OCR failed for page N]`` placeholder. Those placeholders
must not be accepted as the document body or written to the OCR cache.

All tests are pure-python: the OCR boundary is monkeypatched, so nothing hits a
vision endpoint, Qdrant, or the network.
"""

import asyncio
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

    def test_all_pages_failed_not_cached(self, monkeypatch):
        """A fully-failed OCR result must not be written to the cache."""
        monkeypatch.setattr(ocr_mod.settings, "ocr_cache_enabled", True)

        async def fake_extract(path):
            return _all_failed_ocr(3)

        saved: list[tuple] = []
        monkeypatch.setattr(ocr_mod, "extract_scanned_pdf", fake_extract)
        monkeypatch.setattr(ocr_mod, "_get_cache_key", lambda path: "key123")
        monkeypatch.setattr(ocr_mod, "_load_from_cache", lambda key: None)
        monkeypatch.setattr(
            ocr_mod, "_save_to_cache", lambda key, result: saved.append((key, result))
        )

        result = asyncio.run(ocr_mod._extract_with_cache(Path("scanned.pdf")))

        assert saved == []  # not cached
        assert "[OCR failed for page" in result.text  # still returned to caller

    def test_partial_failure_is_still_cached(self, monkeypatch):
        """A partially-failed OCR result (some pages succeeded) is still cached;
        only a complete failure is skipped."""
        monkeypatch.setattr(ocr_mod.settings, "ocr_cache_enabled", True)
        partial = ExtractedContent(
            text=(
                "--- Page 1 ---\n\nthis page has plenty of real extracted text "
                "from the document\n\n--- Page 2 ---\n\n[OCR failed for page 2]"
            ),
            title=None,
            page_count=2,
            word_count=14,
            metadata={"ocr_method": "vision_llm", "ocr_failed_pages": [2]},
        )

        async def fake_extract(path):
            return partial

        saved: list[tuple] = []
        monkeypatch.setattr(ocr_mod, "extract_scanned_pdf", fake_extract)
        monkeypatch.setattr(ocr_mod, "_get_cache_key", lambda path: "key123")
        monkeypatch.setattr(ocr_mod, "_load_from_cache", lambda key: None)
        monkeypatch.setattr(
            ocr_mod, "_save_to_cache", lambda key, result: saved.append((key, result))
        )

        asyncio.run(ocr_mod._extract_with_cache(Path("scanned.pdf")))

        assert len(saved) == 1  # cached normally

    def test_cached_all_failed_is_ignored_and_reextracted(self, monkeypatch):
        """A previously-cached fully-failed result is ignored on read, so OCR is
        retried (self-healing) rather than serving placeholder text."""
        monkeypatch.setattr(ocr_mod.settings, "ocr_cache_enabled", True)
        monkeypatch.setattr(ocr_mod, "_get_cache_key", lambda path: "key123")
        # Cache holds a poisoned all-failed result written before the guard.
        monkeypatch.setattr(
            ocr_mod, "_load_from_cache", lambda key: _all_failed_ocr(2)
        )

        good = ExtractedContent(
            text="real recovered text",
            title=None,
            page_count=2,
            word_count=3,
            metadata={"ocr_method": "vision_llm"},
        )
        calls: list = []

        async def fake_extract(path):
            calls.append(path)
            return good

        saved: list = []
        monkeypatch.setattr(ocr_mod, "extract_scanned_pdf", fake_extract)
        monkeypatch.setattr(
            ocr_mod, "_save_to_cache", lambda key, result: saved.append(result)
        )

        result = asyncio.run(ocr_mod._extract_with_cache(Path("scanned.pdf")))

        assert calls  # OCR was retried instead of returning the cached placeholder
        assert "[OCR failed for page" not in result.text
        assert result.text == "real recovered text"
        assert saved == [good]  # the recovered result is cached
