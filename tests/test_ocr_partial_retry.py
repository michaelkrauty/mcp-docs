"""Partially failed OCR is retried instead of being cached as the document text.

A page that fails OCR (timeout, endpoint outage, open circuit breaker) used to
become a permanent "[OCR failed for page N]" placeholder: the partial result
was cached by file path/size/mtime and indexed. Completed pages are now kept as
progress, failed pages are retried on the next extraction attempt, and a page
is only given up on after repeated failures attributable to that page.

The vision endpoint and PDF rendering are faked; nothing touches the network.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from mcp_docs.extraction import extractor as extractor_mod
from mcp_docs.extraction import ocr as ocr_mod
from mcp_docs.extraction.extractor import ContentExtractor
from mcp_docs.extraction.ocr import IncompleteOCRError
from mcp_docs.extraction.vision_client import (
    VisionOCRClient,
    VisionOCRError,
    VisionOCRUnavailableError,
)
from mcp_docs.models import ExtractedContent, ExtractionError
from mcp_docs.processing.queue import _is_permanent_failure

PDF = Path("scanned.pdf")


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    VisionOCRClient._failure_count = 0
    VisionOCRClient._circuit_open_until = None
    yield
    VisionOCRClient._failure_count = 0
    VisionOCRClient._circuit_open_until = None


class FakeVision:
    """Scripted per-page OCR outcomes; records every page sent to the endpoint."""

    def __init__(self, page_count: int):
        self.page_count = page_count
        self.outcomes: dict[int, list] = {}
        self.rendered: list[list[int]] = []
        self.sent: list[int] = []

    def fail(self, page: int, *errors: Exception) -> None:
        self.outcomes.setdefault(page, []).extend(errors)

    async def ocr_image(self, image_bytes, page_num=1, image_format="png"):
        self.sent.append(page_num)
        queued = self.outcomes.get(page_num)
        if queued:
            raise queued.pop(0)
        return f"text of page {page_num}"


@pytest.fixture
def vision(tmp_path, monkeypatch):
    fake = FakeVision(page_count=3)
    monkeypatch.setattr(ocr_mod.settings, "ocr_cache_enabled", True)
    monkeypatch.setattr(ocr_mod.settings, "ocr_max_page_attempts", 3)
    monkeypatch.setattr(ocr_mod, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(ocr_mod, "_get_cache_key", lambda path: "key")
    monkeypatch.setattr(ocr_mod, "_count_pages", lambda path: fake.page_count)

    def render(path, pages):
        fake.rendered.append(list(pages))
        return {page: f"image {page}".encode() for page in pages}

    async def ocr_image(self, image_bytes, page_num=1, image_format="png"):
        return await fake.ocr_image(image_bytes, page_num, image_format)

    async def close(self):
        return None

    monkeypatch.setattr(ocr_mod, "_render_pages", render)
    monkeypatch.setattr(VisionOCRClient, "ocr_image", ocr_image)
    monkeypatch.setattr(VisionOCRClient, "close", close)
    return fake


def extract() -> ExtractedContent:
    return asyncio.run(ocr_mod._extract_with_cache(PDF))


def timeout(page: int) -> VisionOCRError:
    return VisionOCRError(f"Vision OCR timeout for page {page}")


def outage() -> VisionOCRUnavailableError:
    return VisionOCRUnavailableError("Vision OCR circuit breaker open")


def full_cache(tmp_path: Path) -> Path:
    return tmp_path / "key.json"


def progress_file(tmp_path: Path) -> Path:
    return tmp_path / "key.progress.json"


def test_partial_failure_raises_retryable_error_and_is_not_cached(vision, tmp_path):
    vision.fail(2, timeout(2))

    with pytest.raises(IncompleteOCRError) as raised:
        extract()

    assert raised.value.failed_pages == [2]
    assert isinstance(raised.value, ExtractionError)
    assert not _is_permanent_failure(str(raised.value))
    assert not full_cache(tmp_path).exists()
    progress = json.loads(progress_file(tmp_path).read_text())
    assert progress["pages"] == {"1": "text of page 1", "3": "text of page 3"}
    assert progress["attempts"] == {"2": 1}


def test_retry_only_sends_failed_pages_and_caches_the_complete_result(vision, tmp_path):
    vision.fail(2, timeout(2))
    with pytest.raises(IncompleteOCRError):
        extract()
    vision.sent.clear()

    result = extract()

    assert vision.sent == [2]
    assert vision.rendered[-1] == [2]
    assert result.text == (
        "--- Page 1 ---\n\ntext of page 1\n\n"
        "--- Page 2 ---\n\ntext of page 2\n\n"
        "--- Page 3 ---\n\ntext of page 3"
    )
    assert "ocr_failed_pages" not in result.metadata
    assert full_cache(tmp_path).exists()
    assert not progress_file(tmp_path).exists()

    vision.sent.clear()
    assert extract().text == result.text
    assert vision.sent == []


def test_endpoint_outages_do_not_use_up_page_attempts(vision, tmp_path):
    for _ in range(5):
        vision.fail(2, outage())
        with pytest.raises(IncompleteOCRError):
            extract()

    progress = json.loads(progress_file(tmp_path).read_text())
    assert progress["attempts"] == {}
    assert extract().text.count("text of page") == 3


def test_page_attempts_are_bounded(vision, tmp_path):
    vision.fail(2, timeout(2), timeout(2), timeout(2))
    for _ in range(2):
        with pytest.raises(IncompleteOCRError):
            extract()

    result = extract()

    assert "[OCR failed for page 2]" in result.text
    assert "text of page 1" in result.text and "text of page 3" in result.text
    assert result.metadata["ocr_failed_pages"] == [2]
    assert result.metadata["ocr_attempts_exhausted"] is True
    assert full_cache(tmp_path).exists()
    assert not progress_file(tmp_path).exists()

    vision.sent.clear()
    assert extract().text == result.text
    assert vision.sent == []


def test_progress_that_cannot_be_saved_is_reported(vision, tmp_path, monkeypatch):
    vision.fail(2, timeout(2))

    def disk_full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ocr_mod, "_write_json_atomic", disk_full)

    with pytest.raises(IncompleteOCRError, match="Progress could not be saved") as raised:
        extract()

    assert raised.value.failed_pages == [2]
    assert isinstance(raised.value.progress_error, OSError)
    assert not progress_file(tmp_path).exists()


def test_progress_is_kept_when_the_finished_result_cannot_be_cached(vision, tmp_path, monkeypatch):
    vision.fail(2, timeout(2))
    with pytest.raises(IncompleteOCRError):
        extract()
    write = ocr_mod._write_json_atomic

    def cache_disk_full(path, data):
        if path == full_cache(tmp_path):
            raise OSError(28, "No space left on device")
        write(path, data)

    monkeypatch.setattr(ocr_mod, "_write_json_atomic", cache_disk_full)
    result = extract()
    assert "ocr_failed_pages" not in result.metadata
    assert not full_cache(tmp_path).exists()

    monkeypatch.setattr(ocr_mod, "_write_json_atomic", write)
    vision.sent.clear()

    assert extract().text == result.text
    assert vision.sent == []
    assert full_cache(tmp_path).exists()
    assert not progress_file(tmp_path).exists()


def test_concurrent_extractions_of_the_same_file_take_turns(vision, tmp_path, monkeypatch):
    first_ocr_started, release = threading.Event(), threading.Event()
    send = vision.ocr_image

    async def held_until_released(image_bytes, page_num=1, image_format="png"):
        first_ocr_started.set()
        assert release.wait(5), "Test did not release the first extraction"
        return await send(image_bytes, page_num, image_format)

    monkeypatch.setattr(vision, "ocr_image", held_until_released)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(extract)
        try:
            assert first_ocr_started.wait(5)
            second = pool.submit(extract)
            # Give an unserialized second extraction time to start its own OCR.
            time.sleep(0.3)
            assert vision.rendered == [[1, 2, 3]]
        finally:
            release.set()
        results = first.result(5), second.result(5)

    # The second extraction waited, then read the first one's cached result.
    assert vision.rendered == [[1, 2, 3]]
    assert sorted(vision.sent) == [1, 2, 3]
    assert results[0].text == results[1].text


def test_a_page_the_renderer_skips_is_charged(vision, tmp_path, monkeypatch):
    def render_without_page_2(path, pages):
        vision.rendered.append(list(pages))
        return {page: f"image {page}".encode() for page in pages if page != 2}

    monkeypatch.setattr(ocr_mod, "_render_pages", render_without_page_2)
    for _ in range(2):
        with pytest.raises(IncompleteOCRError):
            extract()

    result = extract()

    assert result.metadata["ocr_failed_pages"] == [2]
    assert 2 not in vision.sent


def test_every_page_exhausted_is_returned_but_not_cached(vision, tmp_path):
    vision.page_count = 2
    for page in (1, 2):
        vision.fail(page, timeout(page), timeout(page), timeout(page))
    for _ in range(2):
        with pytest.raises(IncompleteOCRError):
            extract()

    result = extract()

    assert ocr_mod._all_pages_failed(result)
    assert not full_cache(tmp_path).exists()


def legacy_entry(tmp_path: Path, text: str, failed: list[int], page_count: int = 3) -> None:
    """A cache entry as written by versions that cached partial results."""
    full_cache(tmp_path).write_text(
        json.dumps(
            {
                "text": text,
                "title": None,
                "page_count": page_count,
                "word_count": len(text.split()),
                "metadata": {"ocr_method": "vision_llm", "ocr_failed_pages": failed},
            }
        )
    )


def test_legacy_partial_cache_entry_keeps_its_completed_pages(vision, tmp_path):
    legacy_entry(
        tmp_path,
        "--- Page 1 ---\n\nold page one\n\n"
        "--- Page 2 ---\n\n[OCR failed for page 2]\n\n"
        "--- Page 3 ---\n\nold page three",
        failed=[2],
    )

    result = extract()

    assert vision.sent == [2]
    assert result.text == (
        "--- Page 1 ---\n\nold page one\n\n"
        "--- Page 2 ---\n\ntext of page 2\n\n"
        "--- Page 3 ---\n\nold page three"
    )
    assert "ocr_failed_pages" not in json.loads(full_cache(tmp_path).read_text())["metadata"]


def test_legacy_entry_keeps_pages_whose_ocr_text_was_empty(vision, tmp_path):
    legacy_entry(
        tmp_path,
        "--- Page 1 ---\n\n[OCR failed for page 1]\n\n--- Page 3 ---\n\nold page three",
        failed=[1],
    )

    result = extract()

    assert vision.sent == [1]
    assert result.text == "--- Page 1 ---\n\ntext of page 1\n\n--- Page 3 ---\n\nold page three"


@pytest.mark.parametrize(
    "text",
    [
        "--- Page 2 ---\n\n[OCR failed for page 2]\n\n--- Page 1 ---\n\nout of order",
        "preamble\n\n--- Page 1 ---\n\none\n\n--- Page 2 ---\n\n[OCR failed for page 2]",
        "--- Page 1 ---\n\none\n\n--- Page 2 ---\n\nnot the placeholder",
    ],
)
def test_unrecognized_legacy_entry_is_ocred_again_in_full(vision, tmp_path, text):
    legacy_entry(tmp_path, text, failed=[2])

    result = extract()

    assert sorted(vision.sent) == [1, 2, 3]
    assert result.text.count("text of page") == 3


def test_legacy_all_failed_entry_is_ocred_again(vision, tmp_path):
    legacy_entry(
        tmp_path,
        "--- Page 1 ---\n\n[OCR failed for page 1]\n\n"
        "--- Page 2 ---\n\n[OCR failed for page 2]\n\n"
        "--- Page 3 ---\n\n[OCR failed for page 3]",
        failed=[1, 2, 3],
    )

    result = extract()

    assert sorted(vision.sent) == [1, 2, 3]
    assert "[OCR failed for page" not in result.text
    assert json.loads(full_cache(tmp_path).read_text())["text"] == result.text


def test_complete_legacy_entry_is_still_a_cache_hit(vision, tmp_path):
    legacy_entry(tmp_path, "--- Page 1 ---\n\ndone", failed=[])

    assert extract().text == "--- Page 1 ---\n\ndone"
    assert vision.sent == []


def test_without_a_cache_one_pass_keeps_failed_page_placeholders(vision, monkeypatch):
    monkeypatch.setattr(ocr_mod.settings, "ocr_cache_enabled", False)
    vision.fail(2, timeout(2))

    result = extract()

    assert result.metadata["ocr_failed_pages"] == [2]
    assert "[OCR failed for page 2]" in result.text


def test_extractor_does_not_settle_for_thin_text_while_pages_can_be_retried(monkeypatch):
    thin = ExtractedContent(text="a little text", title=None, page_count=3, word_count=3)
    monkeypatch.setattr(extractor_mod, "extract_pdf", lambda path: thin)
    monkeypatch.setattr(extractor_mod.settings, "ocr_vision_url", "http://vision.invalid")

    def incomplete(path):
        raise IncompleteOCRError(path.name, [2], 3)

    monkeypatch.setattr(ocr_mod, "extract_scanned_pdf_sync", incomplete)

    with pytest.raises(IncompleteOCRError):
        ContentExtractor()._extract_pdf(PDF)


def respond(monkeypatch, client: VisionOCRClient, handler) -> None:
    async def get_client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(client, "_get_client", get_client)


@pytest.mark.parametrize("status", [408, 429, 502, 503, 504])
def test_unavailable_http_statuses_are_endpoint_outages(monkeypatch, status):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")
    respond(monkeypatch, client, lambda request: httpx.Response(status, text="busy"))

    with pytest.raises(VisionOCRUnavailableError):
        asyncio.run(client.ocr_image(b"img"))


@pytest.mark.parametrize("status", [400, 413, 500])
def test_other_http_errors_count_against_the_page(monkeypatch, status):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")
    respond(monkeypatch, client, lambda request: httpx.Response(status, text="bad"))

    with pytest.raises(VisionOCRError) as raised:
        asyncio.run(client.ocr_image(b"img"))
    assert not isinstance(raised.value, VisionOCRUnavailableError)


def test_connection_failure_and_open_breaker_are_endpoint_outages(monkeypatch):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    respond(monkeypatch, client, refuse)
    for _ in range(VisionOCRClient._max_failures):
        with pytest.raises(VisionOCRUnavailableError):
            asyncio.run(client.ocr_image(b"img"))

    with pytest.raises(VisionOCRUnavailableError, match="circuit breaker open"):
        asyncio.run(client.ocr_image(b"img"))


@pytest.mark.parametrize(
    "error", [httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout, httpx.WriteError]
)
def test_failures_before_the_request_is_sent_are_endpoint_outages(monkeypatch, error):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")

    def unreachable(request):
        raise error("not sent", request=request)

    respond(monkeypatch, client, unreachable)

    with pytest.raises(VisionOCRUnavailableError):
        asyncio.run(client.ocr_image(b"img"))


# The endpoint had the whole request, and a page that keeps crashing or
# stalling it must still run out of attempts.
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError])
def test_failures_after_the_request_is_sent_count_against_the_page(monkeypatch, error):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")

    def lost_response(request):
        raise error("no response", request=request)

    respond(monkeypatch, client, lost_response)

    with pytest.raises(VisionOCRError) as raised:
        asyncio.run(client.ocr_image(b"img"))
    assert not isinstance(raised.value, VisionOCRUnavailableError)


def test_render_pages_maps_each_requested_page_to_its_image(monkeypatch):
    calls = []

    class FakePdf2image:
        @staticmethod
        def convert_from_path(path, dpi, first_page, last_page, fmt):
            calls.append((first_page, last_page))
            return [f"image {page}" for page in range(first_page, last_page + 1)]

    monkeypatch.setattr(ocr_mod, "_pdf2image", lambda: FakePdf2image)
    monkeypatch.setattr(ocr_mod, "_prepare_image", lambda img, *args: img.encode())

    images = ocr_mod._render_pages(PDF, [9, 1, 2, 5, 6, 7], batch_size=2)

    assert images == {page: f"image {page}".encode() for page in (1, 2, 5, 6, 7, 9)}
    assert calls == [(1, 2), (5, 6), (7, 7), (9, 9)]


def test_render_pages_matches_a_short_batch_page_by_page(monkeypatch):
    calls = []

    class FakePdf2image:
        @staticmethod
        def convert_from_path(path, dpi, first_page, last_page, fmt):
            calls.append((first_page, last_page))
            # Poppler skips page 2, so a batch comes back one image short.
            return [f"image {page}" for page in range(first_page, last_page + 1) if page != 2]

    monkeypatch.setattr(ocr_mod, "_pdf2image", lambda: FakePdf2image)
    monkeypatch.setattr(ocr_mod, "_prepare_image", lambda img, *args: img.encode())

    images = ocr_mod._render_pages(PDF, [1, 2, 3, 4], batch_size=4)

    assert images == {page: f"image {page}".encode() for page in (1, 3, 4)}
    assert calls == [(1, 4), (1, 1), (2, 2), (3, 3), (4, 4)]


@pytest.mark.parametrize(("pages", "expected"), [(3, 3), (250, 200)])
def test_count_pages_caps_at_the_configured_maximum(monkeypatch, pages, expected):
    class FakePdf2image:
        @staticmethod
        def pdfinfo_from_path(path):
            return {"Pages": pages}

    monkeypatch.setattr(ocr_mod, "_pdf2image", lambda: FakePdf2image)
    monkeypatch.setattr(ocr_mod.settings, "ocr_max_pages", 200)

    assert ocr_mod._count_pages(PDF) == expected


def test_count_pages_rejects_an_empty_pdf(monkeypatch):
    class FakePdf2image:
        @staticmethod
        def pdfinfo_from_path(path):
            return {"Pages": 0}

    monkeypatch.setattr(ocr_mod, "_pdf2image", lambda: FakePdf2image)

    with pytest.raises(ExtractionError, match="No pages found"):
        ocr_mod._count_pages(PDF)
