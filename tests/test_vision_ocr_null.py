"""Regression tests: a vision OCR response with null/missing message content
must not crash the whole document's OCR.

An OpenAI-compatible chat-completions response can legitimately carry
``"content": null`` (a blank or empty generation). The OCR pipeline must treat
that as empty page text, not as a None that propagates into ``_merge_pages`` and
raises ``AttributeError``, aborting OCR for every page of the document.
"""

import asyncio

import pytest

from mcp_docs.extraction.ocr import _merge_pages
from mcp_docs.extraction.vision_client import VisionOCRClient


def test_merge_pages_tolerates_none_page():
    """A page whose OCR content was None must not crash the merge."""
    assert _merge_pages([None]) == ""
    merged = _merge_pages([None, "real text", None])
    assert "real text" in merged


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def post(self, *args, **kwargs):
        return _Resp(self._payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": None}}]},  # explicit null content
        {"choices": [{"message": {}}]},                  # content key absent
    ],
)
def test_ocr_image_coerces_missing_or_null_content_to_empty_string(
    monkeypatch, payload
):
    client = VisionOCRClient(base_url="http://vision.invalid", model="m")
    monkeypatch.setattr(client, "_check_circuit_breaker", lambda: None)

    async def fake_get_client():
        return _FakeClient(payload)

    monkeypatch.setattr(client, "_get_client", fake_get_client)

    result = asyncio.run(client.ocr_image(b"x", page_num=1, image_format="png"))
    assert result == ""
