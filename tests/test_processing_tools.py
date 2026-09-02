"""Tests for document processing tools."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from vector_core.errors import ErrorCode

from mcp_docs.models import ExtractionStatus
from mcp_docs.processing import ProcessingResult, ProcessingStatus
from mcp_docs.tools import processing as processing_mod


@pytest.mark.asyncio
async def test_unknown_document_returns_not_found_without_waiting() -> None:
    """A valid unknown UUID must not create a processor waiter."""
    document_id = uuid4()
    store = MagicMock()
    store.read.return_value = None
    processor = MagicMock()
    processor.wait_for = AsyncMock()
    get_processor = AsyncMock(return_value=processor)

    with (
        patch.object(processing_mod, "get_document_store", return_value=store),
        patch.object(processing_mod, "get_document_processor", get_processor),
    ):
        result = await processing_mod.wait_for_document(str(document_id), timeout=3600.0)

    assert result["error_code"] == ErrorCode.NOT_FOUND.value
    assert str(document_id) in result["message"]
    store.read.assert_called_once_with(document_id)
    get_processor.assert_not_awaited()
    processor.wait_for.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extraction_status",
    [
        ExtractionStatus.QUEUED,
        ExtractionStatus.PROCESSING,
        ExtractionStatus.INDEXED,
    ],
)
async def test_existing_document_still_delegates_to_processor(
    extraction_status: ExtractionStatus,
) -> None:
    """Queued, processing, and terminal documents retain the wait path."""
    document_id = uuid4()
    document = MagicMock(extraction_status=extraction_status)
    store = MagicMock()
    store.read.return_value = document
    now = datetime.now(UTC)
    result = ProcessingResult(
        document_id=document_id,
        status=ProcessingStatus.COMPLETED,
        started_at=now,
        completed_at=now,
    )
    processor = MagicMock()
    processor.wait_for = AsyncMock(return_value=result)
    get_processor = AsyncMock(return_value=processor)

    with (
        patch.object(processing_mod, "get_document_store", return_value=store),
        patch.object(processing_mod, "get_document_processor", get_processor),
    ):
        response = await processing_mod.wait_for_document(str(document_id), timeout=123.0)

    assert response == result.to_dict()
    store.read.assert_called_once_with(document_id)
    get_processor.assert_awaited_once_with()
    processor.wait_for.assert_awaited_once_with(document_id, timeout=123.0)
