"""Tests for doc_type filter validation and index_document error shape."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from vector_core.errors import ErrorCode, is_error_response

from mcp_docs.models import ExtractionStatus
from mcp_docs.server import (
    index_document,
    keyword_search,
    list_documents,
    search_documents,
)
from mcp_docs.tools._validation import validate_doc_type


class TestValidateDocType:
    """Unit tests for the shared doc_type validator/normalizer."""

    def test_none_is_no_filter(self):
        assert validate_doc_type(None) == (None, None)

    def test_empty_string_is_no_filter(self):
        assert validate_doc_type("") == (None, None)

    def test_valid_value_passes_through(self):
        value, err = validate_doc_type("pdf")
        assert value == "pdf"
        assert err is None

    def test_uppercase_is_normalized(self):
        # A wrong-case type used to silently match zero rows; now it's accepted
        # and normalized to the canonical stored value.
        value, err = validate_doc_type("PDF")
        assert value == "pdf"
        assert err is None

    def test_invalid_value_returns_error(self):
        value, err = validate_doc_type("pyhton")
        assert value is None
        assert is_error_response(err)
        assert err["error_code"] == ErrorCode.INVALID_INPUT.value
        assert "pyhton" in err["message"]  # names the offending value
        assert "pdf" in err["message"]  # lists the valid values

    def test_unknown_is_a_valid_type(self):
        # "unknown" is a real stored DocumentType (for unrecognized extensions).
        value, err = validate_doc_type("unknown")
        assert value == "unknown"
        assert err is None

    def test_whitespace_only_is_no_filter(self):
        assert validate_doc_type("   ") == (None, None)

    def test_surrounding_whitespace_is_stripped(self):
        value, err = validate_doc_type("  PDF ")
        assert value == "pdf"
        assert err is None


class TestToolsRejectInvalidDocType:
    """Each tool fails fast on an invalid doc_type instead of silently
    querying for a type that matches nothing."""

    async def test_list_documents_rejects_invalid_doc_type(self):
        store = MagicMock()
        with patch("mcp_docs.tools.documents.get_document_store", return_value=store):
            result = await list_documents(doc_type="not-a-type")
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value
        store.list_summaries.assert_not_called()

    async def test_search_documents_rejects_invalid_doc_type(self):
        with patch(
            "mcp_docs.tools.search.get_search_engine", new_callable=AsyncMock
        ) as get_engine:
            result = await search_documents(query="q", doc_type="not-a-type")
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value
        get_engine.assert_not_called()  # validation runs before the engine

    async def test_keyword_search_rejects_invalid_doc_type(self):
        with patch(
            "mcp_docs.tools.search.get_search_engine", new_callable=AsyncMock
        ) as get_engine:
            result = await keyword_search(keyword="q", doc_type="not-a-type")
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value
        get_engine.assert_not_called()


class TestListDocumentsExtractionStatusValidation:
    """The invalid-extraction_status error lists every valid value, derived
    from the enum so it cannot drift as new statuses are added."""

    async def test_invalid_status_error_lists_all_enum_values(self):
        store = MagicMock()
        with patch("mcp_docs.tools.documents.get_document_store", return_value=store):
            result = await list_documents(extraction_status="bogus")
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value
        for status in ExtractionStatus:
            assert status.value in result["message"], status.value
        store.list_summaries.assert_not_called()

    async def test_cancelled_status_is_accepted(self):
        store = MagicMock()
        store.list_summaries.return_value = []
        with patch("mcp_docs.tools.documents.get_document_store", return_value=store):
            result = await list_documents(extraction_status="cancelled")
        assert not is_error_response(result)
        assert (
            store.list_summaries.call_args.kwargs["extraction_status"]
            == ExtractionStatus.CANCELLED
        )

    async def test_search_documents_passes_normalized_doc_type_to_engine(self):
        # A valid but wrong-case doc_type must reach the engine normalized.
        engine = AsyncMock()
        engine.search.return_value = []
        with patch(
            "mcp_docs.tools.search.get_search_engine", new=AsyncMock(return_value=engine)
        ):
            result = await search_documents(query="q", doc_type="PDF")
        assert result == []
        assert engine.search.call_args.kwargs["doc_type"] == "pdf"


class TestIndexDocumentErrorShape:
    """index_document returns a structured error_response (not a bare {"error": ...}
    dict) when the document is not in an indexable state, so callers using
    is_error_response() correctly detect the failure."""

    async def test_not_ready_returns_conflict_error_response(self):
        doc_id = uuid4()
        document = MagicMock()
        document.extraction_status = ExtractionStatus.QUEUED  # not extracted/indexed
        store = MagicMock()
        store.read.return_value = document
        with patch("mcp_docs.tools.indexing.get_document_store", return_value=store):
            result = await index_document(str(doc_id))
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.CONFLICT.value
