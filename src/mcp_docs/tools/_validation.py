"""Shared input validation for mcp-docs tools."""

from __future__ import annotations

from vector_core.errors import ErrorCode, error_response

from mcp_docs.models import DocumentType


def validate_doc_type(doc_type: str | None) -> tuple[str | None, dict | None]:
    """Validate and normalize a ``doc_type`` filter argument.

    Returns ``(normalized, None)`` on success, where ``normalized`` is the
    canonical lowercase :class:`DocumentType` value (or ``None`` when no filter
    was supplied). Returns ``(None, error)`` — ``error`` being an
    ``error_response`` dict — when ``doc_type`` is a non-empty value that is not
    a recognized document type.

    Without this, a typo'd or wrong-case ``doc_type`` (``"pyhton"``, ``"PDF"``)
    is passed straight to the SQL/Qdrant filter, silently matching zero
    documents — indistinguishable to the caller from "no documents of this
    type." Validating up front turns that into a clear, actionable error.
    """
    if not doc_type:
        # None or empty string: treated as "no filter" by every call site.
        return None, None
    try:
        return DocumentType(doc_type.lower()).value, None
    except ValueError:
        valid = ", ".join(t.value for t in DocumentType)
        return None, error_response(
            ErrorCode.INVALID_INPUT,
            f"Invalid doc_type: {doc_type!r}. Valid values: {valid}",
        )
