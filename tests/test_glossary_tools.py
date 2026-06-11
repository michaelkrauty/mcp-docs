"""Tests for glossary tool wrappers.

vector-core >= 1.2.1 rejects blank domain values in
GlossaryToolHelper.update_entry, but update_glossary_entry documents
passing "" to clear the domain. The wrapper must translate blank
strings to None (the helper's "clear" value) to keep that contract.
"""

from unittest.mock import AsyncMock, patch

import pytest
from vector_core import UNSET

from mcp_docs.tools.glossary import update_glossary_entry


def _mock_helper() -> AsyncMock:
    helper = AsyncMock()
    helper.update_entry = AsyncMock(return_value={"id": "x"})
    return helper


class TestUpdateGlossaryEntryDomainClear:
    @pytest.mark.asyncio
    async def test_empty_string_domain_is_translated_to_none(self) -> None:
        helper = _mock_helper()
        with patch(
            "mcp_docs.tools.glossary.get_glossary_helper",
            AsyncMock(return_value=helper),
        ):
            await update_glossary_entry("USAF", domain="")

        args = helper.update_entry.call_args.args
        assert args[4] is None  # domain positional arg

    @pytest.mark.asyncio
    async def test_whitespace_domain_is_translated_to_none(self) -> None:
        helper = _mock_helper()
        with patch(
            "mcp_docs.tools.glossary.get_glossary_helper",
            AsyncMock(return_value=helper),
        ):
            await update_glossary_entry("USAF", domain="   ")

        args = helper.update_entry.call_args.args
        assert args[4] is None

    @pytest.mark.asyncio
    async def test_unset_domain_is_passed_through(self) -> None:
        helper = _mock_helper()
        with patch(
            "mcp_docs.tools.glossary.get_glossary_helper",
            AsyncMock(return_value=helper),
        ):
            await update_glossary_entry("USAF", definition="new def")

        args = helper.update_entry.call_args.args
        assert args[4] is UNSET

    @pytest.mark.asyncio
    async def test_real_domain_value_is_passed_through(self) -> None:
        helper = _mock_helper()
        with patch(
            "mcp_docs.tools.glossary.get_glossary_helper",
            AsyncMock(return_value=helper),
        ):
            await update_glossary_entry("USAF", domain="military")

        args = helper.update_entry.call_args.args
        assert args[4] == "military"
