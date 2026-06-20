"""Glossary tools using vector-core GlossaryToolHelper.

Tools:
- add_glossary_entry: Add a new glossary entry
- lookup_term: Exact lookup by term or alias
- search_glossary: Semantic search for glossary entries
- list_glossary: List all glossary entries
- update_glossary_entry: Update an existing glossary entry
- delete_glossary_entry: Delete a glossary entry
"""

import warnings

from pydantic.json_schema import PydanticJsonSchemaWarning
from vector_core import UNSET, UnsetType, validate_limit

from mcp_docs.app import mcp
from mcp_docs.singletons import get_glossary_helper

# Suppress warning for UNSET sentinel (intentionally non-JSON-serializable)
warnings.filterwarnings(
    "ignore",
    category=PydanticJsonSchemaWarning,
    message=".*UNSET.*",
)


@mcp.tool()
async def add_glossary_entry(
    term: str,
    expansion: str,
    definition: str,
    domain: str | None = None,
    aliases: list[str] | None = None,
) -> dict:
    """
    Add a new glossary entry.

    Args:
        term: Canonical term (e.g., "USAF")
        expansion: Full expansion (e.g., "United States Air Force")
        definition: Detailed definition
        domain: Optional category (e.g., "military", "tech", "finance")
        aliases: Optional alternative terms that point to this entry

    Returns:
        Created entry as dict
    """
    helper = await get_glossary_helper()
    return await helper.add_entry(term, expansion, definition, domain, aliases)


@mcp.tool()
async def lookup_term(term: str) -> dict:
    """
    Exact lookup by term or alias (case-insensitive).

    Args:
        term: Term to look up (e.g., "usaf", "USAF", "US Air Force")

    Returns:
        Glossary entry if found, or error dict
    """
    helper = await get_glossary_helper()
    return await helper.lookup(term)


@mcp.tool()
async def search_glossary(
    query: str,
    domain: str | None = None,
    limit: int = 10,
) -> list[dict] | dict:
    """
    Semantic search for glossary entries.

    Args:
        query: Natural language search query
        domain: Optional domain filter
        limit: Max results (default 10)

    Returns:
        List of matching entries with relevance scores, or error dict
    """
    helper = await get_glossary_helper()
    limit = validate_limit(limit)
    return await helper.search(query, domain, limit)


@mcp.tool()
async def list_glossary(
    domain: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    List all glossary entries with optional domain filter.

    Args:
        domain: Optional domain filter
        limit: Max results (default 50)

    Returns:
        List of glossary entry summaries
    """
    helper = await get_glossary_helper()
    limit = validate_limit(limit, default=50)
    return await helper.list_entries(domain, limit)


@mcp.tool()
async def update_glossary_entry(
    term_or_id: str,
    term: str | None = None,
    expansion: str | None = None,
    definition: str | None = None,
    domain: str | None | UnsetType = UNSET,
    aliases: list[str] | None | UnsetType = UNSET,
) -> dict:
    """
    Update an existing glossary entry. Only provided fields are updated.

    Args:
        term_or_id: Term (case-insensitive) or UUID to identify the entry
        term: New canonical term (optional)
        expansion: New expansion (optional)
        definition: New definition (optional)
        domain: New domain (optional, pass "" to clear)
        aliases: New aliases (optional, pass [] to clear)

    Returns:
        Updated entry as dict
    """
    helper = await get_glossary_helper()
    # vector-core >= 1.2.1 rejects blank domain values, but this tool's
    # documented way to clear the domain is "". Translate blank to None
    # (the helper's "clear" value) to keep that contract working.
    if isinstance(domain, str) and not domain.strip():
        domain = None
    return await helper.update_entry(
        term_or_id, term, expansion, definition, domain, aliases
    )


@mcp.tool()
async def delete_glossary_entry(term_or_id: str) -> dict:
    """
    Delete a glossary entry by term or UUID.

    Args:
        term_or_id: Term (case-insensitive) or UUID string

    Returns:
        Success status or error
    """
    helper = await get_glossary_helper()
    return await helper.delete_entry(term_or_id)
