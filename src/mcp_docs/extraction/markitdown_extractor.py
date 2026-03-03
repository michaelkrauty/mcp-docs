"""Thin singleton wrapper around MarkItDown for unified document conversion."""

import functools
from pathlib import Path

from markitdown import MarkItDown


@functools.lru_cache(maxsize=1)
def _get_md() -> MarkItDown:
    return MarkItDown(enable_plugins=False)


def extract_text_markitdown(path: Path) -> str:
    """Convert a document to markdown text using MarkItDown."""
    result = _get_md().convert(path)
    return result.markdown or ""
