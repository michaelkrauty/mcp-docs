"""Jupyter notebook (.ipynb) content extraction."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_docs.models import ExtractedContent, ExtractionError


def _cell_source(source: object) -> str:
    """Normalize a notebook cell's ``source`` (a list of lines or a single string)."""
    if isinstance(source, list):
        return "".join(part for part in source if isinstance(part, str))
    if isinstance(source, str):
        return source
    return ""


def _first_h1(markdown: str) -> str | None:
    """Return the text of the first level-1 Markdown heading, if any."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return None


def _notebook_language(notebook: dict) -> str:
    """Best-effort source language for code fences, read from notebook metadata."""
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    language_info = metadata.get("language_info")
    if isinstance(language_info, dict):
        name = language_info.get("name")
        if isinstance(name, str):
            return name
    kernelspec = metadata.get("kernelspec")
    if isinstance(kernelspec, dict):
        language = kernelspec.get("language")
        if isinstance(language, str):
            return language
    return ""


def extract_ipynb(path: Path) -> ExtractedContent:
    """Extract searchable text from a Jupyter notebook (``.ipynb``).

    Markdown cells are kept as prose and code cells as fenced code blocks (tagged
    with the notebook language when known). Cell outputs and raw cells are skipped:
    outputs are frequently large and non-textual (images, long stdout) and only add
    noise to search. The title, if present, is taken from the first level-1 Markdown
    heading.

    Args:
        path: Path to the .ipynb file

    Returns:
        ExtractedContent with the concatenated cell text and word count

    Raises:
        ExtractionError: If the file cannot be read or is not a valid notebook
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ExtractionError(f"Failed to read notebook {path.name}: {e}") from e

    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Invalid notebook JSON in {path.name}: {e}") from e

    if not isinstance(notebook, dict):
        raise ExtractionError(f"Notebook {path.name} is not a JSON object")

    language = _notebook_language(notebook)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        cells = []

    parts: list[str] = []
    title: str | None = None
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = _cell_source(cell.get("source"))
        if not source.strip():
            continue
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            parts.append(source)
            if title is None:
                title = _first_h1(source)
        elif cell_type == "code":
            parts.append(f"```{language}\n{source}\n```")
        # raw and any other cell types are intentionally skipped

    text = "\n\n".join(parts)
    word_count = len(text.split()) if text else 0
    metadata: dict[str, object] = {"cell_count": len(cells)}
    if language:
        metadata["language"] = language

    return ExtractedContent(
        text=text,
        title=title,
        page_count=None,
        word_count=word_count,
        metadata=metadata,
    )
