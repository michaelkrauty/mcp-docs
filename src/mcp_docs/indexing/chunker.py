"""Lossless, source-coordinate document chunking.

Character limits guide readable passages; the embedding client's exact token
splitter enforces the model budget before indexing.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass
from uuid import UUID

from mcp_docs.models import DocumentChunk
from mcp_docs.settings import settings

_HEADINGS = re.compile(r"^(?:# (.+)|([A-Z][A-Z0-9 ]{2,})\n[-=]+)", re.MULTILINE)


@dataclass
class ChunkingResult:
    """Source passages and the preferred boundary strategy."""

    chunks: list[DocumentChunk]
    strategy: str


class DocumentChunker:
    """Prefer section/paragraph boundaries, hard-splitting oversized units."""

    def __init__(
        self,
        max_chars: int | None = None,
        min_chars: int = 2000,
        overlap_chars: int | None = None,
    ):
        self.max_chars = settings.max_chunk_chars if max_chars is None else max_chars
        self.min_chars = min_chars
        self.overlap_chars = (
            settings.chunk_overlap_chars if overlap_chars is None else overlap_chars
        )
        if self.max_chars <= 0 or self.overlap_chars < 0:
            raise ValueError("Chunk size must be positive and overlap nonnegative")

    def chunk(self, document_id: UUID, text: str, page_count: int | None = None) -> ChunkingResult:
        if not text.strip():
            return ChunkingResult([], "single")

        headings = list(_HEADINGS.finditer(text))
        heading_starts = [match.start() for match in headings]
        strategy = (
            "single"
            if len(text) <= self.max_chars
            else ("sections" if len(headings) > 1 else "paragraphs")
        )
        boundaries = (
            heading_starts
            if strategy == "sections"
            else [match.end() for match in re.finditer(r"\n\s*\n", text)]
        )
        chunks: list[DocumentChunk] = []
        start = 0
        covered = 0
        overlap = min(self.overlap_chars, self.max_chars // 4)
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            if end < len(text):
                boundary = bisect_right(boundaries, end) - 1
                if boundary >= 0 and boundaries[boundary] > max(start, covered):
                    end = boundaries[boundary]
            heading = bisect_right(heading_starts, start) - 1
            title = (
                headings[heading].group(1) or headings[heading].group(2) if heading >= 0 else None
            )
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    content=text[start:end],
                    page_start=1 if strategy == "single" and page_count else None,
                    page_end=page_count if strategy == "single" else None,
                    section_title=title,
                    char_start=start,
                    char_end=end,
                )
            )
            covered = end
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
        return ChunkingResult(chunks, strategy)


def chunk_document(
    document_id: UUID, text: str, page_count: int | None = None
) -> list[DocumentChunk]:
    """Return source-coordinate passages, including every heading and tail."""
    return DocumentChunker().chunk(document_id, text, page_count).chunks
