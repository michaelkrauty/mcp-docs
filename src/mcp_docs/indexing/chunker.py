"""Document chunking for indexing.

Philosophy: Leverage the full context window of Qwen3-Embedding-8B (32k tokens).
Most documents fit in a single chunk. Split only when necessary.

Parameters:
- chunk_max_chars: 80,000 chars (~20k tokens)
- chunk_min_chars: 2,000 chars
- chunk_overlap_chars: 500 chars
"""

import re
from dataclasses import dataclass
from uuid import UUID

from mcp_docs.models import DocumentChunk
from mcp_docs.settings import settings


@dataclass
class ChunkingResult:
    """Result of chunking a document."""

    chunks: list[DocumentChunk]
    strategy: str  # "single", "sections", "pages", "paragraphs"


class DocumentChunker:
    """
    Chunks documents for indexing.

    Strategy:
    1. If document fits in max_chars → single chunk
    2. If document has major sections (H1, chapters) → split at section boundaries
    3. Otherwise → split at paragraph boundaries
    """

    def __init__(
        self,
        max_chars: int | None = None,
        min_chars: int = 2000,
        overlap_chars: int | None = None,
    ):
        """
        Initialize chunker.

        Args:
            max_chars: Maximum characters per chunk (default from settings)
            min_chars: Minimum characters per chunk (merge smaller)
            overlap_chars: Overlap between chunks (default from settings)
        """
        self.max_chars = max_chars or settings.max_chunk_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars or settings.chunk_overlap_chars

    def chunk(
        self,
        document_id: UUID,
        text: str,
        page_count: int | None = None,
    ) -> ChunkingResult:
        """
        Chunk document text.

        Args:
            document_id: Document UUID
            text: Full document text
            page_count: Optional page count for page-based chunking

        Returns:
            ChunkingResult with chunks and strategy used
        """
        # Single chunk if small enough
        if len(text) <= self.max_chars:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=0,
                content=text,
                page_start=1 if page_count else None,
                page_end=page_count,
                section_title=None,
                char_start=0,
                char_end=len(text),
            )
            return ChunkingResult(chunks=[chunk], strategy="single")

        # Try section-based chunking (H1 headers)
        sections = self._split_by_sections(text)
        if len(sections) > 1:
            chunks = self._chunks_from_sections(document_id, sections)
            return ChunkingResult(chunks=chunks, strategy="sections")

        # Fall back to paragraph-based chunking
        chunks = self._chunk_by_paragraphs(document_id, text)
        return ChunkingResult(chunks=chunks, strategy="paragraphs")

    def _split_by_sections(self, text: str) -> list[tuple[str | None, str]]:
        """
        Split text by major section headers.

        Returns list of (title, content) tuples.
        """
        # Match markdown H1 or document-style headers
        section_pattern = re.compile(
            r'^(?:# (.+)|(?:^|\n\n)([A-Z][A-Z0-9 ]{2,})\n[-=]+)',
            re.MULTILINE,
        )

        sections: list[tuple[str | None, str]] = []
        last_end = 0
        last_title: str | None = None

        for match in section_pattern.finditer(text):
            # Save content before this header
            if match.start() > last_end:
                content = text[last_end : match.start()].strip()
                if content:
                    sections.append((last_title, content))

            # Get new section title
            last_title = match.group(1) or match.group(2)
            last_end = match.end()

        # Add remaining content
        if last_end < len(text):
            content = text[last_end:].strip()
            if content:
                sections.append((last_title, content))

        return sections

    def _chunks_from_sections(
        self,
        document_id: UUID,
        sections: list[tuple[str | None, str]],
    ) -> list[DocumentChunk]:
        """Create chunks from sections, merging small ones."""
        chunks: list[DocumentChunk] = []
        current_content = ""
        current_title: str | None = None
        current_start = 0
        chunk_index = 0

        for title, content in sections:
            # If adding this section exceeds max, save current and start new
            if current_content and len(current_content) + len(content) > self.max_chars:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=current_content,
                        page_start=None,
                        page_end=None,
                        section_title=current_title,
                        char_start=current_start,
                        char_end=current_start + len(current_content),
                    )
                )
                chunk_index += 1
                # Save length before resetting content for correct position tracking
                chunk_len = len(current_content)
                current_content = ""
                current_title = title
                current_start = current_start + chunk_len

            # Add section to current chunk
            if not current_content:
                current_title = title
            if title and current_title != title:
                current_content += f"\n\n## {title}\n\n"
            current_content += content

        # Save final chunk
        if current_content:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=current_content,
                    page_start=None,
                    page_end=None,
                    section_title=current_title,
                    char_start=current_start,
                    char_end=current_start + len(current_content),
                )
            )

        return chunks

    def _chunk_by_paragraphs(
        self,
        document_id: UUID,
        text: str,
    ) -> list[DocumentChunk]:
        """Chunk text by paragraphs, targeting reasonable chunk sizes."""
        # Split by paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text)

        chunks: list[DocumentChunk] = []
        current_content = ""
        current_start = 0
        chunk_index = 0
        char_offset = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_offset += 2  # Account for \n\n
                continue

            # If adding this paragraph exceeds max, save current
            if current_content and len(current_content) + len(para) + 2 > self.max_chars:
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=current_content,
                        page_start=None,
                        page_end=None,
                        section_title=None,
                        char_start=current_start,
                        char_end=current_start + len(current_content),
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_content)
                current_content = overlap_text + para if overlap_text else para
                current_start = char_offset - len(overlap_text) if overlap_text else char_offset
            else:
                # Add paragraph to current chunk
                if current_content:
                    current_content += "\n\n"
                else:
                    current_start = char_offset
                current_content += para

            char_offset += len(para) + 2  # Include \n\n separator

        # Save final chunk
        if current_content:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=current_content,
                    page_start=None,
                    page_end=None,
                    section_title=None,
                    char_start=current_start,
                    char_end=current_start + len(current_content),
                )
            )

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of a chunk."""
        if len(text) <= self.overlap_chars:
            return ""

        overlap = text[-self.overlap_chars :]

        # Try to break at word boundary
        space_idx = overlap.find(" ")
        if space_idx > 0:
            overlap = overlap[space_idx + 1 :]

        return overlap


def chunk_document(
    document_id: UUID,
    text: str,
    page_count: int | None = None,
) -> list[DocumentChunk]:
    """
    Convenience function to chunk a document.

    Args:
        document_id: Document UUID
        text: Full document text
        page_count: Optional page count

    Returns:
        List of DocumentChunk objects
    """
    chunker = DocumentChunker()
    result = chunker.chunk(document_id, text, page_count)
    return result.chunks
