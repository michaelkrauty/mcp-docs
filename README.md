# mcp-docs

MCP server for document management with multi-format extraction, semantic search, and source integrity tracking.

Supports stateless MCP `2026-07-28` requests and legacy MCP clients from the same stdio server through the official Python SDK v2.

## Prerequisites

- **Python 3.12+**
- **Linux or macOS** (uses POSIX file locking via vector-core; not compatible with Windows)
- [Qdrant](https://qdrant.tech/) vector database (default: `localhost:6333`)
- An OpenAI-compatible embedding API (e.g., llama.cpp, Ollama, or any `/v1/embeddings` endpoint; default: `localhost:8080`)
- **Vision endpoint** *(optional)* — OpenAI-compatible vision model for automatic OCR fallback on scanned PDFs
- **poppler** *(optional, for OCR)* — `apt install poppler-utils` (Linux) or `brew install poppler` (macOS)

## Installation

Requires [vector-core](https://github.com/michaelkrauty/vector-core).

```bash
pip install git+https://github.com/michaelkrauty/vector-core.git@v1.8.1
pip install git+https://github.com/michaelkrauty/mcp-docs.git
```

Or clone both repos and install locally:

```bash
git clone https://github.com/michaelkrauty/vector-core.git
git clone https://github.com/michaelkrauty/mcp-docs.git
pip install -e vector-core/
pip install -e mcp-docs/
```

## Quick Start

```bash
# Register with Claude Code (set env vars to match your setup):
claude mcp add docs \
  -e VECTOR_QDRANT_URL=http://localhost:6333 \
  -e VECTOR_EMBEDDING_URL=http://localhost:8080 \
  -e VECTOR_EMBEDDING_MODEL=your-model-name \
  -e VECTOR_COLLECTION_NAME=my-documents \
  -- mcp-docs

# Or add to your MCP client config (e.g., claude_desktop_config.json):
# {
#   "mcpServers": {
#     "docs": {
#       "command": "mcp-docs",
#       "env": {
#         "VECTOR_QDRANT_URL": "http://localhost:6333",
#         "VECTOR_EMBEDDING_URL": "http://localhost:8080",
#         "VECTOR_EMBEDDING_MODEL": "your-model-name",
#         "VECTOR_COLLECTION_NAME": "my-documents"
#       }
#     }
#   }
# }
```

## Features

- **Multi-Format Extraction**: PDF, DOCX, PPTX, XLSX, CSV, EPUB, XML, TXT, Markdown, Jupyter (`.ipynb`), HTML, RTF
- **Directory Scanning**: Register root directories for automatic discovery
- **Semantic Search**: Hybrid dense + sparse vector search via Qdrant
- **Keyword Search**: Exact keyword/phrase matching in filenames and content
- **Hash Deduplication**: SHA-256 content hashes prevent duplicate ingestion
- **Source Tracking**: Verify document references for fact integrity
- **Background Processing**: Async extraction/indexing with worker queue
- **Filesystem Operations**: Move files/directories with automatic registry updates
- **Glossary**: Shared term definitions (same store as mcp-notes)

## Tools (36 total)

### Documents (6)
| Tool | Description |
|------|-------------|
| `register_document` | Register file for indexing with deduplication |
| `get_document` | Retrieve document by UUID |
| `get_document_by_hash` | Lookup by SHA-256 content hash |
| `update_document_tags` | Modify document tags |
| `delete_document` | Remove from registry |
| `list_documents` | List with tag/status/type/root filters |

### Processing (4)
| Tool | Description |
|------|-------------|
| `get_processing_status` | Check extraction/indexing progress |
| `list_queued_documents` | View processing queue |
| `wait_for_document` | Block until processing completes |
| `cancel_processing` | Stop processing for a document |

### Search (4)
| Tool | Description |
|------|-------------|
| `search_documents` | Hybrid semantic search with filters |
| `keyword_search` | Exact keyword/phrase matching in filenames and content |
| `find_similar_documents` | Content-based similarity matching |
| `get_document_chunks` | Retrieve indexed chunks |

### Indexing (2)
| Tool | Description |
|------|-------------|
| `index_document` | Index single document |
| `index_all_documents` | Batch index with two-pass vocabulary |

### Root Management (6)
| Tool | Description |
|------|-------------|
| `add_document_root` | Register directory for scanning |
| `list_document_roots` | View all roots |
| `get_document_root` | Info on specific root |
| `remove_document_root` | Unregister a root |
| `scan_document_root` | Scan specific root for changes |
| `scan_all_roots` | Scan all enabled roots |

### Hash Verification (3)
| Tool | Description |
|------|-------------|
| `lookup_hash` | Find document by SHA-256 |
| `verify_document_reference` | Check document exists and unchanged |
| `batch_verify_references` | Verify multiple hashes |

### Glossary (6)
| Tool | Description |
|------|-------------|
| `add_glossary_entry` | Add term with expansion, definition, domain |
| `lookup_term` | Exact lookup by term or alias |
| `search_glossary` | Semantic glossary search |
| `list_glossary` | List entries with optional domain filter |
| `update_glossary_entry` | Modify entry metadata |
| `delete_glossary_entry` | Delete entry |

### Filesystem (5)
| Tool | Description |
|------|-------------|
| `move_file` | Move a file and update document registry |
| `create_directory` | Create a directory within a document root |
| `rename_directory` | Rename a directory and update all document paths |
| `move_directory` | Move a directory and update all document paths |
| `delete_directory` | Delete an empty directory |

## Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Text extraction via MarkItDown; automatic OCR fallback via vision LLM for scanned/image-based PDFs |
| Word | `.docx` | Full text via MarkItDown + metadata via python-docx |
| Word (legacy) | `.doc` | RTF-disguised files only; true DOC requires conversion |
| PowerPoint | `.pptx` | Slide text via MarkItDown + metadata via python-pptx |
| PowerPoint (legacy) | `.ppt` | Best-effort via MarkItDown; may require conversion |
| Excel | `.xlsx`, `.xls` | Spreadsheet to markdown table via MarkItDown |
| CSV | `.csv` | Markdown table via the csv module, with encoding fallback (utf-8-sig, utf-8, cp1252, latin-1) for non-ASCII exports |
| EPUB | `.epub` | E-book text extraction via MarkItDown |
| XML | `.xml` | XML content extraction via MarkItDown |
| Text | `.txt`, `.md` | Direct text / markdown with title extraction |
| Jupyter | `.ipynb` | Markdown cells as prose, code cells as language-tagged fenced blocks; outputs and raw cells skipped; title from first H1 |
| HTML | `.html`, `.htm` | Markdown conversion via MarkItDown |
| RTF | `.rtf` | Rich text via striprtf |
| OpenDocument | `.odt` | Not supported; raises error advising conversion to DOCX |

## Document Status

| Status | Meaning |
|--------|---------|
| `Active` | Path exists, hash matches |
| `Modified` | Path exists, different hash |
| `Relocated` | Found at different path (same hash) |
| `Deleted` | File not found anywhere |

## Extraction Pipeline

```
Queued → Processing → Extracted → Indexed
                  ↘ Failed (with error message)
```

## Data Model

### Document
```python
id: UUID
path: "/path/to/document.pdf"
content_hash: "sha256:..."
doc_type: "pdf"
status: "active"
extraction_status: "indexed"
tags: ["research", "2024"]
created_at: datetime
indexed_at: datetime
```

### DocumentRoot
```python
path: "/home/user/documents"
name: "My Documents"
recursive: True
enabled: True
added_at: datetime
last_scanned: datetime
file_count: 42
```

## Storage

| Data | Location |
|------|----------|
| Document registry | `documents.db` in data dir |
| Extracted content | Qdrant collection (set via `VECTOR_COLLECTION_NAME`) |
| Glossary | `glossary.db` in vector-core's shared data dir |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_COLLECTION_NAME` | (required) | Qdrant collection name |
| `DOCS_OCR_VISION_URL` | `""` | OpenAI-compatible vision endpoint for OCR (empty = OCR disabled) |
| `DOCS_OCR_VISION_MODEL` | `""` | Vision model name (empty = let endpoint decide) |
| `DOCS_OCR_DPI` | `300` | DPI for PDF page rendering |
| `DOCS_OCR_TIMEOUT` | `180` | Per-page OCR timeout in seconds |
| `DOCS_OCR_MAX_PAGES` | `200` | Maximum pages to OCR per document |
| `DOCS_OCR_IMAGE_MAX_DIMENSION` | `1536` | Max image width/height sent to vision model |
| `DOCS_OCR_IMAGE_FORMAT` | `jpeg` | Image format: `jpeg` (smaller) or `png` (lossless) |
| `DOCS_OCR_JPEG_QUALITY` | `90` | JPEG quality (1-100) if using jpeg format |
| `DOCS_OCR_CACHE_ENABLED` | `true` | Cache OCR results by file metadata |
| `DOCS_OCR_CONCURRENCY` | `4` | Max concurrent OCR page requests |
| `DOCS_MAX_CHUNK_CHARS` | `80000` | Chunk size (~20k tokens) |
| `DOCS_CHUNK_OVERLAP_CHARS` | `500` | Overlap between chunks |
| `DOCS_MAX_WORKERS` | `2` | Background processing workers |
| `DOCS_MAX_TAGS_PER_DOCUMENT` | `20` | Tag limit |
| `DOCS_MAX_TAG_LENGTH` | `50` | Maximum length of a single tag |

Plus inherited vector-core settings (`VECTOR_QDRANT_URL`, `VECTOR_EMBEDDING_URL`, etc.).

### Changing embedding models

Change the vector-core embedding configuration and restart the server. The first index or search operation builds a compatible physical collection from persisted embedding text before using it. Changing the model, endpoint, dimension or deployment namespace triggers migration, including same-dimension model changes. Set a new `VECTOR_EMBEDDING_CACHE_NAMESPACE` when an unchanged model alias serves different weights or behavior; the embeddings protocol cannot detect that change automatically.

Migration preserves document IDs, metadata, processing states, hashes and canonical sparse vectors. Existing chunks are re-embedded from retained Qdrant content, so unavailable source files do not prevent migration. Oversized inputs gain searchable fragments with exact source spans; the original payload is retained once rather than copied into every fragment. Shared glossary entries use their full retained definitions. Incomplete migrations do not replace the active generation, and previous collections remain available for recovery. Updated writers serialize with migration; restart all clients sharing a collection when changing configuration.

### Content coverage and result semantics

Document passages preserve exact extracted-text slices, including headings and oversized section or paragraph tails. Every retained source span is embedded within the configured model's exact token budget; request batching handles transport limits separately. Search results show the matching passage and source character offsets when available. Fragment results also expose `embedding_span` and `evidence_span`, relative to their canonical retained chunk. A canonical sparse match can describe text beyond its first dense span: `evidence_kind="keyword_excerpt"` explicitly identifies a query-matching excerpt from that retained text, without claiming it is the dense-vector span. `get_document_chunks` returns complete canonical retained chunks, excluding derived search fragments. Character offsets refer to extracted text, not bytes or positions in the original PDF or office file; older retained chunks may not have original-document offsets.

`search_documents(include_chunks=False)` searches document content and groups matches into one result per document. The default passage mode can return several matches from the same document. Filename, title and tag summaries provide auxiliary metadata matches in both modes. `find_similar_documents` compares every indexed source passage against other documents' passages in bounded concurrent batches and ranks each document by its best passage-pair similarity, excluding the source document before retrieval and fetching full payloads only for final winners. Query failures cancel outstanding work and fail the request rather than return partial results. It does not replace body content with metadata summaries or average away distinctive tails.

Coverage applies to retained or newly extracted text. Migration cannot recreate text omitted by an earlier extractor or chunker when the source file is unavailable. Incremental source-backed indexing repairs old document layouts once when originals are available; matching current-layout hashes skip subsequent extraction. Repair verifies an indexed file's registered hash before and after extraction and before publication; changed files require a rescan and keep their retained index. Completion markers are published only after obsolete-point cleanup succeeds, and metadata-only updates do not certify body repair. Missing originals are listed in `unavailable_sources` with document IDs and paths, and their retained index is left unchanged. OCR page limits and extraction failures can leave unextracted source content; notebook outputs are not part of the extracted text. Legacy note metadata in a shared collection is not a recovered note body. Force reindexing still requires the original files; embedding migration uses retained text instead.

Install the optional `tokenizer` extra for model-token-aware input budgeting (`uv sync --extra tokenizer` in a checkout). Configure vector-core with a local tokenizer file, the model's input-token limit and the serving backend's special-token policy. Role prefixes count toward that limit. Tokenization uses the local file; embedding calls do not download tokenizer artifacts. No model-token limit is inferred without this configuration. Explicit character or byte limits can also bound source fragments, and backend input rejections are reported rather than silently truncating content.

## Integration with mcp-notes

- **Shared glossary**: Same `glossary.db`, same terms
- **Shared facts.db**: Documents can be sources for facts
- **Hash verification**: mcp-docs verifies document sources haven't changed

When a fact references a document:
1. Source stores `source_type: "document"`, `source_hash: "sha256:..."`
2. `verify_document_reference` checks if hash still exists
3. If file modified/deleted, fact marked as having stale source

## Dependencies

Requires vector-core components:
- EmbeddingClient, GlobalVocabulary (search)
- QdrantStorage, HybridSearcher (storage)
- GlossaryStore (glossary)
- SourceIntegrityManager (fact verification)

External libraries:
- pypdf (PDF extraction)
- python-docx (DOCX metadata)
- python-pptx (PPTX metadata)
- markitdown (unified text conversion for DOCX, PPTX, XLSX, EPUB, XML, HTML, TXT)
- Python standard-library csv module (CSV tables, with encoding fallback across utf-8-sig, utf-8, cp1252, and latin-1)
- striprtf (RTF extraction)
