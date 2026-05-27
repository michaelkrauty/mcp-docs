# mcp-docs

MCP server for document management with multi-format extraction, semantic search, and source integrity tracking.

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
pip install git+https://github.com/michaelkrauty/vector-core.git@v1.0.5
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

- **Multi-Format Extraction**: PDF, DOCX, PPTX, XLSX, CSV, EPUB, XML, TXT, Markdown, HTML, RTF
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
| CSV | `.csv` | Markdown table representation |
| EPUB | `.epub` | E-book text extraction via MarkItDown |
| XML | `.xml` | XML content extraction via MarkItDown |
| Text | `.txt`, `.md` | Direct text / markdown with title extraction |
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
- markitdown (unified text conversion for DOCX, PPTX, XLSX, CSV, EPUB, XML, HTML, TXT)
- striprtf (RTF extraction)
