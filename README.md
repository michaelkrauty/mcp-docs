# mcp-docs

MCP server for document management with multi-format extraction, semantic search, and source integrity tracking.

## Features

- **Multi-Format Extraction**: PDF, DOCX, PPTX, TXT, Markdown, HTML, RTF, ODT
- **Directory Scanning**: Register root directories for automatic discovery
- **Semantic Search**: Hybrid dense + sparse vector search via Qdrant
- **Hash Deduplication**: SHA-256 content hashes prevent duplicate ingestion
- **Source Tracking**: Verify document references for fact integrity
- **Background Processing**: Async extraction/indexing with worker queue
- **Glossary**: Shared term definitions (same store as mcp-notes)

## Tools (31 total)

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

### Search (3)
| Tool | Description |
|------|-------------|
| `search_documents` | Hybrid semantic search with filters |
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

## Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Text extraction, optional OCR for scanned |
| Word | `.docx` | Full text content |
| PowerPoint | `.pptx` | Slide text extraction |
| Text | `.txt`, `.md`, `.rst` | Direct text |
| HTML | `.html`, `.htm` | Tag stripping |
| RTF | `.rtf` | Rich text |
| OpenDocument | `.odt` | LibreOffice text |

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
                  ↘ Skipped (OCR disabled, file too large)
```

## Data Model

### Document
```python
id: UUID
path: "/path/to/document.pdf"
content_hash: "sha256:..."
document_type: "pdf"
status: "active"
extraction_status: "indexed"
tags: ["research", "2024"]
created: datetime
modified: datetime
```

### DocumentRoot
```python
id: UUID
path: "/home/user/documents"
recursive: True
enabled: True
last_scanned: datetime
```

## Storage

| Data | Location |
|------|----------|
| Document registry | `documents.db` in data dir |
| Extracted content | Qdrant collection (set via `VECTOR_COLLECTION_NAME`) |
| Glossary | `glossary.db` in `~/.cache/vector-core/` |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_COLLECTION_NAME` | (required) | Qdrant collection name |
| `DOCS_MAX_FILE_SIZE_MB` | `100` | Max file size to process |
| `DOCS_OCR_ENABLED` | `false` | Enable OCR for scanned PDFs |
| `DOCS_OCR_LANGUAGE` | `eng` | Tesseract language |
| `DOCS_MAX_CHUNK_CHARS` | `80000` | Chunk size (~20k tokens) |
| `DOCS_CHUNK_OVERLAP_CHARS` | `500` | Overlap between chunks |
| `DOCS_MAX_WORKERS` | `2` | Background processing workers |
| `DOCS_PROCESSING_TIMEOUT_SECONDS` | `300` | Per-document timeout |
| `DOCS_MAX_TAGS_PER_DOCUMENT` | `20` | Tag limit |

Plus inherited vector-core settings (`QDRANT_URL`, `EMBEDDING_URL`, etc.).

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
- python-docx (DOCX extraction)
- python-pptx (PPTX extraction)
