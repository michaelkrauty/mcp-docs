# Changelog

## [1.0.2] - 2026-05-25

### Changed

- Bumped `vector-core` dependency to the reachable `v1.0.4` tag, aligning with corrected Vector Core release metadata.

## [1.0.1] - 2026-05-23

### Changed

- Tagged the first reproducible consumer release after pinning `vector-core` to `v1.0.3`.

## [1.0.0] - 2026-03-20

Initial public release.

### Features

- **36 MCP tools** for document management, search, indexing, glossary, and filesystem operations
- **Multi-format extraction**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, RTF, CSV, EPUB, XML, and plain text
- **Automatic OCR fallback** via vision model API for scanned/image-based PDFs when text extraction yields insufficient content (pdf2image + Pillow)
- **Hybrid vector search** with dense embeddings + TF-IDF sparse vectors (RRF fusion)
- **Document root management** for organizing and scanning document collections
- **Background processing queue** with bounded size and progress tracking
- **Hash-based document verification** for integrity checking
- **Glossary system** for domain-specific term definitions with vector-indexed search
- **Incremental indexing** — only processes new or modified documents
- **Circuit breaker** on OCR client to prevent cascading failures
- **Thread-safe SQLite** with WAL mode and per-thread connections
