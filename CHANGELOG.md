# Changelog

## [1.1.1] - 2026-05-30

### Fixed

- **`keyword_search` no longer crashes on invalid input.** Both validation guards referenced `ErrorCode.VALIDATION_ERROR`, which does not exist on vector-core's `ErrorCode` enum, so an empty keyword (or disabling both `search_filename` and `search_content`) raised `AttributeError` instead of returning the intended structured error. They now use `ErrorCode.VALIDATION_FAILED`, and the "at least one search field" guard runs before the search engine is initialized so it can never be masked by a service error. Covered by regression tests.
- **Jupyter code-fence language is now inferred from the kernel name.** `.ipynb` notebooks whose metadata has only `kernelspec.name` (e.g. `"python3"`) — a very common shape with no `language_info` and no `kernelspec.language` — previously produced untagged code fences. The extractor now falls back to the leading alphabetic run of the kernel name, so `python3`, `python3.11`, and `julia-1.9` tag fences as `python`/`python`/`julia`.

### Changed

- Bumped the `vector-core` dependency to `v1.2.0`.

### Documentation

- Documented Jupyter notebook (`.ipynb`) support in the README (features list and Supported Formats table) — it shipped in v1.1.0 but was previously undocumented.

## [1.1.0] - 2026-05-30

### Added

- **Jupyter notebook (`.ipynb`) support.** Notebooks are now a recognized document type: they are discovered during scanning, and a dedicated extractor parses the notebook JSON into searchable text — Markdown cells are kept as prose and code cells as fenced code blocks tagged with the notebook's language (from `language_info`/`kernelspec`). Cell outputs and raw cells are skipped (outputs are frequently large and non-textual), and the document title is taken from the first level-1 Markdown heading. Malformed notebooks raise a normal extraction error and are marked failed like any other unprocessable document.

## [1.0.4] - 2026-05-30

### Changed

- Bumped the `vector-core` dependency to `v1.1.0`, which adds nested ignore-file support to `FileDiscovery`.

## [1.0.3] - 2026-05-27

### Fixed

- Aligned the runtime package `__version__` constant, project metadata, lockfile package entry, and version regression test.
- Bumped the `vector-core` dependency to `v1.0.5`, where `vector_core.__version__` matches package metadata.

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
