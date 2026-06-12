# Changelog

## [1.1.8] - 2026-06-12

### Changed

- Bump vector-core to v1.2.5 (case-insensitive `find_connections` type filters; symmetric `get_metadata` string round-trip — neither path is used by mcp-docs).

## [1.1.7] - 2026-06-12

### Fixed

- `rename_directory` and `move_directory` no longer corrupt the recorded paths of documents in *sibling* directories that share a name prefix. The batch path update matched with an unanchored, unescaped SQL `LIKE`, so renaming `/data/docs` also rewrote paths under `/data/docs2`, `_` in a directory name acted as a single-character wildcard (renaming `/data/my_dir` rewrote `/data/myxdir`), and — because SQLite `LIKE` is case-insensitive for ASCII — case-only siblings like `/data/Docs` were rewritten too. Matching is now an exact, case-sensitive prefix comparison anchored at the directory boundary, in both the SQLite store (paths and document roots) and the Qdrant index path updater.
- `search_documents(tags=...)` now normalizes tag filters (lowercase, strip) to match how tags are stored. Previously a wrong-case tag like `"Finance"` silently returned no results from search while the same filter worked in `list_documents`, whose SQL path already normalized.
- A stale `extraction_error` no longer survives successful re-extraction. The re-queue path's explicit `extraction_error=None` was silently ignored because `DocumentStore.update()` used `None` as its "not provided" sentinel; the parameter now uses an explicit unset sentinel so `None` clears the stored error, and the successful-extraction path clears it as well.

## [1.1.6] - 2026-06-12

### Changed

- **Bumped `vector-core` to `v1.2.4`.** Pure dependency hygiene: v1.2.4 makes `FactStore.create()`/`update()` raise `ValueError` on blank subject/predicate/object/type fields and out-of-range confidence values before any database access. mcp-docs exposes no fact tools — facts live in mcp-notes; docs uses the glossary helper and document store — so no behavior of this server changes. This keeps the pin current with the shared library.

## [1.1.5] - 2026-06-12

### Fixed

- **Case-only term renames and renaming a term to one of the entry's own aliases now work in `update_glossary_entry`.** Bumped `vector-core` to `v1.2.3`, which fixes the term-uniqueness check in `GlossaryStore.update()` to exclude the entry being updated. Previously the check matched the entry's own rows, so renaming a term to a different casing of itself (e.g. `"USAF"` → `"Usaf"`) or promoting one of the entry's own aliases to be the term raised a spurious `TermExistsError`. Since `update_glossary_entry` delegates term renames to the store, both kinds of rename were impossible through the docs MCP tools; they now succeed as expected.

## [1.1.4] - 2026-06-12

### Fixed

- **`update_glossary_entry` can no longer destroy an entry's aliases when a new alias collides with another entry.** Bumped `vector-core` to `v1.2.2`, which makes `GlossaryStore.create()`/`update()` validate aliases — cross-entry collisions and case-normalized intra-list duplicates — *before* any row is written, raising `TermExistsError` with the store left fully unchanged (plus a rollback-on-error backstop). This matters for mcp-docs because the glossary tools preflight blank and intra-list-duplicate aliases themselves but rely on the store to catch cross-entry collisions: before this bump, an `update_glossary_entry` call whose new alias collided with *another* entry's term or alias returned a clean "duplicate" error while a partial mutation — the target entry's aliases already deleted — could still be committed later on the long-lived connection. That reachable data-loss path is now closed; the duplicate error genuinely means nothing changed.

## [1.1.3] - 2026-06-11

### Fixed

- **`update_glossary_entry` still accepts `domain: ""` to clear the domain.** vector-core v1.2.1 now rejects blank `domain` values in the shared glossary helper, which would have broken this tool's documented empty-string clear convention; the wrapper now translates a blank `domain` to `None` (the helper's clear value) before delegating. Covered by regression tests.

### Changed

- Bumped the `vector-core` dependency to `v1.2.1`. This is a fix release in the shared library; the change most relevant to mcp-docs is per-batch progress callbacks in `embed_all`, so indexing progress is reported as each batch completes rather than only at the end. The remaining fixes (FactStore batch-read ordering, glossary `entry_hash` staleness on alias updates, and blank/duplicate-input validation in the shared glossary tool helper) are inherited shared-library hygiene.

## [1.1.2] - 2026-05-31

### Fixed

- **`doc_type` filter now fails fast on an invalid value instead of silently matching nothing.** In `list_documents`, `search_documents`, and `keyword_search`, a typo'd or wrong-case `doc_type` (e.g. `"pyhton"`, `"PDF"`) was passed straight to the SQL/Qdrant filter and quietly returned an empty result — indistinguishable from "no documents of this type." The value is now validated against the known document types (returning a clear `INVALID_INPUT` error listing the valid values) and normalized to its canonical lowercase form, so a correct-but-wrong-case type such as `"PDF"` now matches as expected. This mirrors the existing `status`/`extraction_status` validation in `list_documents`.
- **`index_document` returns a structured error response when a document isn't ready to index.** It previously returned a bare `{"error": ...}` dict with no `error_code`, so callers using `is_error_response()` treated the failure as success; it now returns a proper `CONFLICT` error like every other failure branch.

### Changed

- `keyword_search`'s `doc_type` filter now uses `MatchValue` (matching the search engine) instead of a raw dict, for type-correct Qdrant filtering.

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
