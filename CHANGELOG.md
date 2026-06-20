# Changelog

## [1.1.22] - 2026-06-20

### Fixed

- **`scan_root` now honors a document root's `recursive=False` setting instead of always walking subdirectories.** A root's `recursive` flag was accepted by `add_document_root`, persisted, and read back into the `DocumentRoot` model, but the scan never consulted it. The descend decision in the file walk checked the scanner instance attribute `self.recursive`, which is set once at construction, and the production scanner is a process-wide singleton built with the default `recursive=True`. As a result every root, including ones added with `recursive=False`, was scanned recursively, and subdirectory files were registered, queued, extracted, and indexed against the user's intent. The walk now descends only when both the scanner instance and the root permit recursion, so a root added with `recursive=False` is no longer walked recursively by the production scanner, while a scanner explicitly constructed with `recursive=False` still scans non-recursively as before.

## [1.1.21] - 2026-06-19

### Fixed

- **`move_file`, `rename_directory`, and `move_directory` no longer refuse to move a document whose extraction permanently failed or was cancelled.** The move and rename tools wait for processing to finish before touching the file, but they gated on `wait_for_documents`, which only reports success when every document reaches `COMPLETED`. A document in a terminal `FAILED` or `CANCELLED` state never becomes `COMPLETED`, so the operation returned `TIMEOUT` ("Document processing did not complete within timeout") and the file could never be moved or renamed, even though nothing was processing it and no timeout had actually elapsed. For example, a directory holding a DRM-protected or otherwise unextractable file became permanently un-movable. The tools now pass `require_completed=False`, so any terminal state (completed, failed, or cancelled) lets the operation proceed; a document genuinely still queued or processing still blocks until it finishes or the wait times out.

## [1.1.20] - 2026-06-19

### Changed

- Bumped `vector-core` to `v1.2.8`. Pure dependency hygiene: v1.2.8 fixes `SparseVectorizer.extend_vocab` IDF recomputation and the `limit=0` semantics of two library list methods, plus two docstring corrections. mcp-docs does not exercise any of these paths (it never passes `limit=0` to the glossary store, and does not use the standalone `SparseVectorizer`), so no behavior of this server changes. This keeps the pin current with the shared library.

## [1.1.19] - 2026-06-19

### Fixed

- **`query_by_path_prefix` now matches the directory boundary exactly, so directory operations no longer over-match sibling directories.** The query used a bare SQL `LIKE` pattern, but SQLite `LIKE` treats `_` and `%` as wildcards and is case-insensitive for ASCII, so a query for a directory like `my_docs/` also matched documents under siblings such as `myXdocs/` (the `_` matched `X`) or `My_Docs/` (case). That drove three directory tools wrong on common directory names: `delete_directory` reported "contains N registered documents" and refused to delete an empty directory whose name collided with a populated sibling, and `rename_directory`/`move_directory` gathered unrelated documents to wait on (up to a 120s timeout). The query now uses the same anchored, case-sensitive `SUBSTR` boundary match already used by `update_paths_batch` and `update_document_roots_batch`, so it returns only documents strictly under the requested directory. The path rewrite itself was already anchored, so no stored paths were affected.

## [1.1.18] - 2026-06-15

### Fixed

- **`update_document_tags` now updates the search index, so tag filters and search results no longer go stale after a tag change.** The tool wrote the new tags to the document registry but never updated the vector index. Each point carries a `tags` payload used to filter searches (`search_documents(tags=...)`) and to report a result's tags, and the document summary point additionally embeds the tags in its searchable content. After a tag change both the filter metadata and the indexed content stayed stale, so searches would still match the old tags, miss the new ones, and show the stale set until a full reindex. The tool now synchronizes the change through a new `DocumentIndexer.update_document_tags_in_index`, which updates the `tags` payload on every one of the document's points and, when the document is indexed, regenerates the summary point so its content and vectors reflect the new tags. The synchronization is metadata-only (it does not need the source file) and best-effort, so a transient backend error does not block the registry update.

## [1.1.17] - 2026-06-14

### Fixed

- **`wait_for_document` now returns immediately for a document that already finished, instead of blocking until the timeout.** `DocumentProcessor.wait_for` only consulted the in-memory completed cache, which does not survive a restart. A document that was extracted, indexed, failed, or cancelled in a previous session was therefore not cached, so `wait_for` created a wait event and blocked for the full timeout (300 seconds by default) waiting for a worker event that would never fire, then reported a timeout for a document that was actually done. It now also checks the persisted extraction status and resolves a terminal document (`extracted`/`indexed` as completed, `failed`, or `cancelled`) from the database right away; queued and processing documents still wait as before. The extraction-to-processing status mapping is now shared between `wait_for` and `get_processing_status` so the two cannot drift.

## [1.1.16] - 2026-06-14

### Fixed

- **`cancel_processing` no longer marks documents as failed, and no longer clobbers documents that are not actually queued.** `DocumentProcessor.cancel()` refused only documents that were currently being processed; for anything else it unconditionally set `extraction_status=failed`. Two problems followed: a document that was already extracted or indexed (so not in progress) had its good status overwritten with `failed`, and because startup recovery re-enqueues failed documents, a cancelled document was reprocessed on the next restart, defeating the cancellation and misreporting it as an extraction error. Cancellation is now guarded to act only on documents that are still `queued`, returns a conflict for anything already processing, complete, failed, or cancelled, and records a dedicated `cancelled` extraction status, which recovery does not re-enqueue. A `cancelled` value was added to the `ExtractionStatus` enum and is surfaced by `get_processing_status`. Documents cancelled by a previous version (stored as `failed`) are migrated to `cancelled` during startup recovery so they are no longer re-enqueued.

## [1.1.15] - 2026-06-14

### Fixed

- **`index_all_documents` now indexes the entire extracted corpus instead of only the 50 most recently touched documents.** `DocumentIndexer.index_all()` enumerated documents with `DocumentStore.query()`, whose default `limit` is 50, so any backlog of more than 50 extracted-but-unindexed documents was silently truncated and most documents never entered the search index. The background indexing pass that runs at startup goes through the same code path and hit the same cap. Indexing now streams the full corpus through a filtered form of `DocumentStore.iter_all(extraction_status=...)`, which applies no row limit, skips an individual unreadable row rather than aborting, and lets a systemic database error propagate instead of masquerading as an empty corpus. This is the document-side counterpart of the facts fix shipped in vector-core v1.2.6.

## [1.1.14] - 2026-06-14

### Fixed

- **`move_directory` now validates that both the source and destination are within a registered document root**, matching `move_file`, `rename_directory`, and `create_directory`. Previously it would move any directory on the filesystem and rewrite document paths even when the paths were outside every document root; when the destination fell outside all roots, the moved documents kept a `document_root` that no longer contained them, leaving the registry inconsistent. Out-of-root source or destination paths are now rejected with a `PERMISSION_DENIED` error before anything is moved.

## [1.1.13] - 2026-06-14

### Fixed

- **Removing a document root with `delete_documents=True` no longer leaves the deleted documents' vectors orphaned in the search index.** `remove_document_root` deleted each document's registry row but never removed its points from Qdrant or marked its fact sources deleted, so those vectors stayed searchable while pointing at documents that no longer existed. Document deletion is now centralized in a shared helper used by both `delete_document` and `remove_document_root`, so removing a root cleans up vector-index points and fact-source links the same way single-document deletion does. The response also reports `sources_marked_deleted`.

## [1.1.12] - 2026-06-14

### Fixed

- **A directory scan that does not cover the whole tree no longer marks unvisited documents as deleted or purges their vectors from the search index.** `scan_root` reconciles deletions by treating any registered document it did not encounter during the filesystem walk as removed from disk. When the walk did not reach every file — because it was truncated at `MAX_FILES_PER_ROOT`, an error aborted traversal, or a subdirectory had become unreadable (`pathlib`'s `rglob`/`glob` silently omit the children of a directory they cannot list rather than raising) — the set of seen paths was incomplete, so documents that were simply never reached were wrongly flagged `DELETED` and had their Qdrant points removed, losing index data for files that still exist on disk. Scanning now uses `os.walk` with an error callback so directory-listing failures are detected instead of silently swallowed, tracks whether the walk completed via a new `complete` boolean on the scan result, and runs deletion reconciliation only when the scan completed. An incomplete scan performs no deletions and reports the condition via `complete` and an explanatory entry in `errors`.

## [1.1.11] - 2026-06-14

### Fixed

- **CSV files with any non-ASCII content (e.g. accented merchant names in a financial export) no longer fail extraction.** `extract_csv` delegated to markitdown's CSV converter, which decodes with the locale default encoding — ASCII under the MCP server's environment — and raised `UnicodeDecodeError`, leaving such CSVs perpetually stuck in `processing`/`failed` and absent from search. CSV extraction now reads with an encoding fallback (`utf-8-sig` → `utf-8` → `latin-1` → `cp1252`, then a lossy UTF-8 last resort) and renders the markdown table directly via the `csv` module, so quoted fields, embedded commas/newlines, and pipe characters are handled correctly regardless of file encoding or locale. `extract_rtf` now shares the same encoding-fallback helper, and a UTF-8 BOM is stripped rather than left as a stray character in the first cell.

## [1.1.10] - 2026-06-14

### Changed

- Bumped the shared `vector-core` library to v1.2.7 (FactStore: case-insensitive `query()`/`list_summaries()` type filters + rejection of inverted `valid_from`/`valid_to` ranges). mcp-docs exposes no fact tools (facts live in mcp-notes), so no behavior of this server changes; this keeps the pin current with the shared library.

## [1.1.9] - 2026-06-13

### Changed

- Bumped the shared `vector-core` library to v1.2.6. v1.2.6 fixes `FactIndexer.index_all()`/`_train_vocabulary()` to index the complete fact corpus instead of the 50 most-recently-modified facts and to register the sparse vocabulary from all facts. mcp-docs exposes no fact tools (facts live in mcp-notes; docs uses the glossary helper and document store), so no behavior of this server changes; this keeps the pin current with the shared library.

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
