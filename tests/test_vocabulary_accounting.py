"""Tests for the docs contribution to the shared sparse vocabulary.

The sparse half of hybrid search is only as good as the vocabulary: a token
that was never registered is silently dropped from every sparse vector
(``GlobalVocabulary.vectorize_document`` ignores unknown tokens), and the
per-codebase document counts feed the IDF weights used to rank *every*
codebase sharing the database.

These tests drive a real temporary ``GlobalVocabulary`` rather than a mock, so
the accounting is genuinely exercised: a mocked vocabulary turns every
registration call into a no-op stub that asserts nothing about correctness.
"""

from __future__ import annotations

import sqlite3
from typing import Any
from uuid import uuid4

import pytest
from qdrant_client.models import FieldCondition, PointStruct
from vector_core import GlobalVocabulary

from mcp_docs.indexing.indexer import DOCS_CODEBASE_ID, DocumentIndexer
from mcp_docs.models import ExtractionStatus
from mcp_docs.storage.database import DocumentStore

COLLECTION = "test_vocab_accounting"


class FakeStorage:
    """Minimal in-memory stand-in for QdrantStorage.

    Stores real points so the indexer can read a document's previously indexed
    content back, which is how the removal side of the vocabulary accounting
    learns what to subtract.
    """

    def __init__(self) -> None:
        self.points: dict[Any, PointStruct] = {}

    async def upsert_batch(self, collection: str, points: list[PointStruct]) -> None:
        for point in points:
            self.points[point.id] = point

    async def delete_by_filter(self, collection: str, field: str, value: Any) -> None:
        for point_id, point in list(self.points.items()):
            if (point.payload or {}).get(field) == value:
                del self.points[point_id]

    async def scroll_points(
        self,
        collection: str,
        filter_conditions: list[FieldCondition] | None = None,
        payload_fields: list[str] | None = None,
        limit: int = 5000,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        wanted = {c.key: c.match.value for c in (filter_conditions or [])}  # type: ignore[union-attr]
        matches = [
            dict(point.payload or {})
            for point in self.points.values()
            if all((point.payload or {}).get(key) == value for key, value in wanted.items())
        ]
        if payload_fields is not None:
            matches = [{k: p.get(k) for k in payload_fields} for p in matches]
        return matches


class FakeEmbedder:
    """Deterministic embedder: dense vectors are irrelevant to this accounting."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_indexer(
    store: DocumentStore, vocab: GlobalVocabulary, storage: FakeStorage | None = None
) -> tuple[DocumentIndexer, FakeStorage]:
    storage = storage or FakeStorage()
    indexer = DocumentIndexer(
        document_store=store,
        storage=storage,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        global_vocab=vocab,
        collection_name=COLLECTION,
    )
    return indexer, storage


@pytest.fixture
def vocab(tmp_path) -> GlobalVocabulary:
    return GlobalVocabulary(db_path=tmp_path / "vocabulary.db")


@pytest.fixture
def store(tmp_path) -> Any:
    store = DocumentStore(db_path=tmp_path / "documents.db")
    yield store
    store.close()


def _register(store: DocumentStore, tmp_path, name: str, text: str) -> Any:
    path = tmp_path / name
    path.write_text(text)
    return store.register(path)


def _doc_freq(vocab: GlobalVocabulary, token: str) -> int:
    """Global document frequency for a token, read straight from the database.

    Document frequency is what IDF is computed from, and it is not exposed on
    the public API, so the assertions read it directly rather than inferring it
    from vector weights.
    """
    conn = sqlite3.connect(vocab.db_path)
    try:
        row = conn.execute(
            "SELECT doc_freq FROM vocabulary WHERE token = ?", (token,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else 0


def _is_known(vocab: GlobalVocabulary, token: str) -> bool:
    """True when a token can appear in a sparse vector at all.

    ``vectorize_document`` silently drops tokens that are absent from the
    vocabulary, so a non-empty vector for a single-token text is exactly the
    property that matters for retrieval.
    """
    return bool(vocab.vectorize_document(token).indices)


class TestIndexDocumentVocabulary:
    """``index_document`` is the primary production path: the processing queue
    auto-indexes every newly extracted document through it."""

    @pytest.mark.asyncio
    async def test_second_document_registers_its_own_tokens(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: registration was gated on the docs document count being
        zero, so only the very first document ever indexed contributed tokens.
        Every later document's novel terms were missing from the vocabulary and
        therefore silently absent from its sparse vector."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        first = _register(store, tmp_path, "first.txt", "alpha content")
        await indexer.index_document(first.id, "alpha content")

        second = _register(store, tmp_path, "second.txt", "zxqnovel content")
        await indexer.index_document(second.id, "zxqnovel content")

        assert _is_known(vocab, "zxqnovel")
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 2

    @pytest.mark.asyncio
    async def test_reindexing_a_document_does_not_inflate_counts(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document re-indexed N times is still one document."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "stable.txt", "alpha beta")
        for _ in range(5):
            await indexer.index_document(doc.id, "alpha beta")

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "alpha") == 1
        assert _doc_freq(vocab, "beta") == 1

    @pytest.mark.asyncio
    async def test_reindex_subtracts_tokens_that_are_gone(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Editing a document out of a term must retire that term's document
        frequency, or IDF drifts away from the real corpus."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "edited.txt", "obsoleteterm stays")
        await indexer.index_document(doc.id, "obsoleteterm stays")
        assert _doc_freq(vocab, "obsoleteterm") == 1

        await indexer.index_document(doc.id, "replacementterm stays")

        assert _doc_freq(vocab, "obsoleteterm") == 0
        assert _doc_freq(vocab, "replacementterm") == 1
        assert _doc_freq(vocab, "stays") == 1
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1

    @pytest.mark.asyncio
    async def test_failed_index_leaves_the_contribution_unchanged(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The vocabulary must be updated before vectorization so a document's
        own novel tokens reach its sparse vector, but a subsequent embedding
        failure must not leave the contribution describing content that was
        never stored."""
        from unittest.mock import AsyncMock, MagicMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        first = _register(store, tmp_path, "kept.txt", "alpha")
        await indexer.index_document(first.id, "alpha")
        before = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)

        failing = MagicMock()
        failing.embed_batch = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
        indexer.embedder = failing

        doomed = _register(store, tmp_path, "doomed.txt", "unstorabletoken")
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            await indexer.index_document(doomed.id, "unstorabletoken")

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == before
        assert _doc_freq(vocab, "unstorabletoken") == 0


class TestFailedVocabularyWriteIsNotCompensated:
    """A delta that never reached the database must not be inverted. Doing so
    would subtract tokens, and a document, that were never added."""

    @pytest.mark.asyncio
    async def test_failed_delta_then_failed_index_leaves_counts_alone(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        established = _register(store, tmp_path, "established.txt", "alpha shared")
        await indexer.index_document(established.id, "alpha shared")
        before_count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        before_shared = _doc_freq(vocab, "shared")

        # The shared vocabulary database is busy for the delta only. A mock
        # that failed every call would make the erroneous compensation fail
        # too, so the counts would come out right for the wrong reason and the
        # test would pass with or without the fix.
        real_update = vocab.update_codebase_incremental
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database is locked")
            return real_update(*args, **kwargs)

        monkeypatch.setattr(vocab, "update_codebase_incremental", flaky)
        failing = MagicMock()
        failing.embed_batch = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
        indexer.embedder = failing

        doomed = _register(store, tmp_path, "doomed.txt", "beta shared")
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            await indexer.index_document(doomed.id, "beta shared")

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == before_count
        assert _doc_freq(vocab, "shared") == before_shared

    @pytest.mark.asyncio
    async def test_failed_batch_delta_then_failed_index_leaves_counts_alone(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same invariant on the bulk path, whose gate covers a batch."""
        from unittest.mock import AsyncMock, MagicMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        established = _register(store, tmp_path, "established.txt", "alpha shared")
        await indexer.index_document(established.id, "alpha shared")
        before_count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        before_shared = _doc_freq(vocab, "shared")

        real_update = vocab.update_codebase_incremental
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database is locked")
            return real_update(*args, **kwargs)

        monkeypatch.setattr(vocab, "update_codebase_incremental", flaky)
        failing = MagicMock()
        failing.embed_batch = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
        indexer.embedder = failing

        fresh = _register(store, tmp_path, "fresh.txt", "beta shared")
        store.update(fresh.id, extraction_status=ExtractionStatus.EXTRACTED)

        result = await indexer.index_all(force=False)

        assert result["indexed"] == 0
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == before_count
        assert _doc_freq(vocab, "shared") == before_shared


class TestDeleteDocumentVocabulary:
    @pytest.mark.asyncio
    async def test_delete_retires_the_document_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Removing a document from the index must remove its tokens too."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        keep = _register(store, tmp_path, "keep.txt", "shared alpha")
        drop = _register(store, tmp_path, "drop.txt", "shared droppedterm")
        await indexer.index_document(keep.id, "shared alpha")
        await indexer.index_document(drop.id, "shared droppedterm")
        assert _doc_freq(vocab, "shared") == 2

        await indexer.delete_document_index(drop.id)

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "droppedterm") == 0
        assert _doc_freq(vocab, "shared") == 1

    @pytest.mark.asyncio
    async def test_failed_deletion_keeps_the_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The points are still there, so their terms are still in the corpus.

        Writing them off would understate the document frequency of everything
        the document still contains.
        """
        from unittest.mock import AsyncMock

        indexer, storage = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "stubborn.txt", "stubbornterm")
        await indexer.index_document(doc.id, "stubbornterm")

        async def refuse(*_args, **_kwargs):
            raise RuntimeError("qdrant unavailable")

        storage.delete_by_filter = refuse

        await indexer.delete_document_index(doc.id)

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "stubbornterm") == 1

    @pytest.mark.asyncio
    async def test_a_lost_response_still_retires_the_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raised error does not prove the points survived.

        Qdrant can apply the delete and lose only the response. Reading the
        points back is what settles it, and their absence means the deletion
        happened.
        """
        from unittest.mock import AsyncMock

        indexer, storage = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "vanished.txt", "vanishedterm")
        await indexer.index_document(doc.id, "vanishedterm")

        real_delete = storage.delete_by_filter

        async def apply_then_fail(*args, **kwargs):
            await real_delete(*args, **kwargs)
            raise RuntimeError("connection reset")

        storage.delete_by_filter = apply_then_fail

        await indexer.delete_document_index(doc.id)

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 0
        assert _doc_freq(vocab, "vanishedterm") == 0

    @pytest.mark.asyncio
    async def test_delete_of_an_unindexed_document_is_a_no_op(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        keep = _register(store, tmp_path, "keep.txt", "alpha")
        await indexer.index_document(keep.id, "alpha")

        await indexer.delete_document_index(uuid4())

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "alpha") == 1


class TestIndexAllVocabulary:
    @pytest.mark.asyncio
    async def test_incremental_run_keeps_the_rest_of_the_corpus(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an incremental run re-registered the codebase with only
        the changed documents. ``register_codebase`` replaces the whole
        contribution, so every unchanged document's tokens were subtracted and
        the document count collapsed to the size of the delta."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        established = _register(store, tmp_path, "established.txt", "establishedterm shared")
        await indexer.index_document(established.id, "establishedterm shared")

        fresh = _register(store, tmp_path, "fresh.txt", "freshterm shared")
        store.update(fresh.id, extraction_status=ExtractionStatus.EXTRACTED)

        result = await indexer.index_all(force=False)

        assert result["indexed"] == 1
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 2
        assert _doc_freq(vocab, "establishedterm") == 1
        assert _doc_freq(vocab, "freshterm") == 1
        assert _doc_freq(vocab, "shared") == 2

    @pytest.mark.asyncio
    async def test_force_run_rebuilds_the_whole_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A force rebuild reindexes the entire corpus, so it may replace the
        contribution outright, and the result must describe exactly that
        corpus (no drift accumulated from earlier incremental runs)."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        first = _register(store, tmp_path, "one.txt", "alpha shared")
        second = _register(store, tmp_path, "two.txt", "beta shared")
        await indexer.index_document(first.id, "alpha shared")
        await indexer.index_document(second.id, "beta shared")

        result = await indexer.index_all(force=True)

        assert result["indexed"] == 2
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 2
        assert _doc_freq(vocab, "shared") == 2
        assert _doc_freq(vocab, "alpha") == 1
        assert _doc_freq(vocab, "beta") == 1

    @pytest.mark.asyncio
    async def test_force_run_keeps_documents_it_could_not_re_extract(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A force rebuild skips documents whose file has gone missing, but it
        does not delete their points either. Dropping their tokens would leave
        searchable content with no vocabulary contribution."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        present = _register(store, tmp_path, "present.txt", "presentterm")
        missing = _register(store, tmp_path, "missing.txt", "missingterm")
        await indexer.index_document(present.id, "presentterm")
        await indexer.index_document(missing.id, "missingterm")

        (tmp_path / "missing.txt").unlink()

        await indexer.index_all(force=True)

        assert _doc_freq(vocab, "presentterm") == 1
        assert _doc_freq(vocab, "missingterm") == 1
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 2


class TestStoredTokenSymmetry:
    """The tokens registered for a document and the tokens later subtracted for
    it must come from the same text, or every re-index leaks the difference."""

    @pytest.mark.asyncio
    async def test_summary_tokens_are_registered_and_retired(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The summary point's content (filename, title, tags) is vectorized
        like any other point, so its tokens belong in the vocabulary."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "distinctivefilename.txt", "body")
        await indexer.index_document(doc.id, "body")
        assert _doc_freq(vocab, "distinctivefilename") == 1

        await indexer.delete_document_index(doc.id)
        assert _doc_freq(vocab, "distinctivefilename") == 0

    @pytest.mark.asyncio
    async def test_repeated_edits_do_not_leak_document_frequency(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alternating content must land back on exactly the starting counts."""
        from unittest.mock import AsyncMock

        indexer, _ = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "churn.txt", "alpha")
        for _ in range(4):
            await indexer.index_document(doc.id, "alpha stable")
            await indexer.index_document(doc.id, "beta stable")

        assert _doc_freq(vocab, "alpha") == 0
        assert _doc_freq(vocab, "beta") == 1
        assert _doc_freq(vocab, "stable") == 1
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
