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
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from qdrant_client.models import FieldCondition, Filter, PointStruct
from vector_core import GlobalVocabulary
from vector_core.storage.embedding_fragments import is_derived_fragment

from mcp_docs.indexing.indexer import DOCS_CODEBASE_ID, DocumentIndexer
from mcp_docs.models import ExtractionStatus
from mcp_docs.storage.database import DocumentStore

COLLECTION = "test_vocab_accounting"

pytestmark = pytest.mark.usefixtures("embedding_generation")


class FakeStorage:
    """Minimal in-memory stand-in for QdrantStorage.

    Stores real points so the indexer can read a document's previously indexed
    content back, which is how the removal side of the vocabulary accounting
    learns what to subtract.
    """

    def __init__(self) -> None:
        self.points: dict[Any, PointStruct] = {}
        self.url = "http://isolated.invalid"

    async def get_client(self):
        return self

    async def update_payload(self, collection, filter_conditions, payload):
        matches, _ = await self.scroll(
            collection,
            scroll_filter=Filter(must=filter_conditions),
            limit=max(1, len(self.points)),
        )
        for point in matches:
            point.payload.update(payload)

    async def retrieve(self, collection, ids, **kwargs):
        return [self.points[point_id] for point_id in ids if point_id in self.points]

    async def upsert(self, collection, points, **kwargs):
        await self.upsert_batch(collection, points)

    async def delete(self, collection, points_selector, **kwargs):
        for point_id in points_selector.points:
            self.points.pop(point_id, None)

    async def scroll(self, collection, scroll_filter=None, offset=None, limit=128, **kwargs):
        from qdrant_client.local.payload_filters import check_filter

        points = [
            point
            for point in self.points.values()
            if scroll_filter is None
            or check_filter(
                scroll_filter,
                point.payload or {},
                point.id,
                dict.fromkeys(point.vector, True) if isinstance(point.vector, dict) else {},
            )
        ]
        start = offset or 0
        end = min(start + limit, len(points))
        return points[start:end], end if end < len(points) else None

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
            matches = [{k: p[k] for k in payload_fields if k in p} for p in matches]
        return matches


class FakeEmbedder:
    """Deterministic embedder: dense vectors are irrelevant to this accounting."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    @staticmethod
    def split_text(text: str, **kwargs):
        return [SimpleNamespace(start=0, end=len(text), text=text)]

    async def embed_all(self, texts, **kwargs):
        return await self.embed_batch(texts)


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
        row = conn.execute("SELECT doc_freq FROM vocabulary WHERE token = ?", (token,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else 0


def _summary_point(storage: FakeStorage, document_id: Any) -> PointStruct:
    """Canonical summary point for a document, skipping derived fragments."""
    matches = [
        point
        for point in storage.points.values()
        if (point.payload or {}).get("document_id") == str(document_id)
        and (point.payload or {}).get("type") == "document"
        and not is_derived_fragment(point.payload or {})
    ]
    assert len(matches) == 1
    return matches[0]


def _sparse_indices(point: PointStruct) -> list[int]:
    vector = point.vector
    assert isinstance(vector, dict)
    return list(vector["sparse"].indices)


def _assert_same_contribution(left: GlobalVocabulary, right: GlobalVocabulary) -> None:
    """Document frequencies and the docs document count match.

    A retired term stays in the append-only vocabulary at document frequency
    0, so a token missing from the other side counts as 0.
    """
    assert left.get_codebase_doc_count(DOCS_CODEBASE_ID) == right.get_codebase_doc_count(
        DOCS_CODEBASE_ID
    )
    left_freq = left._get_doc_freq()
    right_freq = right._get_doc_freq()
    for token in set(left_freq) | set(right_freq):
        assert left_freq.get(token, 0) == right_freq.get(token, 0), token


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
        failing.split_text.side_effect = FakeEmbedder.split_text
        failing.embed_all = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
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
        failing.split_text.side_effect = FakeEmbedder.split_text
        failing.embed_all = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
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
        failing.split_text.side_effect = FakeEmbedder.split_text
        failing.embed_all = AsyncMock(side_effect=RuntimeError("embedding unavailable"))
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
    async def test_an_unverifiable_deletion_keeps_the_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the read-back fails too, the outcome is genuinely unknown.

        Keeping the contribution is the safer answer: the points may still be
        searchable, and retiring their terms would understate the document
        frequency of everything still in them.
        """
        from unittest.mock import AsyncMock

        indexer, storage = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        doc = _register(store, tmp_path, "opaque.txt", "opaqueterm")
        await indexer.index_document(doc.id, "opaqueterm")

        async def refuse(*_args, **_kwargs):
            raise RuntimeError("qdrant unavailable")

        storage.delete_by_filter = refuse
        storage.scroll_points = refuse

        await indexer.delete_document_index(doc.id)

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "opaqueterm") == 1

    @pytest.mark.asyncio
    async def test_a_partial_deletion_keeps_the_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some points gone is not all points gone.

        A surviving chunk is still searchable, so the document's terms stay on
        the books until the deletion actually completes.
        """
        from unittest.mock import AsyncMock

        indexer, storage = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())

        body = "\n\n".join(f"survivingterm{i} paragraph body text" for i in range(40))
        doc = _register(store, tmp_path, "partial.txt", body)
        await indexer.index_document(doc.id, body)
        assert len(storage.points) > 1, "fixture must produce more than one point"

        async def drop_one_then_fail(*_args, **_kwargs):
            storage.points.pop(next(iter(storage.points)))
            raise RuntimeError("connection reset")

        storage.delete_by_filter = drop_one_then_fail

        await indexer.delete_document_index(doc.id)

        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == 1
        assert _doc_freq(vocab, "survivingterm") == 1

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


class TestSummaryMetadataVocabulary:
    """Tag and filename edits rewrite only the summary point.

    The document's vocabulary contribution is the union of every canonical
    point, so the delta is that union with the new summary in place of the
    stored one. It has to land before the summary is vectorized, and it must
    not change the document count.
    """

    @staticmethod
    def _indexer(store, vocab, monkeypatch):
        from unittest.mock import AsyncMock

        indexer, storage = _make_indexer(store, vocab)
        monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())
        return indexer, storage

    @pytest.mark.asyncio
    async def test_tag_rename_registers_the_new_term_and_retires_the_old(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, storage = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "notes.txt", "ordinarybody")
        doc = store.update_tags(doc.id, ["zxqoldtag"])
        await indexer.index_document(doc.id, "ordinarybody")
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)

        doc = store.update_tags(doc.id, ["zxqnewtag"])
        await indexer.update_document_tags_in_index(doc)

        freqs = vocab._get_doc_freq()
        assert freqs["zxqnewtag"] == 1
        assert freqs["zxqoldtag"] == 0
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count
        indices = _sparse_indices(_summary_point(storage, doc.id))
        assert vocab._get_vocab()["zxqnewtag"] in indices
        assert vocab._get_vocab()["zxqoldtag"] not in indices

    @pytest.mark.asyncio
    async def test_filename_rename_registers_the_new_term_and_retires_the_old(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, storage = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "zxqoldbasename.txt", "ordinarybody")
        await indexer.index_document(doc.id, "ordinarybody")
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)

        doc = store.update(doc.id, filename="zxqnewbasename.txt")
        await indexer.update_document_filename_in_index(doc)

        freqs = vocab._get_doc_freq()
        assert freqs["zxqnewbasename"] == 1
        assert freqs["zxqoldbasename"] == 0
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count
        indices = _sparse_indices(_summary_point(storage, doc.id))
        assert vocab._get_vocab()["zxqnewbasename"] in indices
        assert vocab._get_vocab()["zxqoldbasename"] not in indices

    @pytest.mark.asyncio
    async def test_term_shared_with_a_chunk_keeps_its_document_frequency(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A term that also lives in a chunk is still in the document's union."""
        indexer, _ = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "notes.txt", "sharedterm lives in the chunk")
        doc = store.update_tags(doc.id, ["sharedterm"])
        await indexer.index_document(doc.id, "sharedterm lives in the chunk")
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        assert vocab._get_doc_freq()["sharedterm"] == 1

        doc = store.update_tags(doc.id, ["zxqreplacement"])
        await indexer.update_document_tags_in_index(doc)

        assert vocab._get_doc_freq()["sharedterm"] == 1
        assert vocab._get_doc_freq()["zxqreplacement"] == 1
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count

    @pytest.mark.asyncio
    async def test_repeated_tag_renames_match_a_fresh_registration(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, _ = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "churn.txt", "stablebody")
        doc = store.update_tags(doc.id, ["zxqalpha"])
        await indexer.index_document(doc.id, "stablebody")
        for _ in range(4):
            doc = store.update_tags(doc.id, ["zxqbeta"])
            await indexer.update_document_tags_in_index(doc)
            doc = store.update_tags(doc.id, ["zxqalpha"])
            await indexer.update_document_tags_in_index(doc)
        # Finish on the other tag. Ending where we started would also match a
        # fresh registration if the renames never touched the vocabulary.
        doc = store.update_tags(doc.id, ["zxqbeta"])
        await indexer.update_document_tags_in_index(doc)

        fresh = GlobalVocabulary(db_path=tmp_path / "fresh-tags.db")
        fresh_store = DocumentStore(db_path=tmp_path / "fresh-tags-docs.db")
        try:
            fresh_indexer, _ = self._indexer(fresh_store, fresh, monkeypatch)
            fresh_doc = _register(fresh_store, tmp_path, "churn.txt", "stablebody")
            fresh_doc = fresh_store.update_tags(fresh_doc.id, ["zxqbeta"])
            await fresh_indexer.index_document(fresh_doc.id, "stablebody")
            _assert_same_contribution(vocab, fresh)
        finally:
            fresh_store.close()

    @pytest.mark.asyncio
    async def test_repeated_filename_renames_match_a_fresh_registration(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, _ = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "zxqalpha.txt", "stablebody")
        await indexer.index_document(doc.id, "stablebody")
        for _ in range(4):
            doc = store.update(doc.id, filename="zxqbeta.txt")
            await indexer.update_document_filename_in_index(doc)
            doc = store.update(doc.id, filename="zxqalpha.txt")
            await indexer.update_document_filename_in_index(doc)
        doc = store.update(doc.id, filename="zxqbeta.txt")
        await indexer.update_document_filename_in_index(doc)

        fresh = GlobalVocabulary(db_path=tmp_path / "fresh-names.db")
        fresh_store = DocumentStore(db_path=tmp_path / "fresh-names-docs.db")
        try:
            fresh_indexer, _ = self._indexer(fresh_store, fresh, monkeypatch)
            fresh_doc = _register(fresh_store, tmp_path, "zxqbeta.txt", "stablebody")
            await fresh_indexer.index_document(fresh_doc.id, "stablebody")
            _assert_same_contribution(vocab, fresh)
        finally:
            fresh_store.close()

    @pytest.mark.asyncio
    async def test_summary_refresh_failure_restores_the_stored_contribution(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delta applied before a failed rebuild is put back in line with storage."""
        indexer, _ = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "notes.txt", "bodytoken")
        doc = store.update_tags(doc.id, ["zxqoldtag"])
        await indexer.index_document(doc.id, "bodytoken")
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        doc = store.update_tags(doc.id, ["zxqnewtag"])

        async def fail_refresh(_document) -> None:
            raise RuntimeError("summary unavailable")

        monkeypatch.setattr(indexer, "_refresh_summary", fail_refresh)
        await indexer.update_document_tags_in_index(doc)

        freqs = vocab._get_doc_freq()
        # The new term reached the append-only vocabulary, then its document
        # frequency was given back because the summary was never stored.
        assert "zxqnewtag" in freqs
        assert freqs["zxqnewtag"] == 0
        indexed, stored = await indexer._indexed_token_set(doc.id)
        assert indexed
        assert "zxqoldtag" in stored
        assert "zxqnewtag" not in stored
        for token, freq in freqs.items():
            assert freq == (1 if token in stored else 0), token
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count

    @pytest.mark.asyncio
    async def test_vocabulary_write_failure_refreshes_summary_without_compensation(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, storage = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "notes.txt", "bodytoken")
        doc = store.update_tags(doc.id, ["zxqoldtag"])
        await indexer.index_document(doc.id, "bodytoken")
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        before = dict(vocab._get_doc_freq())

        real_update = vocab.update_codebase_incremental
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database is locked")
            return real_update(*args, **kwargs)

        monkeypatch.setattr(vocab, "update_codebase_incremental", flaky)
        doc = store.update_tags(doc.id, ["zxqnewtag"])
        await indexer.update_document_tags_in_index(doc)

        # The failed write is not inverted. A mock that kept failing would hide
        # that compensation, because the repair would fail too.
        assert calls["n"] == 1
        content = _summary_point(storage, doc.id).payload["content"]
        assert "zxqnewtag" in content
        assert "zxqoldtag" not in content
        assert vocab._get_doc_freq() == before
        assert "zxqnewtag" not in vocab._get_doc_freq()
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count

    @pytest.mark.asyncio
    async def test_unindexed_document_does_not_change_the_vocabulary(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, _ = self._indexer(store, vocab, monkeypatch)
        kept = _register(store, tmp_path, "kept.txt", "kepttoken")
        await indexer.index_document(kept.id, "kepttoken")
        before = dict(vocab._get_doc_freq())
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)

        pending = _register(store, tmp_path, "pending.txt", "pendingbody")
        pending = store.update_tags(pending.id, ["zxqpendingtag"])
        await indexer.update_document_tags_in_index(pending)
        pending = store.update(pending.id, filename="zxqrenamedpending.txt")
        await indexer.update_document_filename_in_index(pending)

        assert vocab._get_doc_freq() == before
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count

    @pytest.mark.asyncio
    async def test_unreadable_points_refresh_without_a_vocabulary_change(
        self, tmp_path, store, vocab, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer, storage = self._indexer(store, vocab, monkeypatch)
        doc = _register(store, tmp_path, "notes.txt", "bodytoken")
        doc = store.update_tags(doc.id, ["zxqoldtag"])
        await indexer.index_document(doc.id, "bodytoken")
        before = dict(vocab._get_doc_freq())
        count = vocab.get_codebase_doc_count(DOCS_CODEBASE_ID)
        doc = store.update_tags(doc.id, ["zxqnewtag"])

        real_scroll = storage.scroll_points
        calls = {"n": 0}

        async def flaky_scroll(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("scroll failed")
            return await real_scroll(*args, **kwargs)

        storage.scroll_points = flaky_scroll
        await indexer.update_document_tags_in_index(doc)

        content = _summary_point(storage, doc.id).payload["content"]
        assert "zxqnewtag" in content
        assert vocab._get_doc_freq() == before
        assert vocab.get_codebase_doc_count(DOCS_CODEBASE_ID) == count
