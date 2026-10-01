"""Document operations must never query or mutate an incompatible generation."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from mcp_docs.embedding import document_embedding_text
from mcp_docs.indexing.indexer import DocumentIndexer
from mcp_docs.models import ExtractionStatus
from mcp_docs.search.engine import DocumentSearchEngine


@pytest.fixture
def routing(monkeypatch):
    events = []

    @asynccontextmanager
    async def lock(storage, logical_name):
        assert logical_name == "documents"
        events.append("locked")
        try:
            yield
        finally:
            events.append("unlocked")

    async def ensure(storage, logical_name, embedder, resolver, *, lock_held=False):
        assert logical_name == "documents"
        assert resolver is document_embedding_text
        if lock_held:
            assert events[-1] == "locked"
        events.append("resolved")
        return SimpleNamespace(physical_name="documents_generation_new")

    ensure_mock = AsyncMock(side_effect=ensure)
    monkeypatch.setattr("mcp_docs.embedding.ensure_embedding_collection", ensure_mock)
    monkeypatch.setattr("mcp_docs.embedding.embedding_collection_lock", lock)
    return events, ensure_mock


def components():
    storage = AsyncMock()
    storage.scroll_points.return_value = []
    embedder = AsyncMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    embedder.embed_single_cached.return_value = [0.1, 0.2]
    vocab = MagicMock()
    vocab.tokenize.return_value = ["example"]
    vocab.vectorize_document.return_value = SimpleNamespace(indices=[1], values=[1.0])
    return storage, embedder, vocab


async def test_child_task_does_not_reuse_parent_generation_after_exit(routing):
    events, ensure = routing
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(MagicMock(), storage, embedder, vocab, "documents")
    parent_finished = asyncio.Event()

    async def child_write():
        await parent_finished.wait()
        assert indexer.collection_name == "documents"
        await indexer.delete_document_index(uuid4())

    async with indexer.collection_operation(write=True):
        child = asyncio.create_task(child_write())
    parent_finished.set()
    await child
    assert ensure.await_count == 2
    assert events == ["locked", "resolved", "unlocked"] * 2
    assert storage.delete_by_filter.await_args.args[0] == "documents_generation_new"


async def test_search_resolves_generation_before_embedding(routing):
    events, ensure = routing
    storage, embedder, vocab = components()
    engine = DocumentSearchEngine(storage, embedder, vocab, "documents")
    engine._searcher = AsyncMock()
    engine._searcher.search.return_value = []

    async def embed(query, *, role):
        assert role == "query"
        assert events == ["resolved"]
        return [0.1, 0.2]

    embedder.embed_single_cached.side_effect = embed
    assert await engine.search("example") == []
    assert engine._searcher.search.await_args.kwargs["collection"] == "documents_generation_new"
    embedder.embed_single_cached.assert_awaited_once_with("example", role="query")
    assert engine.collection_name == "documents"
    ensure.assert_awaited_once()


async def test_migration_failure_prevents_search(monkeypatch):
    storage, embedder, vocab = components()
    engine = DocumentSearchEngine(storage, embedder, vocab, "documents")
    engine._searcher = AsyncMock()
    monkeypatch.setattr(
        "mcp_docs.embedding.ensure_embedding_collection",
        AsyncMock(side_effect=RuntimeError("incomplete migration")),
    )
    with pytest.raises(RuntimeError, match="incomplete migration"):
        await engine.search("example")
    embedder.embed_single_cached.assert_not_awaited()
    engine._searcher.search.assert_not_awaited()
    assert engine.collection_name == "documents"


async def test_incremental_index_migrates_indexed_missing_source(
    routing, document_store, sample_text
):
    events, ensure = routing
    doc = document_store.register(sample_text)
    document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)
    before = document_store.read(doc.id)
    sample_text.unlink()
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")

    assert await indexer.index_all() == {"indexed": 0, "total": 1}
    ensure.assert_awaited_once()
    assert document_store.read(doc.id) == before
    assert events == ["locked", "resolved", "unlocked"]
    vocab.register_codebase.assert_not_called()
    vocab.update_codebase_incremental.assert_not_called()
    embedder.embed_batch.assert_not_awaited()


async def test_index_document_persists_input_and_uses_one_locked_generation(
    routing, document_store, sample_text
):
    events, ensure = routing
    doc = document_store.register(sample_text)
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")

    async def upsert(collection, points):
        assert events == ["locked", "resolved"]
        assert collection == "documents_generation_new"
        assert [p.payload["embedding_text"] for p in points] == [
            doc.filename,
            "Previously extracted text",
        ]

    storage.upsert_batch.side_effect = upsert
    assert await indexer.index_document(doc.id, "Previously extracted text") == 2
    assert storage.delete_by_filter.await_args.args[0] == "documents_generation_new"
    ensure.assert_awaited_once()
    assert events == ["locked", "resolved", "unlocked"]
    assert indexer.collection_name == "documents"
    embedder.embed_batch.assert_awaited_once_with([doc.filename, "Previously extracted text"])


@pytest.mark.parametrize(
    "method", ["update_document_tags_in_index", "update_document_filename_in_index"]
)
async def test_summary_refresh_binds_generation(method, routing, document_store, sample_text):
    doc = document_store.register(sample_text)
    doc = document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")
    await getattr(indexer, method)(doc)
    assert storage.update_payload.await_args.args[0] == "documents_generation_new"
    collection, points = storage.upsert_batch.await_args.args
    assert collection == "documents_generation_new"
    assert points[0].payload["embedding_text"] == points[0].payload["content"]


async def test_delete_and_chunk_reads_bind_generation(routing):
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(MagicMock(), storage, embedder, vocab, "documents")
    await indexer.delete_document_index(uuid4())
    assert storage.delete_by_filter.await_args.args[0] == "documents_generation_new"
    engine = DocumentSearchEngine(storage, embedder, vocab, "documents")
    assert await engine.get_document_chunks(uuid4()) == []
    assert storage.scroll_points.await_args.args[0] == "documents_generation_new"


async def test_similarity_freezes_target_for_vector_read_and_query(routing):
    storage, embedder, vocab = components()
    storage.scroll_points.return_value = [{"document_id": str(uuid4())}]
    storage.get_client.return_value.scroll.return_value = (
        [SimpleNamespace(vector={"dense": [0.1, 0.2]})],
        None,
    )
    storage.query_dense.return_value = []
    engine = DocumentSearchEngine(storage, embedder, vocab, "documents")
    assert await engine.find_similar(uuid4()) == []
    assert storage.get_client.return_value.scroll.await_args.args[0] == "documents_generation_new"
    assert storage.query_dense.await_args.kwargs["collection"] == "documents_generation_new"


@pytest.mark.parametrize("kind", ["document", "doc_chunk"])
async def test_document_text_uses_persisted_content_without_source(kind):
    assert (
        await document_embedding_text(
            {"type": kind, "content": "Full stored content", "path": "/missing/source"}
        )
        == "Full stored content"
    )


async def test_shared_payload_resolves_via_authoritative_store(monkeypatch):
    resolver = AsyncMock(return_value="Full untruncated glossary definition")
    monkeypatch.setattr("mcp_docs.embedding.resolve_shared_embedding_text", resolver)
    payload = {"type": "glossary", "glossary_id": str(uuid4())}
    assert await document_embedding_text(payload) == "Full untruncated glossary definition"
    resolver.assert_awaited_once_with(payload)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"type": "note", "title": "Retained note", "tags": ["tag"]}, "Retained note\nTags: tag"),
        ({"type": "chunk", "note_id": "note", "content": "Full note chunk"}, "Full note chunk"),
        ({"type": "fact", "content": "Retained fact"}, "Retained fact"),
    ],
)
async def test_docs_first_shared_legacy_payloads(payload, expected):
    assert await document_embedding_text(payload) == expected


async def test_docs_first_glossary_uses_full_definition(tmp_path, monkeypatch):
    from vector_core.glossary.store import GlossaryStore
    from vector_core.settings import settings
    from vector_core.utils.hashing import hash_content

    monkeypatch.setattr(settings, "shared_data_dir", tmp_path)
    store = GlossaryStore()
    definition = "Complete definition " * 150
    try:
        entry = store.create("TERM", "Expansion", definition)
        payload = {
            "type": "glossary",
            "glossary_id": str(entry.id),
            "term": entry.term,
            "expansion": entry.expansion,
            "definition": definition[:2000],
            "domain": entry.domain,
            "aliases": entry.aliases,
            "entry_hash": hash_content(f"{entry.term}|{entry.expansion}|{definition}"),
        }
        assert await document_embedding_text(payload) == f"TERM Expansion {definition}"
    finally:
        store.close()


async def test_real_migration_retains_unavailable_document_and_index_state(
    document_store, sample_text, tmp_path, monkeypatch
):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import PointStruct, SparseVector
    from vector_core import EmbeddingClient, QdrantStorage
    from vector_core.settings import settings
    from vector_core.storage.embedding_migration import (
        EmbeddingMigrationError,
        active_embedding_collection,
    )

    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    doc = document_store.register(sample_text)
    document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)
    before = document_store.read(doc.id)
    sample_text.unlink()
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    embedder = EmbeddingClient(model="replacement", dim=3)
    embedder.embed_single = AsyncMock(return_value=[1.0, 0.0, 0.0])
    embedder.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[1.0, 0.0, 0.0] for _ in texts]
    )
    vocab = MagicMock()
    try:
        await storage.create_collection("documents", dense_dim=2)
        raw = await storage.get_client()
        points = [
            PointStruct(
                id=number,
                vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={
                    "type": kind,
                    "document_id": str(doc.id),
                    "content": content,
                    "doc_hash": "unchanged",
                    "content_hash": doc.content_hash,
                    "path": str(sample_text),
                    "chunk_index": 0,
                },
            )
            for number, kind, content in [
                (1, "document", "Retained summary"),
                (2, "doc_chunk", "Retained full chunk"),
            ]
        ]
        await raw.upsert("documents", points, wait=True)
        original, _ = await raw.scroll("documents", with_vectors=True)
        indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")
        assert await indexer.index_all() == {"indexed": 0, "total": 1}
        target = await active_embedding_collection(storage, "documents")
        assert target != "documents"
        migrated, _ = await raw.scroll(target, with_vectors=True)
        assert {p.id for p in migrated} == {0, 1, 2}
        assert document_store.read(doc.id) == before
        assert (await raw.scroll("documents", with_vectors=True))[0] == original
        embedder.embed_all.assert_awaited_once_with(
            ["Retained summary", "Retained full chunk"], role="document"
        )
        for point in migrated:
            if point.id:
                assert len(point.vector["dense"]) == 3
                assert point.vector["sparse"] == original[0].vector["sparse"]
                assert point.payload["doc_hash"] == "unchanged"

        # A second configured identity builds from the current complete generation.
        newer = EmbeddingClient(model="same-dimension-change", dim=3)
        newer.embed_single = AsyncMock(return_value=[0.0, 1.0, 0.0])
        newer.embed_all = AsyncMock(
            side_effect=lambda texts, **kwargs: [[0.0, 1.0, 0.0] for _ in texts]
        )
        engine = DocumentSearchEngine(storage, newer, vocab, "documents")
        assert len(await engine.get_document_chunks(doc.id)) == 1
        assert await active_embedding_collection(storage, "documents") != target
        with pytest.raises(EmbeddingMigrationError, match="superseded"):
            await indexer.delete_document_index(doc.id)
        assert document_store.read(doc.id) == before
        vocab.update_codebase_incremental.assert_not_called()
        await newer.close()
    finally:
        await embedder.close()
        await storage.close()


@pytest.mark.parametrize("tool", ["update_document_tags", "delete_document"])
async def test_source_metadata_unchanged_when_generation_unavailable(
    tool, document_store, sample_text, monkeypatch
):
    from mcp_docs.tools import documents

    doc = document_store.register(sample_text)
    indexer = AsyncMock()
    indexer.collection_operation = MagicMock(side_effect=RuntimeError("generation unavailable"))
    monkeypatch.setattr(documents, "get_document_store", lambda: document_store)
    monkeypatch.setattr(documents, "get_document_indexer", AsyncMock(return_value=indexer))
    args = {"document_id": str(doc.id)}
    if tool == "update_document_tags":
        args["tags"] = ["changed"]
    with pytest.raises(RuntimeError, match="generation unavailable"):
        await getattr(documents, tool)(**args)
    assert document_store.read(doc.id) == doc


@pytest.mark.parametrize("tool", ["update_document_tags", "delete_document"])
@pytest.mark.parametrize("document_id", ["invalid-uuid", str(uuid4())])
async def test_invalid_document_request_never_initializes_migration(
    tool, document_id, document_store, monkeypatch
):
    from mcp_docs.tools import documents

    get_indexer = AsyncMock()
    monkeypatch.setattr(documents, "get_document_store", lambda: document_store)
    monkeypatch.setattr(documents, "get_document_indexer", get_indexer)
    args = {"document_id": document_id}
    if tool == "update_document_tags":
        args["tags"] = ["tag"]
    await getattr(documents, tool)(**args)
    get_indexer.assert_not_awaited()


@pytest.mark.parametrize("tool", ["update_document_tags", "delete_document"])
async def test_source_and_index_mutations_share_generation_lock(
    tool, document_store, sample_text, monkeypatch
):
    from mcp_docs.tools import documents

    doc = document_store.register(sample_text)
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")
    lock = asyncio.Lock()
    updating = asyncio.Event()
    migration_attempted = asyncio.Event()
    observed = []

    @asynccontextmanager
    async def held(storage, logical_name):
        async with lock:
            yield

    monkeypatch.setattr("mcp_docs.embedding.embedding_collection_lock", held)
    monkeypatch.setattr(
        "mcp_docs.embedding.ensure_embedding_collection",
        AsyncMock(return_value=SimpleNamespace(physical_name="documents_generation_new")),
    )
    monkeypatch.setattr(documents, "get_document_store", lambda: document_store)
    monkeypatch.setattr(documents, "get_document_indexer", AsyncMock(return_value=indexer))
    monkeypatch.setattr(documents, "get_integrity_manager", MagicMock())

    async def write_index(*args, **kwargs):
        assert lock.locked()
        updating.set()
        await migration_attempted.wait()
        assert observed == []  # Migration cannot snapshot half the mutation.

    if tool == "delete_document":
        storage.delete_by_filter.side_effect = write_index
    else:
        storage.update_payload.side_effect = write_index

    async def migrate():
        await updating.wait()
        migration_attempted.set()
        async with lock:
            observed.append(document_store.read(doc.id))

    migration = asyncio.create_task(migrate())
    if tool == "delete_document":
        await documents.delete_document(str(doc.id))
    else:
        await documents.update_document_tags(str(doc.id), ["updated"])
    await migration
    assert observed == [None] if tool == "delete_document" else observed[0].tags == ["updated"]


@pytest.mark.parametrize("tool", ["move_file", "rename_directory", "move_directory"])
async def test_filesystem_and_registry_move_under_generation_lock(
    tool, document_store, tmp_path, monkeypatch
):
    from mcp_docs.tools import filesystem

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "document.txt"
    source.write_text("Document content")
    document_store.add_root(str(tmp_path))
    doc = document_store.register(source)
    storage, embedder, vocab = components()
    storage.scroll_points.return_value = [{"path": str(source)}]
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")
    held = False

    @asynccontextmanager
    async def lock(storage, logical_name):
        nonlocal held
        held = True
        try:
            yield
        finally:
            held = False

    monkeypatch.setattr("mcp_docs.embedding.embedding_collection_lock", lock)
    monkeypatch.setattr(
        "mcp_docs.embedding.ensure_embedding_collection",
        AsyncMock(return_value=SimpleNamespace(physical_name="documents_generation_new")),
    )
    monkeypatch.setattr(filesystem, "get_document_store", lambda: document_store)
    monkeypatch.setattr(filesystem, "get_document_indexer", AsyncMock(return_value=indexer))
    monkeypatch.setattr(filesystem, "get_document_processor", AsyncMock(return_value=AsyncMock()))
    move = filesystem.shutil.move

    def checked_move(*args):
        assert held
        return move(*args)

    monkeypatch.setattr(filesystem.shutil, "move", checked_move)

    async def checked_update(*args, **kwargs):
        assert held
        assert document_store.read(doc.id).path != str(source)
        assert not source.exists()

    storage.update_payload.side_effect = checked_update
    if tool == "move_file":
        result = await filesystem.move_file(str(source), str(tmp_path / "document.txt"))
    elif tool == "rename_directory":
        result = await filesystem.rename_directory(str(source_dir), "renamed")
    else:
        result = await filesystem.move_directory(str(source_dir), str(tmp_path / "moved"))
    assert result["success"] is True
    storage.update_payload.assert_awaited_once()
    assert not held


async def test_duplicate_relocation_preflights_before_registry_change(
    document_store, sample_text, tmp_path, monkeypatch
):
    from mcp_docs.tools import documents

    doc = document_store.register(sample_text)
    relocated = tmp_path / "relocated.txt"
    relocated.write_bytes(sample_text.read_bytes())
    indexer = AsyncMock()
    indexer.collection_operation = MagicMock(side_effect=RuntimeError("superseded"))
    monkeypatch.setattr(documents, "get_document_store", lambda: document_store)
    monkeypatch.setattr(documents, "get_document_indexer", AsyncMock(return_value=indexer))
    with pytest.raises(RuntimeError, match="superseded"):
        await documents.register_document(str(relocated))
    assert document_store.read(doc.id) == doc


@pytest.mark.parametrize("all_roots", [False, True])
async def test_scan_locks_registry_changes_but_enqueues_after_unlock(
    all_roots, document_store, sample_text, monkeypatch
):
    from mcp_docs.tools import roots

    root = document_store.add_root(str(sample_text.parent))
    storage, embedder, vocab = components()
    indexer = DocumentIndexer(document_store, storage, embedder, vocab, "documents")
    held = False

    @asynccontextmanager
    async def lock(storage, logical_name):
        nonlocal held
        held = True
        try:
            yield
        finally:
            held = False

    async def scan(*args, **kwargs):
        assert held
        await kwargs["enqueue_callback"](uuid4(), sample_text)
        await kwargs["delete_callback"](uuid4())
        result = SimpleNamespace(to_dict=lambda: {"scanned": True})
        return [result] if all_roots else result

    async def enqueue(*args):
        assert not held

    scanner = AsyncMock()
    scanner.scan_root.side_effect = scan
    scanner.scan_all_roots.side_effect = scan
    processor = AsyncMock()
    processor.enqueue.side_effect = enqueue
    monkeypatch.setattr("mcp_docs.embedding.embedding_collection_lock", lock)
    monkeypatch.setattr(
        "mcp_docs.embedding.ensure_embedding_collection",
        AsyncMock(return_value=SimpleNamespace(physical_name="documents_generation_new")),
    )
    monkeypatch.setattr(roots, "get_document_store", lambda: document_store)
    monkeypatch.setattr(roots, "get_document_indexer", AsyncMock(return_value=indexer))
    monkeypatch.setattr(roots, "get_document_scanner", AsyncMock(return_value=scanner))
    monkeypatch.setattr(roots, "get_document_processor", AsyncMock(return_value=processor))
    if all_roots:
        await roots.scan_all_roots()
    else:
        await roots.scan_document_root(root.path)
    processor.enqueue.assert_awaited_once()


async def test_deleted_duplicate_is_enqueued_as_new_registration(
    document_store, sample_text, tmp_path, monkeypatch
):
    from mcp_docs.tools import documents

    old = document_store.register(sample_text)
    relocated = tmp_path / "new.txt"
    relocated.write_bytes(sample_text.read_bytes())

    @asynccontextmanager
    async def operation(**kwargs):
        document_store.delete(old.id)
        yield

    indexer = AsyncMock()
    indexer.collection_operation = operation
    processor = AsyncMock()
    monkeypatch.setattr(documents, "get_document_store", lambda: document_store)
    monkeypatch.setattr(documents, "get_document_indexer", AsyncMock(return_value=indexer))
    monkeypatch.setattr(documents, "get_document_processor", AsyncMock(return_value=processor))
    result = await documents.register_document(str(relocated))
    new = document_store.get_by_path(str(relocated))
    assert new.id != old.id
    assert not result.get("already_registered")
    processor.enqueue.assert_awaited_once_with(new.id, relocated)
    indexer.update_document_path_in_index.assert_not_awaited()


async def test_deferred_enqueue_continues_after_failure(tmp_path):
    from mcp_docs.tools.roots import _enqueue_pending

    processor = AsyncMock()
    processor.enqueue.side_effect = [RuntimeError("queue failure"), None]
    pending = [(uuid4(), tmp_path / "first"), (uuid4(), tmp_path / "second")]
    result = SimpleNamespace(root_path=str(tmp_path), errors=[])
    await _enqueue_pending(processor, pending, [result])
    assert processor.enqueue.await_count == 2
    assert result.errors == [f"{tmp_path / 'first'}: enqueue failed: queue failure"]


async def test_deferred_queue_rejection_is_visible(tmp_path):
    from mcp_docs.tools.roots import _enqueue_pending

    processor = AsyncMock()
    processor.enqueue.side_effect = [False, True]
    result = SimpleNamespace(root_path=str(tmp_path), errors=[])
    await _enqueue_pending(
        processor, [(uuid4(), tmp_path / "one"), (uuid4(), tmp_path / "two")], [result]
    )
    assert processor.enqueue.await_count == 2
    assert len(result.errors) == 1
    assert "Processing queue rejected" in result.errors[0]


async def test_deferred_enqueue_errors_respect_scanner_cap(tmp_path, caplog):
    from mcp_docs.scanning.scanner import MAX_ERRORS
    from mcp_docs.tools.roots import _enqueue_pending

    processor = AsyncMock()
    processor.enqueue.return_value = False
    result = SimpleNamespace(root_path=str(tmp_path), errors=["prior"] * (MAX_ERRORS - 1))
    pending = [(uuid4(), tmp_path / str(i)) for i in range(3)]
    await _enqueue_pending(processor, pending, [result])
    assert processor.enqueue.await_count == 3
    assert len(result.errors) == MAX_ERRORS
    assert len(caplog.records) == 1


def test_processing_attempt_survives_metadata_changes(document_store, sample_text):
    doc = document_store.register(sample_text)
    token = document_store.start_processing_attempt(doc.id)
    document_store.update(doc.id, title="Unrelated metadata change")
    assert document_store.fail_processing_attempt(doc.id, token, "extraction failure")
    assert document_store.read(doc.id).extraction_status == ExtractionStatus.FAILED


def test_new_processing_attempt_rejects_old_failure(document_store, sample_text):
    doc = document_store.register(sample_text)
    old = document_store.start_processing_attempt(doc.id)
    new = document_store.start_processing_attempt(doc.id)
    assert old != new
    assert not document_store.fail_processing_attempt(doc.id, old, "stale failure")
    assert document_store.read(doc.id).extraction_status == ExtractionStatus.PROCESSING
    assert document_store.fail_processing_attempt(doc.id, new, "current failure")


async def test_superseded_worker_does_not_report_extraction_completed(document_store, sample_text):
    from mcp_docs.processing.queue import DocumentProcessor, ProcessingStatus, ProcessingTask

    doc = document_store.register(sample_text)
    indexer = AsyncMock()
    indexer.collection_operation = MagicMock(side_effect=RuntimeError("superseded"))
    extractor = MagicMock()
    extractor.extract.return_value = SimpleNamespace(
        text="Extracted text", title="New title", page_count=1, word_count=2
    )
    processor = DocumentProcessor(document_store, indexer=indexer, extractor=extractor)
    try:
        result = await processor._process(ProcessingTask(doc.id, sample_text))
        assert result.status == ProcessingStatus.FAILED
        current = document_store.read(doc.id)
        assert current.title == doc.title
        assert current.extraction_status == ExtractionStatus.FAILED
        assert current.extraction_error == "superseded"
        indexer.index_document.assert_not_awaited()
    finally:
        processor._executor.shutdown(wait=True)


async def test_stale_worker_failure_preserves_newer_indexed_attempt(document_store, sample_text):
    from mcp_docs.processing.queue import DocumentProcessor, ProcessingStatus, ProcessingTask

    doc = document_store.register(sample_text)
    indexer = AsyncMock()

    @asynccontextmanager
    async def superseded(**kwargs):
        document_store.update(doc.id, extraction_status=ExtractionStatus.INDEXED)
        raise RuntimeError("superseded")
        yield

    indexer.collection_operation = superseded
    extractor = MagicMock()
    processor = DocumentProcessor(document_store, indexer=indexer, extractor=extractor)
    try:
        result = await processor._process(ProcessingTask(doc.id, sample_text))
        assert result.status == ProcessingStatus.FAILED
        assert document_store.read(doc.id).extraction_status == ExtractionStatus.INDEXED
        assert document_store.read(doc.id).extraction_error is None
    finally:
        processor._executor.shutdown(wait=True)
