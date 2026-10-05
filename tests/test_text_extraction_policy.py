"""Text extraction changes invalidate only their prior successful cache keys."""

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from vector_core.embeddings.sparse import SparseVector

from mcp_docs.extraction import extract_content
from mcp_docs.indexing.indexer import SOURCE_LAYOUT_VERSION, DocumentIndexer
from mcp_docs.models import DocumentType, ExtractionStatus

from .test_vocabulary_accounting import FakeEmbedder, FakeStorage


def legacy_hash(document):
    """The successful layout-2 cache key before whole-file text extraction."""
    content = (
        f"2:{document.id}:{document.content_hash}:{document.title or ''}:{','.join(document.tags)}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@pytest.mark.parametrize("doc_type", list(DocumentType))
def test_extraction_policy_changes_only_text_hashes(document_store, sample_text, doc_type):
    document = document_store.register(sample_text).model_copy(update={"doc_type": doc_type.value})
    indexer = DocumentIndexer(document_store, MagicMock(), MagicMock(), MagicMock())
    assert SOURCE_LAYOUT_VERSION == 2
    if doc_type in {DocumentType.TXT, DocumentType.MD}:
        assert indexer._doc_hash(document) != legacy_hash(document)
    else:
        assert indexer._doc_hash(document) == legacy_hash(document)


@pytest.mark.usefixtures("embedding_generation")
async def test_incremental_text_repair_runs_once_without_reextracting_pdf(
    document_store, temp_dir, monkeypatch
):
    storage = FakeStorage()
    vocabulary = MagicMock()
    vocabulary.tokenize.return_value = []
    vocabulary.vectorize_document.side_effect = lambda _: SparseVector(indices=[], values=[])
    indexer = DocumentIndexer(document_store, storage, FakeEmbedder(), vocabulary, "policy")
    monkeypatch.setattr(indexer, "ensure_collection", AsyncMock())
    documents = []
    sources = {}
    for extension in ("txt", "md", "pdf"):
        path = temp_dir / ("example." + extension)
        text = f"# Example {extension}\r\n\r\n\r\nSource text with trailing spaces  \r\n"
        path.write_bytes(text.encode("utf-8"))
        document = document_store.register(path)
        document = document_store.update(
            document.id,
            title="Custom title" if extension == "txt" else None,
            word_count=999,
            extraction_status=ExtractionStatus.INDEXED,
        )
        documents.append(document)
        sources[str(document.id)] = text
        points = [
            indexer._create_point(
                "document", document, document.filename, [0.1] * 4, source_layout=2
            ),
            indexer._create_point(
                "doc_chunk",
                document,
                "old normalized body",
                [0.1] * 4,
                chunk_index=0,
                source_layout=2,
            ),
        ]
        for point in points:
            point.payload["doc_hash"] = legacy_hash(document)
        await storage.upsert_batch("policy", points)

    pdf_id = str(documents[-1].id)
    pdf_registration = documents[-1].model_dump()
    pdf_before = {
        identifier: point.model_copy(deep=True)
        for identifier, point in storage.points.items()
        if point.payload["document_id"] == pdf_id
    }

    def extract(path, doc_type):
        assert doc_type in {DocumentType.TXT, DocumentType.MD}
        return extract_content(path, doc_type)

    extraction = MagicMock(side_effect=extract)
    monkeypatch.setattr("mcp_docs.indexing.indexer.extract_content", extraction)
    result = await indexer.index_all(force=False)
    assert result["indexed"] == 2
    assert not result["errors"]
    assert {call.args[1] for call in extraction.call_args_list} == {
        DocumentType.TXT,
        DocumentType.MD,
    }
    assert extraction.call_count == 2
    for point in storage.points.values():
        assert point.payload["source_layout"] == 2
        if point.payload["type"] == "doc_chunk" and point.payload["document_id"] != pdf_id:
            assert point.payload["content"] == sources[point.payload["document_id"]]
    assert {identifier: storage.points[identifier] for identifier in pdf_before} == pdf_before
    current = [document_store.read(doc.id) for doc in documents]
    assert current[0].title == "Custom title"
    assert current[1].title == "Example md"
    assert document_store.read(documents[-1].id).model_dump() == pdf_registration
    for old, updated in zip(documents[:2], current[:2], strict=True):
        assert updated.word_count == len(sources[str(old.id)].split())
        for field in (
            "id",
            "content_hash",
            "path",
            "filename",
            "doc_type",
            "tags",
            "document_root",
        ):
            assert getattr(updated, field) == getattr(old, field)
    assert await indexer._get_indexed_hashes() == {indexer._doc_hash(doc) for doc in current}
    assert (await indexer.index_all(force=False))["skipped"] is True
    assert extraction.call_count == 2
