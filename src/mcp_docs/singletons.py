"""Singleton instances for mcp-docs services.

Provides thread-safe and async-safe singleton patterns for:
- DocumentStore (sync)
- GlossaryStore (sync)
- SourceIntegrityManager (sync)
- DocumentProcessor (async)
- DocumentScanner (async)
- DocumentIndexer (async)
- DocumentSearchEngine (async)
- GlossaryIndexer (async)
- GlossaryToolHelper (async)
"""

from vector_core import AsyncSingleton, SyncSingleton
from vector_core.facts import FactStore, SourceIntegrityManager
from vector_core.glossary import GlossaryIndexer, GlossaryStore, GlossaryToolHelper

from mcp_docs.indexing import DocumentIndexer
from mcp_docs.processing import DocumentProcessor
from mcp_docs.scanning import DocumentScanner
from mcp_docs.search import DocumentSearchEngine
from mcp_docs.settings import settings
from mcp_docs.storage.database import DocumentStore

# Sync singletons using SyncSingleton pattern from vector-core
# Provides thread-safe initialization with reentrant lock
_document_store: SyncSingleton[DocumentStore] = SyncSingleton("document_store")
_glossary_store: SyncSingleton[GlossaryStore] = SyncSingleton("glossary_store")
_integrity_manager: SyncSingleton[SourceIntegrityManager] = SyncSingleton(
    "integrity_manager"
)

# Async singletons (use AsyncSingleton pattern)
_document_processor: AsyncSingleton[DocumentProcessor] = AsyncSingleton("document_processor")
_document_scanner: AsyncSingleton[DocumentScanner] = AsyncSingleton("document_scanner")
_document_indexer: AsyncSingleton[DocumentIndexer] = AsyncSingleton("document_indexer")
_search_engine: AsyncSingleton[DocumentSearchEngine] = AsyncSingleton("search_engine")
_glossary_indexer: AsyncSingleton[GlossaryIndexer] = AsyncSingleton("glossary_indexer")
_glossary_helper: AsyncSingleton[GlossaryToolHelper] = AsyncSingleton("glossary_helper")


def get_document_store() -> DocumentStore:
    """Get or create DocumentStore instance (thread-safe via SyncSingleton)."""
    return _document_store.get(DocumentStore)


async def get_document_processor() -> DocumentProcessor:
    """Get or create DocumentProcessor instance (async-safe)."""
    async def _create_processor() -> DocumentProcessor:
        processor = DocumentProcessor(document_store=get_document_store())
        await processor.start()
        return processor

    return await _document_processor.get(_create_processor)


async def get_document_scanner() -> DocumentScanner:
    """Get or create DocumentScanner instance (async-safe)."""
    return await _document_scanner.get(
        lambda: DocumentScanner(document_store=get_document_store())
    )


async def get_search_engine() -> DocumentSearchEngine:
    """Get or create DocumentSearchEngine instance (async-safe)."""
    return await _search_engine.get(DocumentSearchEngine)


async def get_document_indexer() -> DocumentIndexer:
    """Get or create DocumentIndexer instance (async-safe)."""
    return await _document_indexer.get(
        lambda: DocumentIndexer(document_store=get_document_store())
    )


def get_glossary_store() -> GlossaryStore:
    """Get or create GlossaryStore instance (thread-safe via SyncSingleton)."""
    return _glossary_store.get(GlossaryStore)


async def get_glossary_indexer() -> GlossaryIndexer:
    """Get or create GlossaryIndexer instance (async-safe).

    Reuses storage/embedder from DocumentIndexer for resource efficiency.
    """
    async def _create_glossary_indexer() -> GlossaryIndexer:
        # Get document indexer to reuse its storage/embedder
        doc_indexer = await get_document_indexer()
        await doc_indexer._ensure_components()

        return GlossaryIndexer(
            glossary_store=get_glossary_store(),
            collection_name=settings.collection_name,
            storage=doc_indexer.storage,
            embedder=doc_indexer.embedder,
            global_vocab=doc_indexer.global_vocab,
        )

    return await _glossary_indexer.get(_create_glossary_indexer)


async def get_glossary_helper() -> GlossaryToolHelper:
    """Get or create GlossaryToolHelper instance (async-safe)."""
    async def _create_helper() -> GlossaryToolHelper:
        indexer = await get_glossary_indexer()
        return GlossaryToolHelper(
            store=get_glossary_store(),
            indexer=indexer,
        )

    return await _glossary_helper.get(_create_helper)


def get_integrity_manager() -> SourceIntegrityManager:
    """
    Get or create SourceIntegrityManager instance (thread-safe via SyncSingleton).

    Uses the shared facts.db from vector-core for marking document sources.
    """
    return _integrity_manager.get(
        lambda: SourceIntegrityManager(
            fact_store=FactStore(db_path=settings.facts_db_path),
        )
    )


async def cleanup_async_resources() -> None:
    """Clean up async resources (called during server shutdown)."""
    await _document_processor.close(lambda p: p.stop())
    await _document_scanner.close(None)
    await _document_indexer.close(lambda i: i.close())
    await _search_engine.close(lambda e: e.close())
    await _glossary_indexer.close(lambda i: i.close())
    await _glossary_helper.close(None)
