"""Bind document operations to one compatible embedding generation."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from vector_core.storage.embedding_migration import (
    embedding_collection_lock,
    ensure_embedding_collection,
    resolve_shared_embedding_text,
)

P = ParamSpec("P")
R = TypeVar("R")


async def document_embedding_text(payload: dict[str, Any]) -> str:
    """Recover exact stored document text without reopening source files."""
    return await resolve_shared_embedding_text(payload)


class EmbeddingCollection:
    """Operation-local routing shared by indexing and search."""

    def __init__(self) -> None:
        self._active_collection: ContextVar[tuple[asyncio.Task[Any] | None, str] | None] = (
            ContextVar("document_embedding_collection", default=None)
        )

    def active_collection_name(self) -> str | None:
        """Ignore bindings inherited by a different asyncio task."""
        active = self._active_collection.get()
        if active is not None and active[0] is asyncio.current_task():
            return active[1]
        return None

    @asynccontextmanager
    async def collection_operation(self, *, write: bool) -> AsyncIterator[None]:
        # Nested indexing helpers stay on the outer operation's physical target.
        if self.active_collection_name() is not None:
            yield
            return
        owner: Any = self
        await owner._ensure_components()
        logical_name = owner.collection_name

        async def bind(*, lock_held: bool) -> Any:
            return await ensure_embedding_collection(
                owner.storage,
                logical_name,
                owner.embedder,
                document_embedding_text,
                lock_held=lock_held,
            )

        if write:
            async with embedding_collection_lock(owner.storage, logical_name):
                generation = await bind(lock_held=True)
                token = self._active_collection.set(
                    (asyncio.current_task(), generation.physical_name)
                )
                try:
                    yield
                finally:
                    self._active_collection.reset(token)
        else:
            generation = await bind(lock_held=False)
            token = self._active_collection.set((asyncio.current_task(), generation.physical_name))
            try:
                yield
            finally:
                self._active_collection.reset(token)


def embedding_operation(
    *, write: bool = False
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Resolve once per operation; serialize mutations with migration."""

    def decorate(method: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            owner: Any = args[0]
            async with owner.collection_operation(write=write):
                return await method(*args, **kwargs)

        return wrapped

    return decorate
