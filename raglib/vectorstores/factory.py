"""Factory for selecting vector store backends."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from raglib.vectorstores.base_store import BaseVectorStore
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)

CHROMA_ALIASES = {"chroma", "chromadb"}
MEMORY_ALIASES = {"memory", "in_memory", "inmemory", "mock"}


class VectorStoreFactory:
    """Build vector store instances from a backend name or a prebuilt instance."""

    @staticmethod
    def build(
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseVectorStore:
        """Return the configured backend (default: in-memory).

        Failures are raised, never silently replaced by another backend. Pass
        ``vector_db_kwargs={"fallback_to_memory": True}`` to opt into a fallback.
        """

        if isinstance(vector_db, BaseVectorStore):
            logger.info("Using prebuilt vector store instance: %s", type(vector_db).__name__)
            return vector_db

        backend = (vector_db or "memory").strip().lower()
        options = dict(vector_db_kwargs or {})
        fallback_to_memory = bool(options.pop("fallback_to_memory", False))
        options.pop("recover_with_fresh_persist", None)  # v1 option, no longer used

        if backend in MEMORY_ALIASES:
            return InMemoryVectorStore()

        if backend in CHROMA_ALIASES:
            try:
                from raglib.vectorstores.chroma_store import ChromaVectorStore

                return ChromaVectorStore(**options)
            except Exception as exc:  # noqa: BLE001
                if not fallback_to_memory:
                    raise
                logger.warning("Chroma unavailable (%s); fallback_to_memory=True so using InMemoryVectorStore", exc)
                return InMemoryVectorStore()

        raise ValueError(
            f"Unknown vector_db '{backend}'. Supported values: ['chroma', 'memory'] "
            "or pass a BaseVectorStore instance."
        )
