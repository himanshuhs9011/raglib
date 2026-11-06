"""Vector store package exports."""

from __future__ import annotations

import logging

from raglib.vectorstores.base_store import BaseVectorStore, VectorHit, VectorRecord
from raglib.vectorstores.chroma_store import ChromaVectorStore
from raglib.vectorstores.factory import VectorStoreFactory
from raglib.vectorstores.in_memory_store import InMemoryVectorStore


def __getattr__(name: str):
    """Lazy imports for optional backends (their client libraries are extras)."""

    if name == "QdrantVectorStore":
        from raglib.vectorstores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore
    if name == "PGVectorStore":
        from raglib.vectorstores.pgvector_store import PGVectorStore

        return PGVectorStore
    raise AttributeError(name)


logger = logging.getLogger(__name__)

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "VectorHit",
    "VectorStoreFactory",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "PGVectorStore",
]
