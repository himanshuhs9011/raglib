"""raglib's single entry point, ``RAG``, split into one module per concern.

``core`` holds the constructor; each mixin module holds one area (persistence, graph, chunking,
ingestion, permissions, guards, querying, providers, extras). Import from here:
``from raglib.facade import RAG``.
"""

from __future__ import annotations

from raglib.facade.chunking import unwrap_embedding
from raglib.facade.core import (
    OPENROUTER_FREE_RPM,
    ORCHESTRATORS,
    RAG,
    RAG_TYPE_ALIASES,
)
from raglib.facade.graph import GRAPH_GLOBAL_OPTIONS, GRAPH_INDEXER_OPTIONS, GRAPH_LOCAL_OPTIONS

RAG.__module__ = "raglib.facade"  # the public name, wherever the class body lives

__all__ = [
    "RAG",
    "RAG_TYPE_ALIASES",
    "ORCHESTRATORS",
    "OPENROUTER_FREE_RPM",
    "GRAPH_INDEXER_OPTIONS",
    "GRAPH_LOCAL_OPTIONS",
    "GRAPH_GLOBAL_OPTIONS",
    "unwrap_embedding",
]
