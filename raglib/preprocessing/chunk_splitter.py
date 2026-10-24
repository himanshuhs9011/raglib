"""Chunk splitting utilities for preparing long documents for retrieval.

This module re-exports the canonical ChunkSplitter from raglib.loaders so
preprocessing and loaders share one implementation (fixed / recursive / llm).
"""

from __future__ import annotations

import warnings

from raglib.loaders.chunk_splitter import (
    CHUNKING_STRATEGY_ALIASES,
    DEFAULT_RECURSIVE_SEPARATORS,
    SUPPORTED_CHUNKING_STRATEGIES,
    ChunkSplitter,
)

__all__ = [
    "ChunkSplitter",
    "SUPPORTED_CHUNKING_STRATEGIES",
    "CHUNKING_STRATEGY_ALIASES",
    "DEFAULT_RECURSIVE_SEPARATORS",
]

warnings.warn(
    "raglib.preprocessing is deprecated; import ChunkSplitter from raglib.loaders instead.",
    DeprecationWarning,
    stacklevel=2,
)
