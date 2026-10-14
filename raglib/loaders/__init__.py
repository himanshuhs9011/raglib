"""Loader package exports for ingestion and chunking."""

from __future__ import annotations

import logging

from raglib.loaders.chunk_splitter import (
    CHUNKING_STRATEGY_ALIASES,
    DEFAULT_RECURSIVE_SEPARATORS,
    SUPPORTED_CHUNKING_STRATEGIES,
    ChunkSplitter,
)
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.loaders.pdf_loader import PDFLoader
from raglib.loaders.pptx_loader import PPTXLoader
from raglib.loaders.text_loader import TextLoader
from raglib.loaders.web_loader import WebLoader
from raglib.loaders.word_loader import WordLoader

logger = logging.getLogger(__name__)

__all__ = [
    "SourceInput",
    "DocumentLoader",
    "ChunkSplitter",
    "SUPPORTED_CHUNKING_STRATEGIES",
    "CHUNKING_STRATEGY_ALIASES",
    "DEFAULT_RECURSIVE_SEPARATORS",
    "TextLoader",
    "WordLoader",
    "PPTXLoader",
    "PDFLoader",
    "WebLoader",
]
