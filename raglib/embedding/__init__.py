"""Embedding package exports."""

from __future__ import annotations

import logging

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.embedding.embedding_factory import EmbeddingFactory
from raglib.embedding.google_embedding import GoogleEmbedding
from raglib.embedding.huggingface_embedding import HuggingFaceEmbedding
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.embedding.ollama_embedding import OllamaEmbedding
from raglib.embedding.openai_embedding import OpenAIEmbedding

logger = logging.getLogger(__name__)

__all__ = [
    "BaseEmbedding",
    "CachedEmbedding",
    "EmbeddingFactory",
    "MockEmbedding",
    "HuggingFaceEmbedding",
    "OpenAIEmbedding",
    "GoogleEmbedding",
    "OllamaEmbedding",
]
