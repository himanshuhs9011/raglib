"""Ollama embedding implementation through LangChain adapters."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import (
    OLLAMA_DEFAULT_BASE_URL,
    PROVIDER_DEFAULT_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class OllamaEmbedding(BaseEmbedding):
    """Embed text with local Ollama models through LangChain OllamaEmbeddings."""

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Ollama embeddings with model and base URL overrides."""

        try:
            from langchain_ollama import OllamaEmbeddings  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("pip install langchain-ollama") from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["ollama"]
        resolved_base_url = base_url or OLLAMA_DEFAULT_BASE_URL
        self._model = OllamaEmbeddings(model=resolved_model, base_url=resolved_base_url)
        self._dimension = 0
        self._model_name = resolved_model
        self._base_url = resolved_base_url
        logger.info(
            "Using Ollama embedding model='%s' base_url='%s'",
            resolved_model,
            resolved_base_url,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents using Ollama embeddings."""

        logger.debug("OllamaEmbedding embed_documents called with batch size=%d", len(texts))
        vectors = self._model.embed_documents(texts)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Ollama embeddings."""

        logger.debug("OllamaEmbedding embed_query called with text length=%d", len(text))
        vector = self._model.embed_query(text)
        if vector:
            self._dimension = len(vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality when known."""

        return self._dimension
