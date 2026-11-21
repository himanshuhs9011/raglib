"""Local Ollama embeddings over its ``/api/embed`` endpoint."""

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
    """Embed text with a local Ollama model (no client package needed)."""

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 120.0):
        """Initialize with model and base URL overrides."""

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["ollama"]
        root = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        if root.endswith("/v1"):  # an OpenAI-style URL was given; the native API lives at the host root
            root = root[: -len("/v1")]
        self._base_url = root
        self.timeout = timeout
        self._dimension = 0
        self._model_name = resolved_model
        logger.info("Using Ollama embedding model='%s' base_url='%s'", resolved_model, root)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""

        from raglib.llm.http import post_json

        if not texts:
            return []
        body = post_json(
            f"{self._base_url}/api/embed", {"model": self._model_name, "input": list(texts)}, {}, timeout=self.timeout
        )
        vectors = [[float(value) for value in row] for row in body.get("embeddings") or []]
        if len(vectors) != len(texts):
            raise ValueError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs")
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""

        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality when known."""

        return self._dimension
