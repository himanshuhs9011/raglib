"""Google Gemini embeddings over the ``batchEmbedContents`` REST API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_BATCH = 100  # Gemini's limit per batchEmbedContents call


class GoogleEmbedding(BaseEmbedding):
    """Embed text with Gemini embedding models (documents and queries use their own task types)."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize with an API key (or auth ``extra_headers``) and optional model / endpoint override."""

        from raglib.llm.clients import GEMINI_BASE_URL

        if not api_key and not extra_headers:
            raise ValueError("Google embeddings require an API key (GOOGLE_API_KEY or GEMINI_API_KEY)")
        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["google"]
        self._model_path = resolved_model if resolved_model.startswith("models/") else f"models/{resolved_model}"
        self.base_url = (base_url or GEMINI_BASE_URL).rstrip("/")
        self._headers = dict(extra_headers or {})
        if api_key:
            self._headers["x-goog-api-key"] = api_key
        self.timeout = timeout
        self._dimension = 0
        self._model_name = resolved_model
        logger.info("Using Google embedding model='%s'", resolved_model)

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from raglib.llm.http import post_json

        vectors: List[List[float]] = []
        url = f"{self.base_url}/{self._model_path}:batchEmbedContents"
        for start in range(0, len(texts), MAX_REQUESTS_PER_BATCH):
            batch = texts[start : start + MAX_REQUESTS_PER_BATCH]
            requests: List[Dict[str, Any]] = [
                {"model": self._model_path, "content": {"parts": [{"text": text}]}, "taskType": task_type}
                for text in batch
            ]
            body = post_json(url, {"requests": requests}, self._headers, timeout=self.timeout)
            rows = body.get("embeddings") or []
            if len(rows) != len(batch):
                raise ValueError(f"Gemini returned {len(rows)} embeddings for {len(batch)} inputs")
            vectors.extend([float(value) for value in row.get("values") or []] for row in rows)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (``RETRIEVAL_DOCUMENT`` task type)."""

        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query (``RETRIEVAL_QUERY`` task type)."""

        return self._embed([text], "RETRIEVAL_QUERY")[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality (known after the first call)."""

        return self._dimension
