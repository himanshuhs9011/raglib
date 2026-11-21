"""OpenAI-protocol embeddings (OpenAI, Azure OpenAI, Mistral, Together, vLLM, LM Studio, ...)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

MAX_INPUTS_PER_REQUEST = 2048  # OpenAI's limit on the size of the ``input`` array


class OpenAIEmbedding(BaseEmbedding):
    """Embed text through the ``/embeddings`` endpoint over plain HTTPS (no SDK needed)."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        compatible: bool = False,
        azure: bool = False,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        input_types: Optional[Sequence[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenAI-protocol embeddings.

        ``compatible=True`` targets OpenAI-compatible servers (Mistral, Together, vLLM, LM Studio,
        ...). ``azure=True`` uses Azure OpenAI (``model_name`` = deployment, ``base_url`` = endpoint).
        ``input_types=("passage", "query")`` sends an ``input_type`` field, which asymmetric models
        such as NVIDIA NIM retrievers require; ``extra_params`` adds fields like ``{"truncate": "END"}``.
        """

        from raglib.llm.clients import OPENAI_BASE_URL, azure_api_version

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"]
        if azure and not base_url:
            raise ValueError("Azure embeddings require the endpoint (base_url / AZURE_OPENAI_ENDPOINT)")
        if compatible and not base_url:
            raise ValueError("OpenAI-compatible embeddings require base_url")
        import os

        root = (base_url or os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        self._headers: Dict[str, str] = dict(extra_headers or {})
        if azure:
            self._url = f"{root}/openai/deployments/{resolved_model}/embeddings?api-version={azure_api_version()}"
            if api_key:
                self._headers["api-key"] = api_key
        else:
            self._url = f"{root}/embeddings"
            if api_key:
                self._headers["Authorization"] = f"Bearer {api_key}"
        self._send_model = not azure
        self._send_encoding_format = not compatible  # some compatible servers reject unknown fields
        self.base_url = root
        if input_types is not None and len(tuple(input_types)) != 2:
            raise ValueError("input_types must be (document_type, query_type)")
        self.input_types = tuple(input_types) if input_types is not None else None
        self.extra_params = dict(extra_params or {})
        self.timeout = timeout
        self._dimension = 1536 if resolved_model == PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"] else 0
        self._model_name = resolved_model
        logger.info("Using OpenAI-style embedding model='%s' base_url=%s", resolved_model, root)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (split into requests of at most 2048 inputs)."""

        return self._embed(texts, self.input_types[0] if self.input_types else None)

    def _embed(self, texts: List[str], input_type: Optional[str]) -> List[List[float]]:
        from raglib.llm.http import post_json

        vectors: List[List[float]] = []
        for start in range(0, len(texts), MAX_INPUTS_PER_REQUEST):
            batch = [text if text.strip() else " " for text in texts[start : start + MAX_INPUTS_PER_REQUEST]]
            payload: Dict[str, Any] = {"input": batch, **self.extra_params}
            if input_type:
                payload["input_type"] = input_type
            if self._send_model:
                payload["model"] = self._model_name
            if self._send_encoding_format:
                payload["encoding_format"] = "float"
            body = post_json(self._url, payload, self._headers, timeout=self.timeout)
            rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
            if len(rows) != len(batch):
                raise ValueError(f"Embedding API returned {len(rows)} vectors for {len(batch)} inputs")
            vectors.extend([float(value) for value in row["embedding"]] for row in rows)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query (``input_type`` = the query type when configured)."""

        return self._embed([text], self.input_types[1] if self.input_types else None)[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality (known after the first call for non-default models)."""

        return self._dimension
