"""OpenAI embedding implementation through LangChain adapters."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """Embed text with OpenAI through LangChain OpenAIEmbeddings."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        compatible: bool = False,
        azure: bool = False,
        timeout: float = 60.0,
    ):
        """Initialize OpenAI embeddings.

        ``compatible=True`` targets OpenAI-compatible servers (Mistral, Together, vLLM, LM Studio,
        ...): texts are sent as strings instead of OpenAI tiktoken token arrays.
        ``azure=True`` uses Azure OpenAI (``model_name`` = deployment, ``base_url`` = endpoint).
        """

        try:
            from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings  # type: ignore[import-not-found]
            from pydantic import SecretStr
        except ImportError as exc:
            raise ImportError("pip install raglib-py[openai]") from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"]
        secret_api_key = SecretStr(api_key) if api_key else None
        if azure:
            import os

            self._model = AzureOpenAIEmbeddings(
                azure_deployment=resolved_model,
                azure_endpoint=base_url,
                api_key=secret_api_key,
                api_version=os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
                timeout=timeout,
                max_retries=0,
            )
        elif compatible:
            self._model = OpenAIEmbeddings(
                model=resolved_model,
                api_key=secret_api_key or SecretStr("not-needed"),
                base_url=base_url,
                check_embedding_ctx_length=False,
                timeout=timeout,
                max_retries=0,
            )
        else:
            self._model = OpenAIEmbeddings(
                model=resolved_model, api_key=secret_api_key, base_url=base_url, timeout=timeout, max_retries=0
            )
        self._dimension = 1536 if resolved_model == PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"] else 0
        self._model_name = resolved_model
        logger.info("Using OpenAI-style embedding model='%s' base_url=%s", resolved_model, base_url or "default")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings using OpenAI embeddings."""

        logger.debug("OpenAIEmbedding embed_documents called with batch size=%d", len(texts))
        vectors = self._model.embed_documents(texts)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string using OpenAI embeddings."""

        logger.debug("OpenAIEmbedding embed_query called with text length=%d", len(text))
        vector = self._model.embed_query(text)
        if vector:
            self._dimension = len(vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""

        return self._dimension
