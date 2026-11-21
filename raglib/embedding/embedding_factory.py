"""Factory for building embedding implementations by provider name."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.google_embedding import GoogleEmbedding
from raglib.embedding.huggingface_embedding import HuggingFaceEmbedding
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.embedding.ollama_embedding import OllamaEmbedding
from raglib.embedding.openai_embedding import OpenAIEmbedding

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Build embedding clients from provider strings with sensible fallbacks."""

    @staticmethod
    def build(
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        input_types: Optional[Sequence[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> BaseEmbedding:
        """Return the configured embedding implementation for a provider name.

        ``headers`` are extra HTTP headers (custom providers) for the OpenAI-style and Google APIs;
        ``input_types`` / ``extra_params`` are passed to OpenAI-style embeddings (see ``OpenAIEmbedding``).
        """

        provider_name = (provider or "mock").strip().lower()

        if provider_name == "mock":
            logger.info("EmbeddingFactory selected provider='mock'")
            return MockEmbedding()

        if provider_name in {"free", "local", "huggingface"}:
            logger.info("EmbeddingFactory selected provider='huggingface'")
            return HuggingFaceEmbedding(model_name=model_name)

        if provider_name == "openai":
            logger.info("EmbeddingFactory selected provider='openai'")
            return OpenAIEmbedding(api_key=api_key, model_name=model_name, base_url=base_url)

        if provider_name == "openai_compatible":
            if not base_url or not model_name:
                raise ValueError("openai_compatible embeddings require base_url and model_name")
            logger.info("EmbeddingFactory selected provider='openai_compatible' (%s)", base_url)
            return OpenAIEmbedding(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                compatible=True,
                extra_headers=headers,
                input_types=input_types,
                extra_params=extra_params,
            )

        if provider_name == "azure":
            if not base_url or not model_name:
                raise ValueError("Azure embeddings require the endpoint (base_url / AZURE_OPENAI_ENDPOINT) and deployment")
            return OpenAIEmbedding(api_key=api_key, model_name=model_name, base_url=base_url, azure=True)

        if provider_name == "google":
            logger.info("EmbeddingFactory selected provider='google'")
            return GoogleEmbedding(api_key=api_key, model_name=model_name, base_url=base_url, extra_headers=headers)

        if provider_name == "ollama":
            logger.info("EmbeddingFactory selected provider='ollama'")
            return OllamaEmbedding(model_name=model_name, base_url=base_url)

        raise ValueError(f"Unknown embedding provider '{provider_name}'.")
