"""Model providers on the facade: resolving the chat, embedding, vision and web-search providers from names, keys and environment, with rate limits and prompt caching."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from raglib.constants import (
    OFFLINE_ENV_VAR,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_VISION_PROVIDERS,
    WEB_REQUIRED_RAG_TYPES,
)
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.embedding.cached_embedding import CachedEmbedding
from raglib.embedding.embedding_factory import EmbeddingFactory
from raglib.facade.extras import _LocalSearchProvider
from raglib.llm.base_client import (
    BaseLLMClient,
    unwrap_llm,
)
from raglib.llm.constants import PROVIDER_DEFAULT_VISION_MODEL
from raglib.llm.mock_client import MockLLMClient
from raglib.llm.provider_detector import LLMProviderDetector
from raglib.llm.providers import (
    PROVIDERS,
    detect_provider_from_env,
    detect_provider_from_key,
    get_provider,
    normalize_provider,
    parse_model_string,
    provider_table,
)
from raglib.llm.resilience import shared_limiter
from raglib.providers.base_provider import BaseSearchProvider
from raglib.providers.factory import build_search_provider, normalize_provider_name
from raglib.vision.base_vision import BaseVisionClient
from raglib.vision.vision_factory import VisionFactory

logger = logging.getLogger("raglib.facade")  # one logger name for the whole facade, as before the split


OPENROUTER_FREE_RPM = 20  # OpenRouter free-model limit (requests/minute)


def _enable_prompt_caching(client: Any) -> None:
    """Turn on Anthropic prompt caching on ``client`` and its fallbacks (no-op for other providers)."""

    from raglib.llm.clients import AnthropicClient

    seen = 0
    for candidate in [client, *getattr(client, "clients", [])]:
        inner = unwrap_llm(candidate)
        if isinstance(inner, AnthropicClient):
            inner.prompt_caching = True
            seen += 1
    if not seen:
        logger.info("prompt_caching=True has no effect: the chat model is not an Anthropic model")


class ProvidersMixin:
    """Model providers on the facade: resolving the chat, embedding, vision and web-search providers from names, keys and environment, with rate limits and prompt caching (mixin of ``RAG``)."""

    def _resolve_chat_llm(
        self,
        chat_llm: Optional[Union[str, BaseLLMClient, Any]],
        api_key: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        timeout: float,
        max_retries: int,
    ):
        """Return (client, provider_name). Never silently falls back to a mock model.

        Accepts a provider name ("openrouter"), "provider/model" ("openrouter/deepseek/deepseek-chat"),
        an API key (provider detected from its prefix), a BaseLLMClient, or a LangChain chat model.
        With nothing configured, the first provider whose API key env var is set is used.
        """

        if isinstance(chat_llm, BaseLLMClient):
            provider = "mock" if isinstance(unwrap_llm(chat_llm), MockLLMClient) else None
            return chat_llm, provider
        if chat_llm is not None and not isinstance(chat_llm, str):
            return LLMProviderDetector.detect(value=chat_llm), None

        provider: Optional[str] = None
        text = chat_llm.strip() if isinstance(chat_llm, str) else ""
        if text.lower() == "mock":
            return MockLLMClient(), "mock"
        if text:
            provider, model_from_string = parse_model_string(text)
            if provider is None:
                provider = detect_provider_from_key(text)
                if provider is None:
                    raise ValueError(
                        f"Unknown chat_llm '{text}'. Use a provider name, 'provider/model', or a BaseLLMClient. "
                        f"Providers:\n{provider_table()}"
                    )
                api_key = api_key or text
            elif model_from_string and not model_name:
                model_name = model_from_string
        elif api_key:
            provider = detect_provider_from_key(api_key)
            if provider is None:
                raise ValueError(
                    "Could not detect the provider from the API key; also pass chat_llm='<provider>' "
                    f"(e.g. 'deepseek', 'openrouter'). Providers:\n{provider_table()}"
                )
        elif self._offline:
            logger.warning("offline mode: using MockLLMClient (answers are templated, not generated)")
            return MockLLMClient(), "mock"
        else:
            provider = detect_provider_from_env()
            if provider is None:
                raise ValueError(
                    "No chat model configured. Pass llm='openrouter/<model>' (or chat_llm='openai', ...), set a "
                    "provider API key environment variable, pass a BaseLLMClient / LangChain chat model, "
                    f"or use offline=True (or {OFFLINE_ENV_VAR}=1) for the offline mock.\nProviders:\n{provider_table()}"
                )
            logger.info("No chat_llm given; using '%s' (API key found in environment)", provider)

        spec = get_provider(provider)
        key = spec.resolve_api_key(api_key)
        if spec.requires_key and not key:
            raise ValueError(
                f"chat_llm='{spec.name}' requires chat_api_key (or llm_key) or one of environment variables: "
                f"{list(spec.env_keys)}"
            )
        from raglib.llm.clients import build_chat_client

        client = build_chat_client(
            provider=spec.name,
            api_key=key,
            model_name=model_name,
            base_url=base_url,
            temperature=getattr(self, "_llm_temperature", 0.0),
            max_tokens=getattr(self, "_llm_max_tokens", None),
            timeout=timeout,
            max_retries=max_retries,
        )
        rpm = self._effective_rpm(spec.name, client.model_name)
        if rpm:
            client.rate_limiter = shared_limiter(spec.name, key, rpm)
        return client, spec.name

    def _effective_rpm(self, provider: str, model_name: Optional[str]) -> Optional[float]:
        """Explicit ``llm_rpm`` wins; OpenRouter free models default to their 20 requests/minute cap."""

        explicit = getattr(self, "_llm_rpm", None)
        if explicit is not None:
            return explicit or None
        model = (model_name or "").lower()
        if provider == "openrouter" and (model.endswith(":free") or model == "openrouter/free"):
            return OPENROUTER_FREE_RPM
        return None

    def _resolve_embedding(
        self,
        embedding_llm: Optional[Union[str, BaseEmbedding]],
        api_key: Optional[str],
        chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        chat_api_key: Optional[str] = None,
    ) -> Optional[BaseEmbedding]:
        """Resolve the embedding model; ``None`` means BM25-only retrieval. Never uses hash vectors silently.

        ``api_key`` is an explicit embedding key; ``chat_api_key`` is reused only when the
        embedding provider is the same as the chat provider.
        """

        if isinstance(embedding_llm, BaseEmbedding):
            if isinstance(embedding_llm, CachedEmbedding):
                return embedding_llm
            # Your own model gets the same retry policy as the LLM (llm_max_retries).
            return CachedEmbedding(embedding_llm, max_retries=self._llm_max_retries)

        if embedding_llm is None:
            chat_spec = PROVIDERS.get(chat_provider or "")
            if chat_provider == "mock":
                provider = "mock"
            elif chat_spec is not None and chat_spec.default_embedding_model:
                provider = chat_spec.name
            elif self._can_import("sentence_transformers"):
                logger.info("No embedding_llm given; using local HuggingFace embeddings")
                provider = "huggingface"
            else:
                logger.warning(
                    "No embedding model available for chat provider '%s'; using BM25 keyword retrieval only. "
                    "Pass embedding_llm='openai' | 'mistral' | 'ollama' | 'huggingface' | 'provider/model' "
                    "for semantic search.",
                    chat_provider,
                )
                return None
        else:
            provider = embedding_llm.strip()
            for separator in ("/", ":"):
                head, _, tail = provider.partition(separator)
                if tail and (head.lower() in SUPPORTED_EMBEDDING_PROVIDERS or normalize_provider(head)):
                    provider, model_name = head, model_name or tail
                    break
            provider = provider.lower()

        provider = normalize_provider(provider) or provider
        if provider in {"none", "bm25", "sparse"}:
            return None
        if provider in {"mock", "huggingface", "free", "local"}:
            return CachedEmbedding(
                EmbeddingFactory.build(provider=provider, model_name=model_name, base_url=base_url, timeout=self._embedding_timeout),
                max_retries=self._llm_max_retries,
            )
        if provider in PROVIDERS and PROVIDERS[provider].custom and PROVIDERS[provider].protocol == "anthropic":
            raise ValueError(
                f"Provider '{provider}' uses the Anthropic API, which has no embeddings endpoint; "
                "pass embedding_llm='huggingface' | 'openai' | 'ollama' | ... or 'none'"
            )
        if provider not in PROVIDERS or provider in {"anthropic", "groq"}:
            raise ValueError(
                f"Unknown embedding provider '{provider}'. Supported: "
                f"{sorted((SUPPORTED_EMBEDDING_PROVIDERS | set(PROVIDERS) | {'none'}) - {'groq', 'anthropic'})}"
            )

        spec = PROVIDERS[provider]
        key = spec.resolve_api_key(api_key or (chat_api_key if provider == chat_provider else None))
        if spec.requires_key and not key:
            raise ValueError(
                f"Embedding provider '{provider}' requires embedding_api_key (or llm_key/chat_api_key) "
                f"or a configured environment variable: {list(spec.env_keys)}"
            )
        resolved_model = model_name or spec.default_embedding_model
        rpm = self._effective_rpm(provider, resolved_model)
        limiter = shared_limiter(provider, key, rpm) if rpm else None
        if spec.custom:
            if not resolved_model:
                raise ValueError(
                    f"Custom provider '{provider}' has no embedding_model; set it in register_provider()/the "
                    f"providers file, or pass embedding_llm='{provider}/<model>'"
                )
            embedding = EmbeddingFactory.build(
                provider="google" if spec.protocol == "gemini" else "openai_compatible",
                api_key=key,
                model_name=resolved_model,
                base_url=spec.resolve_base_url(base_url),
                headers=spec.resolve_headers(),
                input_types=spec.embedding_input_types,
                extra_params=spec.embedding_params,
                timeout=self._embedding_timeout,
            )
        elif provider in {"openai", "google", "ollama"}:
            embedding = EmbeddingFactory.build(
                provider=provider, api_key=key, model_name=resolved_model, base_url=base_url, timeout=self._embedding_timeout
            )
        else:
            if not resolved_model:
                raise ValueError(
                    f"Provider '{provider}' has no default embedding model; pass embedding_model=... "
                    "or embedding_llm='provider/model'"
                )
            embedding = EmbeddingFactory.build(
                provider="azure" if provider == "azure" else "openai_compatible",
                api_key=key,
                model_name=resolved_model,
                base_url=spec.resolve_base_url(base_url),
                input_types=spec.embedding_input_types,
                extra_params=spec.embedding_params,
                timeout=self._embedding_timeout,
            )
        return CachedEmbedding(embedding, max_retries=getattr(self, "_llm_max_retries", 2), rate_limiter=limiter)

    def _resolve_vision(
        self,
        vision_llm: Optional[Union[str, BaseVisionClient]],
        api_key: Optional[str],
        chat_provider: Optional[str],
        model_name: Optional[str],
        base_url: Optional[str],
        explicit_key: Optional[str] = None,
        chat_api_key: Optional[str] = None,
    ) -> Optional[BaseVisionClient]:
        """Resolve OCR/vision for scanned pages; None disables it (pages are skipped with a warning).

        ``explicit_key`` is vision_api_key / llm_key; ``chat_api_key`` is only reused for a custom
        provider when it is the chat provider too (a key is never sent to a different host).
        """

        if isinstance(vision_llm, BaseVisionClient):
            return vision_llm
        if vision_llm is None:
            chat_spec = PROVIDERS.get(chat_provider or "")
            chat_model = getattr(unwrap_llm(getattr(self, "_llm_raw", None)), "model_name", None)
            model_spec = chat_spec.model(chat_model) if chat_spec is not None and chat_spec.custom else None
            if model_spec is not None and model_spec.image:
                # the custom chat model accepts images: reuse it (same provider, same key) for OCR
                return self._chat_vision(chat_spec, model_name or chat_model, explicit_key, chat_api_key, base_url)
            if chat_provider in {"openai", "anthropic", "google"} and api_key:
                provider = chat_provider
            elif chat_spec is not None and chat_provider in PROVIDER_DEFAULT_VISION_MODEL and api_key:
                return self._chat_vision(chat_spec, model_name, explicit_key, chat_api_key, base_url)  # e.g. nvidia
            else:
                return None
        else:
            head, _, tail = vision_llm.strip().partition("/")
            spec = PROVIDERS.get(normalize_provider(head) or "")
            provider = head.lower()
            model_name = model_name or tail or None
            if spec is not None and (spec.custom or (spec.supports_vision and provider not in SUPPORTED_VISION_PROVIDERS)):
                same = spec.name == chat_provider
                return self._chat_vision(spec, model_name, explicit_key, chat_api_key if same else None, base_url)
        if provider not in SUPPORTED_VISION_PROVIDERS:
            raise ValueError(f"Unknown vision provider '{provider}'. Supported: {sorted(SUPPORTED_VISION_PROVIDERS)}")
        return VisionFactory.build(
            provider=provider,
            api_key=api_key if provider != "mock" else None,
            model_name=model_name,
            base_url=base_url,
        )

    def _chat_vision(
        self,
        spec: Any,
        model: Optional[str],
        explicit_key: Optional[str],
        chat_api_key: Optional[str],
        base_url: Optional[str],
    ) -> BaseVisionClient:
        """OCR through a chat client: custom providers' image models, or built-ins such as NVIDIA."""

        from raglib.llm.clients import build_chat_client
        from raglib.vision.chat_vision import ChatVisionClient

        image_models = [item.id for item in spec.models if item.image]
        model = model or (image_models[0] if image_models else None) or PROVIDER_DEFAULT_VISION_MODEL.get(spec.name)
        if not model:
            hint = "declares no model with image=True" if spec.custom else "has no default vision model"
            raise ValueError(f"Provider '{spec.name}' {hint}; pass vision_llm='{spec.name}/<model>'")
        declared = spec.model(model)
        if declared is not None and not declared.image:
            raise ValueError(
                f"Model '{model}' of provider '{spec.name}' is not marked image=True, so it cannot read scanned pages"
                + (f"; image models: {image_models}" if image_models else "")
            )
        client = build_chat_client(
            spec.name,
            api_key=explicit_key or chat_api_key,
            model_name=model,
            base_url=base_url,
            max_tokens=8192,  # room for reasoning tokens before the transcription
            timeout=getattr(self, "_llm_timeout", 60.0),
            max_retries=getattr(self, "_llm_max_retries", 2),
        )
        return ChatVisionClient(client)

    def _resolve_web_provider(
        self,
        provider_name: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        cse_id: Optional[str],
        provider_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[BaseSearchProvider]:
        """Build the web provider. Web search is OFF unless configured (or required by rag_type)."""

        if provider_name is None:
            if self._rag_type not in WEB_REQUIRED_RAG_TYPES:
                return None
            logger.info("rag_type='%s' needs web search; defaulting to DuckDuckGo", self._rag_type)
            provider_name = "duckduckgo"
        name = normalize_provider_name(provider_name)
        if name == "local":
            if provider_kwargs:
                raise ValueError("web_search_provider='local' does not accept web_search_provider_kwargs")
            return _LocalSearchProvider(documents_fn=lambda: self._retriever.documents)
        return build_search_provider(
            provider_name=name,
            api_key=api_key,
            base_url=base_url,
            cse_id=cse_id,
            provider_kwargs=provider_kwargs,
        )

    def _validate_web_provider_connection(self, provider: BaseSearchProvider) -> None:
        """Perform a lightweight web provider credential check."""

        try:
            provider.search(query="raglib api key validation", num_results=1)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"web_search_provider='{provider.name}' failed validation. Check internet access and web_search_api_key."
            ) from exc
