"""Provider auto-detection for chat LLM clients."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from raglib.constants import LLM_KEY_PREFIX_MAP
from raglib.llm.base_client import BaseLLMClient
from raglib.llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)


class LLMProviderDetector:
    """Detect and build the correct chat LLM client from user input."""

    KEY_PREFIX_MAP = dict(LLM_KEY_PREFIX_MAP)

    @classmethod
    def known_providers(cls) -> set:
        """Built-in plus registered custom providers (read live, so register_provider() counts)."""

        from raglib.llm.providers import PROVIDERS, normalize_provider

        normalize_provider("")  # loads RAGLIB_PROVIDERS_FILE if set
        return set(PROVIDERS)

    @classmethod
    def detect(
        cls,
        value: Optional[Union[str, BaseLLMClient, Any]],
        provider_hint: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseLLMClient:
        """Return a ready-to-use BaseLLMClient from key, provider, or instance input."""

        if isinstance(value, BaseLLMClient):
            logger.info("Chat LLM client provided directly; skipping provider detection.")
            return value

        if value is not None and not isinstance(value, str):
            if hasattr(value, "invoke"):
                from raglib.llm.langchain_adapter import LangChainAdapter

                provider_name = type(value).__name__
                logger.info("Wrapping provided LangChain-style model: %s", provider_name)
                return LangChainAdapter(langchain_model=value, provider_name=provider_name)
            raise TypeError(
                "value must be None, str, BaseLLMClient, or a LangChain-style chat model with invoke()"
            )

        if provider_hint:
            provider = provider_hint.strip().lower()
            if provider not in cls.known_providers():
                raise ValueError(
                    f"Unknown provider_hint '{provider_hint}'. "
                    f"Supported providers: {sorted(cls.known_providers())}"
                )
            api_key: Optional[str] = None
            if isinstance(value, str) and value.strip() and value.strip().lower() not in cls.known_providers():
                api_key = value.strip()
            logger.info("Using explicit provider hint: %s", provider)
            return cls._build_client(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
            )

        if value is None:
            logger.info("No LLM key provided. Using MockLLMClient (offline mode).")
            return MockLLMClient()

        cleaned = value.strip()
        if not cleaned:
            logger.warning("Empty LLM value provided. Falling back to MockLLMClient.")
            return MockLLMClient()

        lowered = cleaned.lower()
        if lowered in cls.known_providers():
            logger.info("Detected provider from explicit name: %s", lowered)
            return cls._build_client(
                provider=lowered,
                api_key=None,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
            )

        provider = cls._detect_from_key(cleaned)
        if provider:
            logger.info("Detected provider '%s' from API key prefix.", provider)
            return cls._build_client(
                provider=provider,
                api_key=cleaned,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
            )

        raise ValueError(
            "Could not detect an LLM provider from the value you provided.\n"
            f"Supported providers: {sorted(cls.known_providers())}\n"
            f"Supported key prefixes: {list(cls.KEY_PREFIX_MAP.keys())}\n"
            "To use an unsupported provider, pass a BaseLLMClient instance directly."
        )

    @classmethod
    def infer_provider(
        cls,
        value: Optional[Union[str, BaseLLMClient]],
        provider_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Infer a provider name without instantiating a chat client."""

        if provider_hint:
            candidate = provider_hint.strip().lower()
            if candidate in cls.known_providers():
                return candidate
            return None

        if value is None or isinstance(value, BaseLLMClient):
            return None

        if not isinstance(value, str):
            return None

        cleaned = value.strip()
        if not cleaned:
            return None

        lowered = cleaned.lower()
        if lowered in cls.known_providers():
            return lowered

        return cls._detect_from_key(cleaned)

    @classmethod
    def _detect_from_key(cls, key: str) -> Optional[str]:
        """Detect a provider from API key prefix with deterministic priority order."""

        from raglib.llm.providers import detect_provider_from_key

        return detect_provider_from_key(key)

    @classmethod
    def _build_client(
        cls,
        provider: str,
        api_key: Optional[str],
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseLLMClient:
        """Instantiate the native client for the provider."""

        from raglib.llm.clients import build_chat_client

        return build_chat_client(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )
