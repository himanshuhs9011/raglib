"""Single place that builds web search providers from a name + config."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

from raglib.providers.base_provider import BaseSearchProvider, ProviderNotConfiguredError

logger = logging.getLogger(__name__)

WEB_PROVIDER_ALIASES: Dict[str, str] = {
    "ddg": "duckduckgo",
    "duckduckgo_search": "duckduckgo",
    "google": "google_cse",
    "local_web": "local",
}

WEB_PROVIDERS_REQUIRING_API_KEY = {"tavily", "serpapi", "brave", "bing", "google_cse", "exa"}

WEB_PROVIDER_ENV_KEYS: Dict[str, Sequence[str]] = {
    "tavily": ("TAVILY_API_KEY",),
    "serpapi": ("SERPAPI_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"),
    "bing": ("BING_SEARCH_API_KEY", "BING_SEARCH_V7_SUBSCRIPTION_KEY"),
    "google_cse": ("GOOGLE_API_KEY",),
    "exa": ("EXA_API_KEY",),
    "searxng": ("SEARXNG_API_KEY",),
}

SUPPORTED_WEB_PROVIDERS = {"duckduckgo", "tavily", "serpapi", "brave", "bing", "google_cse", "exa", "searxng"}


def normalize_provider_name(name: Optional[str]) -> str:
    """Normalize aliases to canonical provider names."""

    normalized = (name or "duckduckgo").strip().lower()
    return WEB_PROVIDER_ALIASES.get(normalized, normalized)


def _env(keys: Sequence[str]) -> Optional[str]:
    """Return the first non-empty environment variable among ``keys``."""

    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def build_search_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    cse_id: Optional[str] = None,
    provider_kwargs: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
) -> BaseSearchProvider:
    """Build a provider; resolves keys from env vars when ``use_env`` and validates options."""

    from raglib.providers.bing_provider import BingProvider
    from raglib.providers.brave_provider import BraveProvider
    from raglib.providers.duckduckgo_provider import DuckDuckGoProvider
    from raglib.providers.exa_provider import ExaProvider
    from raglib.providers.google_cse_provider import GoogleCSEProvider
    from raglib.providers.searxng_provider import SearxNGProvider
    from raglib.providers.serpapi_provider import SerpAPIProvider
    from raglib.providers.tavily_provider import TavilyProvider

    name = normalize_provider_name(provider_name)
    if name not in SUPPORTED_WEB_PROVIDERS:
        raise ValueError(f"Unknown web_search_provider '{provider_name}'. Supported: {sorted(SUPPORTED_WEB_PROVIDERS)}")
    kwargs = dict(provider_kwargs or {})
    key = (api_key or "").strip() or (_env(WEB_PROVIDER_ENV_KEYS.get(name, ())) if use_env else None)
    if name in WEB_PROVIDERS_REQUIRING_API_KEY and not key:
        raise ProviderNotConfiguredError(
            f"web_search_provider='{name}' requires web_search_api_key "
            f"or one of environment variables: {list(WEB_PROVIDER_ENV_KEYS.get(name, ()))}"
        )
    timeout = int(kwargs.pop("timeout", 10))

    if name == "duckduckgo":
        provider: BaseSearchProvider = DuckDuckGoProvider(
            region=str(kwargs.pop("region", "wt-wt")),
            safesearch=str(kwargs.pop("safesearch", "moderate")),
            timelimit=kwargs.pop("timelimit", None),
            backend=str(kwargs.pop("backend", "auto")),
        )
    elif name == "tavily":
        provider = TavilyProvider(
            api_key=key,
            topic=str(kwargs.pop("topic", "general")),
            search_depth=str(kwargs.pop("search_depth", "basic")),
            include_answer=bool(kwargs.pop("include_answer", False)),
            timeout=timeout,
        )
    elif name == "serpapi":
        provider = SerpAPIProvider(
            api_key=key,
            engine=str(kwargs.pop("engine", "google")),
            location=str(kwargs.pop("location", "United States")),
            timeout=timeout,
        )
    elif name == "brave":
        provider = BraveProvider(
            api_key=key,
            country=str(kwargs.pop("country", "us")),
            search_lang=str(kwargs.pop("search_lang", "en")),
            timeout=timeout,
        )
    elif name == "bing":
        provider = BingProvider(api_key=key, market=str(kwargs.pop("market", "en-US")), timeout=timeout)
    elif name == "google_cse":
        resolved_cse = cse_id or kwargs.pop("cse_id", None) or (_env(("GOOGLE_CSE_ID", "GOOGLE_CSE_ENGINE_ID")) if use_env else None)
        if not resolved_cse:
            raise ProviderNotConfiguredError(
                "web_search_provider='google_cse' requires web_search_cse_id or environment variable GOOGLE_CSE_ID"
            )
        provider = GoogleCSEProvider(api_key=key, cse_id=str(resolved_cse), safe=str(kwargs.pop("safe", "off")), timeout=timeout)
    elif name == "exa":
        provider = ExaProvider(api_key=key, use_autoprompt=bool(kwargs.pop("use_autoprompt", True)), timeout=timeout)
    else:  # searxng
        resolved_url = base_url or kwargs.pop("base_url", None) or (os.getenv("SEARXNG_BASE_URL") if use_env else None)
        if not resolved_url:
            raise ProviderNotConfiguredError(
                "web_search_provider='searxng' requires web_search_base_url or environment variable SEARXNG_BASE_URL"
            )
        provider = SearxNGProvider(
            api_key=key,
            base_url=str(resolved_url),
            categories=str(kwargs.pop("categories", "general")),
            language=str(kwargs.pop("language", "en")),
            timeout=timeout,
        )

    if kwargs:
        raise ValueError(f"Unknown web_search_provider_kwargs for provider '{name}': {', '.join(sorted(kwargs))}")
    return provider
