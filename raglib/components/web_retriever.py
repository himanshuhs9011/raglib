"""Web retrieval component backed by provider abstraction."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import replace
from typing import Any, Dict, List, Optional

from raglib.core.runtime import report_degraded
from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.providers.duckduckgo_provider import DuckDuckGoProvider
from raglib.providers.factory import build_search_provider
from raglib.schemas import Document

logger = logging.getLogger(__name__)

# Providers that search the local corpus instead of the public web; they receive ``filters``.
CORPUS_BACKED_PROVIDERS = {"local_web"}


class WebRetriever:
    """Retrieves documents from web providers with configurable fallback."""

    def __init__(
        self,
        provider: Optional[BaseSearchProvider] = None,
        provider_name: str = "duckduckgo",
        provider_api_key: Optional[str] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        fallback_provider: Optional[BaseSearchProvider] = None,
        enable_duckduckgo_fallback: bool = False,
        top_k: int = 5,
        fail_silently: bool = True,
    ):
        """Initialize web retriever using provider instance or provider config."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._build_provider(
                provider_name=provider_name,
                provider_api_key=provider_api_key,
                provider_kwargs=provider_kwargs,
            )

        self.top_k = top_k
        self.fail_silently = fail_silently

        if fallback_provider is not None:
            self.fallback_provider = fallback_provider
        elif enable_duckduckgo_fallback and self.provider.name != DuckDuckGoProvider.PROVIDER_NAME:
            self.fallback_provider = DuckDuckGoProvider()
        else:
            self.fallback_provider = None

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve top-k web documents for a query.

        ``filters`` restrict providers that search the corpus (``web_search_provider="local"``).
        Public web results are not corpus documents, so other providers ignore them.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        logger.info("WebRetriever querying provider=%s", self.provider.name)
        try:
            docs = _search(self.provider, query, self.top_k, filters)
        except Exception as exc:  # noqa: BLE001
            report_degraded(f"web_search:{self.provider.name}", exc, logger)
            fallback_docs = self._retrieve_from_fallback(query=query, cause=exc, filters=filters)
            if fallback_docs:
                return fallback_docs
            if self.fail_silently:
                return []
            raise ProviderError(f"Web retrieval failed for provider={self.provider.name}: {exc}") from exc

        return self._normalize_documents(docs=docs, provider_name=self.provider.name)

    def _retrieve_from_fallback(self, query: str, cause: Exception, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve from fallback provider after primary provider failure."""

        if self.fallback_provider is None:
            return []

        logger.warning(
            "WebRetriever falling back from provider=%s to provider=%s due to: %s",
            self.provider.name,
            self.fallback_provider.name,
            cause,
        )

        try:
            docs = _search(self.fallback_provider, query, self.top_k, filters)
        except Exception as fallback_exc:  # noqa: BLE001
            report_degraded(f"web_search:{self.fallback_provider.name}", fallback_exc, logger)
            if self.fail_silently:
                return []
            raise ProviderError(
                "Web retrieval failed for primary provider "
                f"'{self.provider.name}' and fallback provider '{self.fallback_provider.name}'"
            ) from fallback_exc

        return self._normalize_documents(docs=docs, provider_name=self.fallback_provider.name)

    def _normalize_documents(self, docs: List[Document], provider_name: str) -> List[Document]:
        """Normalize documents to unified title/content/url/score metadata schema."""

        normalized: List[Document] = []
        for idx, doc in enumerate(docs[: self.top_k]):
            title = str(doc.metadata.get("title", "")).strip() or self._derive_title(doc.content)
            url = str(doc.metadata.get("url") or doc.metadata.get("link") or "").strip()
            snippet = str(doc.metadata.get("snippet", "")).strip() or doc.content

            score = float(doc.score)
            if score == 0.0:
                score = max(0.0, 1.0 - (idx / max(self.top_k, 1)))

            metadata = {
                **doc.metadata,
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": doc.metadata.get("provider", provider_name),
                "score": score,
            }

            normalized.append(
                replace(
                    doc,
                    content=snippet,
                    score=score,
                    source=doc.source or provider_name,
                    metadata=metadata,
                )
            )

        return normalized

    def _build_provider(
        self,
        provider_name: str,
        provider_api_key: Optional[str],
        provider_kwargs: Optional[Dict[str, Any]],
    ) -> BaseSearchProvider:
        """Build a provider instance from name and config (no env lookup: explicit config only)."""

        return build_search_provider(
            provider_name=provider_name,
            api_key=provider_api_key,
            provider_kwargs=provider_kwargs,
            use_env=False,
        )

    def _derive_title(self, content: str) -> str:
        """Derive a compact title when provider response omits one."""

        line = content.strip().split("\n", maxsplit=1)[0].strip()
        if not line:
            return "Web result"
        return line[:120]


_warned: set = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    """Log ``message`` the first time ``key`` is seen in this process."""

    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message)


def accepts_filters(method: Any) -> bool:
    """True when ``method`` takes a ``filters`` keyword (or ``**kwargs``)."""

    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "filters" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _search(provider: BaseSearchProvider, query: str, num_results: int, filters: Optional[Dict[str, Any]]) -> List[Document]:
    """Call ``provider.search``, passing ``filters`` to corpus-backed providers.

    Other providers cannot apply filters: with filters set, a warning is logged once per provider.
    """

    if filters and provider.name in CORPUS_BACKED_PROVIDERS:
        return provider.search(query=query, num_results=num_results, filters=filters)  # type: ignore[call-arg]
    if filters:
        _warn_once(f"provider:{provider.name}", f"filters were not applied by provider {provider.name}")
    return provider.search(query=query, num_results=num_results)


def web_retrieve(web_retriever: Any, query: str, filters: Optional[Dict[str, Any]]) -> List[Document]:
    """``web_retriever.retrieve``, with ``filters`` when its ``retrieve`` accepts them.

    A custom web retriever written for 2.2 (``retrieve(self, query)``) keeps working: it is called
    without filters, and a warning is logged once.
    """

    if filters:
        if accepts_filters(web_retriever.retrieve):
            return web_retriever.retrieve(query, filters=filters)
        name = type(web_retriever).__name__
        _warn_once(f"retriever:{name}", f"filters were not applied by web retriever {name} (its retrieve() has no filters parameter)")
    return web_retriever.retrieve(query)


def corpus_web_retrieve(hybrid_retriever: Any, query: str, filters: Optional[Dict[str, Any]]) -> List[Document]:
    """``hybrid_retriever.retrieve`` (corpus + web), with ``filters`` when given.

    Without filters it is called exactly as in 2.2. A custom hybrid retriever whose ``retrieve``
    takes no ``filters`` searches the corpus unfiltered, so a filtered query raises
    ``FilterNotSupportedError`` instead of returning documents outside the filter.
    """

    if not filters:
        return hybrid_retriever.retrieve(query)
    if not accepts_filters(hybrid_retriever.retrieve):
        from raglib.access import FilterNotSupportedError

        raise FilterNotSupportedError(
            f"hybrid retriever {type(hybrid_retriever).__name__} cannot apply metadata filters (its retrieve() has no filters parameter)"
        )
    return hybrid_retriever.retrieve(query, filters=filters)
