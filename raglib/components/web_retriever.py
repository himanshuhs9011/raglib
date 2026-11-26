"""Web retrieval component backed by provider abstraction."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import replace
from typing import Any, Dict, List, Optional

from raglib.access import FilterNotSupportedError
from raglib.core.runtime import report_degraded
from raglib.providers.base_provider import CHAIN_INDEX_KEY, BaseSearchProvider, ProviderError
from raglib.providers.duckduckgo_provider import DuckDuckGoProvider
from raglib.providers.factory import build_search_provider
from raglib.schemas import Document
from raglib.vectorstores.base_store import MetadataFilter

logger = logging.getLogger(__name__)


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

    def retrieve(self, query: str, filters: Optional[MetadataFilter] = None) -> List[Document]:
        """Retrieve top-k web documents for a query.

        ``filters`` apply only to providers that search the local corpus (see
        ``BaseSearchProvider.searches_corpus``); public internet results are not corpus documents.
        """

        if not isinstance(query, str) or not query.strip():
            return []

        logger.info("WebRetriever querying provider=%s", self.provider.name)
        try:
            docs = self._search(self.provider, query, filters)
        except FilterNotSupportedError:
            raise
        except Exception as exc:  # noqa: BLE001
            report_degraded(f"web_search:{self.provider.name}", exc, logger)
            fallback_docs = self._retrieve_from_fallback(query=query, cause=exc, filters=filters)
            if fallback_docs:
                return fallback_docs
            if self.fail_silently:
                return []
            raise ProviderError(f"Web retrieval failed for provider={self.provider.name}: {exc}") from exc

        return self._normalize_documents(docs=docs, provider_name=self.provider.name, provider=self.provider)

    def _search(self, provider: BaseSearchProvider, query: str, filters: Optional[MetadataFilter]) -> List[Document]:
        """Call one provider, enforcing ``filters`` on providers that search the local corpus."""

        if filters and getattr(provider, "searches_corpus", False):
            if not getattr(provider, "supports_filters", False):
                raise FilterNotSupportedError(
                    f"Search provider '{provider.name}' searches the corpus but cannot apply metadata filters"
                )
            return provider.search(query=query, num_results=self.top_k, filters=filters)  # type: ignore[call-arg]
        if not getattr(provider, "searches_corpus", False):
            from raglib.core.runtime import record_search_usage

            record_search_usage(f"web:{provider.name}", 1)  # billable API call (priced per 1K searches)
            if filters and not getattr(provider, "internal", False):
                _warn_once(
                    f"provider:{provider.name}",
                    f"filters were not applied by provider {provider.name} (public web results are not corpus documents)",
                )
        docs = provider.search(query=query, num_results=self.top_k)
        if filters and getattr(provider, "internal", False):
            docs = [doc for doc in docs if not _internal(provider, doc) or self._internal_visible(doc, filters)]
        return docs

    def _internal_visible(self, doc: Document, filters: MetadataFilter) -> bool:
        """An internal provider's result passes the query's filters by the tags it carries.

        A result whose tags cannot be read (an unknown ``access_level``) is hidden on its own;
        the other results are kept. ``source`` filters see the result's source, as in the corpus.
        """

        from raglib.vectorstores.base_store import matches_filter

        metadata = {key: value for key, value in (doc.metadata or {}).items() if key != CHAIN_INDEX_KEY}
        if doc.source:
            metadata.setdefault("source", doc.source)
        access = getattr(self, "access", None)
        if access is not None:
            try:
                metadata = access.tag(metadata)  # derive the _acl_* keys the permission filter checks
            except ValueError as exc:
                logger.warning("Hiding an internal search result with unreadable access tags: %s", exc)
                return False
        return matches_filter(metadata, filters, array_contains=True)

    def _retrieve_from_fallback(
        self, query: str, cause: Exception, filters: Optional[MetadataFilter] = None
    ) -> List[Document]:
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
            docs = self._search(self.fallback_provider, query, filters)
        except FilterNotSupportedError:
            raise
        except Exception as fallback_exc:  # noqa: BLE001
            report_degraded(f"web_search:{self.fallback_provider.name}", fallback_exc, logger)
            if self.fail_silently:
                return []
            raise ProviderError(
                "Web retrieval failed for primary provider "
                f"'{self.provider.name}' and fallback provider '{self.fallback_provider.name}'"
            ) from fallback_exc

        return self._normalize_documents(docs=docs, provider_name=self.fallback_provider.name, provider=self.fallback_provider)

    def _normalize_documents(self, docs: List[Document], provider_name: str, provider: Any = None) -> List[Document]:
        """Normalize documents to unified title/content/url/score metadata schema.

        With access control (``public_access_level`` set by ``RAG``), results of a public provider
        are tagged ``source_type="web"`` at the lowest access level, so the citation check and the
        "non-public" rules treat them as public. Results of an ``internal`` or corpus provider keep
        their own tags (none means restricted).
        """

        level = getattr(self, "public_access_level", None)
        normalized: List[Document] = []
        for idx, doc in enumerate(docs[: self.top_k]):
            origin = _origin(provider, doc)
            restricted = bool(getattr(origin, "internal", False) or getattr(origin, "searches_corpus", False))
            title = str(doc.metadata.get("title", "")).strip() or self._derive_title(doc.content)
            url = str(doc.metadata.get("url") or doc.metadata.get("link") or "").strip()
            snippet = str(doc.metadata.get("snippet", "")).strip() or doc.content

            score = float(doc.score)
            if score == 0.0:
                score = max(0.0, 1.0 - (idx / max(self.top_k, 1)))

            metadata = {
                **{key: value for key, value in doc.metadata.items() if key != CHAIN_INDEX_KEY},
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": doc.metadata.get("provider", provider_name),
                "score": score,
            }
            if level is not None and not restricted:
                metadata.setdefault("access_level", level)
                metadata.setdefault("source_type", "web")
            elif level is not None:
                metadata.setdefault("source_type", "internal")

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


def _origin(provider: Any, doc: Document) -> Any:
    """The provider that returned ``doc``: the chain member for a ``ProviderChain`` result."""

    index = (doc.metadata or {}).get(CHAIN_INDEX_KEY)
    members = getattr(provider, "providers", None)
    if isinstance(index, int) and members is not None and 0 <= index < len(members):
        return members[index]
    return provider


def _internal(provider: Any, doc: Document) -> bool:
    return bool(getattr(_origin(provider, doc), "internal", False))


_warned: set = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str) -> None:
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


def web_retrieve(retriever: Any, query: str, filters: Optional[MetadataFilter] = None) -> List[Document]:
    """``retriever.retrieve``, with ``filters`` when set and its ``retrieve`` accepts them.

    A custom web retriever written for 2.2 (``retrieve(self, query)``) keeps working: it is called
    without filters, and a warning is logged once.
    """

    if filters:
        if accepts_filters(retriever.retrieve):
            return retriever.retrieve(query, filters=filters)
        name = type(retriever).__name__
        _warn_once(f"retriever:{name}", f"filters were not applied by web retriever {name} (its retrieve() has no filters parameter)")
    return retriever.retrieve(query)


def corpus_web_retrieve(hybrid_retriever: Any, query: str, filters: Optional[MetadataFilter] = None) -> List[Document]:
    """``hybrid_retriever.retrieve`` (corpus + web): as in 2.2 without filters; with filters, a
    custom hybrid retriever that cannot take them raises ``FilterNotSupportedError`` (it would
    search the corpus unfiltered)."""

    if not filters:
        return hybrid_retriever.retrieve(query)
    if not accepts_filters(hybrid_retriever.retrieve):
        raise FilterNotSupportedError(
            f"hybrid retriever {type(hybrid_retriever).__name__} cannot apply metadata filters (its retrieve() has no filters parameter)"
        )
    return hybrid_retriever.retrieve(query, filters=filters)
