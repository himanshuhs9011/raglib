"""Web retrieval component backed by provider abstraction."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.providers.duckduckgo_provider import DuckDuckGoProvider
from raglib.providers.factory import build_search_provider
from raglib.schemas import Document

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

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k web documents for a query."""

        if not isinstance(query, str) or not query.strip():
            return []

        logger.info("WebRetriever querying provider=%s", self.provider.name)
        try:
            docs = self.provider.search(query=query, num_results=self.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebRetriever provider failed: %s", exc)
            fallback_docs = self._retrieve_from_fallback(query=query, cause=exc)
            if fallback_docs:
                return fallback_docs
            if self.fail_silently:
                return []
            raise ProviderError(f"Web retrieval failed for provider={self.provider.name}: {exc}") from exc

        return self._normalize_documents(docs=docs, provider_name=self.provider.name)

    def _retrieve_from_fallback(self, query: str, cause: Exception) -> List[Document]:
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
            docs = self.fallback_provider.search(query=query, num_results=self.top_k)
        except Exception as fallback_exc:  # noqa: BLE001
            logger.warning("WebRetriever fallback provider failed: %s", fallback_exc)
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
