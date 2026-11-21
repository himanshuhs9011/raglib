"""DuckDuckGo free web provider implementation."""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(BaseSearchProvider):
    """Free web search through the ``ddgs`` package (no API key)."""

    PROVIDER_NAME: str = "duckduckgo"

    def __init__(
        self,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "auto",
    ):
        """Initialize DuckDuckGo provider configuration."""

        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.backend = backend

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute DuckDuckGo search and map results to Document objects."""

        if not isinstance(query, str) or not query.strip():
            return []

        top_k = max(1, int(num_results))
        logger.debug("Running DuckDuckGo search")

        return self._search_ddgs(query=query, num_results=top_k)

    def _search_ddgs(self, query: str, num_results: int) -> List[Document]:
        """Search with the ddgs client (``pip install raglib-py[web]``)."""

        try:
            from ddgs import DDGS  # type: ignore[import-not-found]
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore[import-not-found]
            except ImportError as exc:
                message = (
                    "No DuckDuckGo client package found. Install one of: "
                    "pip install ddgs OR pip install duckduckgo-search"
                )
                logger.error(message)
                raise ProviderError(message) from exc

        except Exception as exc:  # noqa: BLE001
            message = (
                "Failed to initialize DuckDuckGo search client. "
                "Install one of: pip install ddgs OR pip install duckduckgo-search"
            )
            logger.error(message)
            raise ProviderError(message) from exc

        logger.debug("DuckDuckGo provider using ddgs")
        documents: List[Document] = []

        try:
            with DDGS() as ddgs:
                search_kwargs = {
                    "region": self.region,
                    "safesearch": self.safesearch,
                    "timelimit": self.timelimit,
                    "max_results": num_results,
                }
                if self.backend:
                    search_kwargs["backend"] = self.backend

                rows = ddgs.text(query, **search_kwargs)
                for idx, row in enumerate((rows or [])[:num_results]):
                    title = str(row.get("title", ""))
                    snippet = str(row.get("body", ""))
                    url = str(row.get("href", ""))
                    content = snippet or title
                    digest = hashlib.sha256((url or content).encode("utf-8")).hexdigest()[:16]
                    doc_id = f"ddg-{idx}-{digest}"
                    documents.append(
                        Document(
                            id=doc_id,
                            content=content,
                            score=max(0.0, 1.0 - (idx / max(num_results, 1))),
                            metadata={
                                "title": title or self._derive_title(content),
                                "snippet": snippet or content,
                                "url": url,
                                "provider": self.name,
                            },
                            source=self.name,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DuckDuckGo search failed")
            raise ProviderError(f"DuckDuckGo search failed: {exc}") from exc

        return documents

    def _derive_title(self, snippet: str) -> str:
        """Build a short title from snippet text when title is missing."""

        cleaned = " ".join(snippet.split())
        if not cleaned:
            return "DuckDuckGo result"
        sentence = cleaned.split(".")[0].strip()
        candidate = sentence or cleaned
        return candidate[:120]
