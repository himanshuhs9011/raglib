"""DuckDuckGo free web provider implementation."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional

from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(BaseSearchProvider):
    """Performs web search using LangChain community DuckDuckGo tools."""

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
        logger.info("Running DuckDuckGo search for query=%s", query)

        documents = self._search_langchain_structured(query=query, num_results=top_k)
        if documents:
            return documents

        try:
            raw_result = self._run_langchain_search(query=query, num_results=top_k)
            documents = self._parse_run_output(raw_result=raw_result, num_results=top_k)
            if documents:
                return documents
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGoSearchRun failed, falling back to ddgs client: %s", exc)

        return self._search_ddgs_fallback(query=query, num_results=top_k)

    def _run_langchain_search(self, query: str, num_results: int) -> str:
        """Run DuckDuckGoSearchRun and validate response before further processing."""

        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        except ImportError as exc:
            message = (
                "langchain-community and ddgs are required for DuckDuckGo default search. "
                "Install with: pip install langchain-community ddgs"
            )
            logger.error(message)
            raise ProviderError(message) from exc

        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                region=self.region,
                safesearch=self.safesearch,
                time=self.timelimit,
                max_results=num_results,
                backend=self.backend,
            )
            tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
            result = tool.invoke(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DuckDuckGoSearchRun failed")
            raise ProviderError(f"DuckDuckGoSearchRun failed: {exc}") from exc

        if not isinstance(result, str) or not result.strip():
            raise ProviderError("DuckDuckGoSearchRun returned an empty result")

        return result.strip()

    def _search_langchain_structured(self, query: str, num_results: int) -> List[Document]:
        """Try DuckDuckGoSearchResults for title/url/snippet-rich result rows."""

        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        except ImportError:
            return []

        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                region=self.region,
                safesearch=self.safesearch,
                time=self.timelimit,
                max_results=num_results,
                backend=self.backend,
            )
            tool = DuckDuckGoSearchResults(
                api_wrapper=wrapper,
                output_format="list",
                backend=self.backend,
            )
            rows = tool.invoke(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGoSearchResults failed, falling back to parsed run output: %s", exc)
            return []

        if isinstance(rows, list) and not rows:
            try:
                fallback_tool = DuckDuckGoSearchResults(output_format="list")
                fallback_rows = fallback_tool.invoke(query)
                if isinstance(fallback_rows, list):
                    rows = fallback_rows
            except Exception as exc:  # noqa: BLE001
                logger.debug("DuckDuckGoSearchResults default invocation failed: %s", exc)

        if not isinstance(rows, list):
            return []

        documents: List[Document] = []
        for idx, row in enumerate(rows[:num_results]):
            if not isinstance(row, dict):
                continue

            title = str(row.get("title", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            url = str(row.get("link", "")).strip()
            content = snippet or title
            if not content:
                continue

            score = max(0.0, 1.0 - (idx / max(num_results, 1)))
            digest = hashlib.sha256((url or content).encode("utf-8")).hexdigest()[:16]
            doc_id = f"ddg-lc-{idx}-{digest}"
            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    score=score,
                    metadata={
                        "title": title or self._derive_title(content),
                        "snippet": snippet or content,
                        "url": url,
                        "provider": self.name,
                    },
                    source=self.name,
                )
            )

        return documents

    def _parse_run_output(self, raw_result: str, num_results: int) -> List[Document]:
        """Convert plain-text DuckDuckGoSearchRun output into Document rows."""

        if not raw_result.strip():
            return []

        splitter = re.compile(
            r"\s+\d+\s+(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago\s+-\s+",
            flags=re.IGNORECASE,
        )
        parts = [part.strip() for part in splitter.split(raw_result) if part.strip()]

        if not parts:
            parts = [raw_result.strip()]

        documents: List[Document] = []
        for idx, snippet in enumerate(parts[:num_results]):
            title = self._derive_title(snippet)
            score = max(0.0, 1.0 - (idx / max(num_results, 1)))
            digest = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
            doc_id = f"ddg-run-{idx}-{digest}"
            documents.append(
                Document(
                    id=doc_id,
                    content=snippet,
                    score=score,
                    metadata={
                        "title": title,
                        "snippet": snippet,
                        "url": "",
                        "provider": self.name,
                    },
                    source=self.name,
                )
            )

        return documents

    def _search_ddgs_fallback(self, query: str, num_results: int) -> List[Document]:
        """Final fallback path using ddgs direct client."""

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

        logger.info("DuckDuckGo provider using ddgs fallback for query=%s", query)
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
