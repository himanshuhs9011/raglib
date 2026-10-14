"""Tavily web provider implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import List, Optional

from raglib.providers.base_provider import (
    BaseSearchProvider,
    ProviderError,
    ProviderNotConfiguredError,
)
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class TavilyProvider(BaseSearchProvider):
    """Searches the web through the Tavily API."""

    PROVIDER_NAME: str = "tavily"
    ENDPOINT: str = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str],
        topic: str = "general",
        search_depth: str = "basic",
        include_answer: bool = False,
        timeout: int = 10,
    ):
        """Initialize Tavily credentials and request options."""

        self.api_key = api_key
        self.topic = topic
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute Tavily search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("TavilyProvider requires a valid api_key")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
        }

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.ENDPOINT,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("Tavily request failed")
            raise ProviderError(f"Tavily request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("results", [])[:num_results]):
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            url = str(row.get("url", ""))
            digest = hashlib.sha256((url or content).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"tavily-{idx}-{digest}",
                    content=f"{title}\n{content}".strip(),
                    metadata={"title": title, "url": url, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
