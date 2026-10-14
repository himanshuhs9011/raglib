"""SerpAPI provider implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from raglib.providers.base_provider import (
    BaseSearchProvider,
    ProviderError,
    ProviderNotConfiguredError,
)
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class SerpAPIProvider(BaseSearchProvider):
    """Searches using SerpAPI's Google engine endpoint."""

    PROVIDER_NAME: str = "serpapi"
    ENDPOINT: str = "https://serpapi.com/search.json"

    def __init__(
        self,
        api_key: Optional[str],
        engine: str = "google",
        location: str = "United States",
        timeout: int = 10,
    ):
        """Initialize SerpAPI credentials and query defaults."""

        self.api_key = api_key
        self.engine = engine
        self.location = location
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute SerpAPI search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("SerpAPIProvider requires a valid api_key")

        params = {
            "api_key": self.api_key,
            "engine": self.engine,
            "q": query,
            "location": self.location,
            "num": num_results,
        }
        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("SerpAPI request failed")
            raise ProviderError(f"SerpAPI request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("organic_results", [])[:num_results]):
            title = str(row.get("title", ""))
            snippet = str(row.get("snippet", ""))
            link = str(row.get("link", ""))
            digest = hashlib.sha256((link or snippet).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"serpapi-{idx}-{digest}",
                    content=f"{title}\n{snippet}".strip(),
                    metadata={"title": title, "url": link, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
