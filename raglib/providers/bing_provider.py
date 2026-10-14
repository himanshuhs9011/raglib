"""Bing Web Search API provider implementation."""

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


class BingProvider(BaseSearchProvider):
    """Searches using the Bing Web Search API."""

    PROVIDER_NAME: str = "bing"
    ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(
        self,
        api_key: Optional[str],
        market: str = "en-US",
        timeout: int = 10,
    ):
        """Initialize Bing credentials and query defaults."""

        self.api_key = api_key
        self.market = market
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute Bing search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("BingProvider requires a valid api_key")

        params = urllib.parse.urlencode({"q": query, "count": num_results, "mkt": self.market})
        url = f"{self.ENDPOINT}?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Ocp-Apim-Subscription-Key": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("Bing request failed")
            raise ProviderError(f"Bing request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("webPages", {}).get("value", [])[:num_results]):
            title = str(row.get("name", ""))
            snippet = str(row.get("snippet", ""))
            link = str(row.get("url", ""))
            digest = hashlib.sha256((link or snippet).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"bing-{idx}-{digest}",
                    content=f"{title}\n{snippet}".strip(),
                    metadata={"title": title, "url": link, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
