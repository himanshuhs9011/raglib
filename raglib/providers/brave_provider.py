"""Brave Search API provider implementation."""

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


class BraveProvider(BaseSearchProvider):
    """Searches using the Brave Web Search API."""

    PROVIDER_NAME: str = "brave"
    ENDPOINT: str = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: Optional[str],
        country: str = "us",
        search_lang: str = "en",
        timeout: int = 10,
    ):
        """Initialize Brave credentials and request parameters."""

        self.api_key = api_key
        self.country = country
        self.search_lang = search_lang
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute Brave search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("BraveProvider requires a valid api_key")

        params = {
            "q": query,
            "count": num_results,
            "country": self.country,
            "search_lang": self.search_lang,
        }
        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("Brave request failed")
            raise ProviderError(f"Brave request failed: {exc}") from exc

        results = data.get("web", {}).get("results", [])
        docs: List[Document] = []
        for idx, row in enumerate(results[:num_results]):
            title = str(row.get("title", ""))
            description = str(row.get("description", ""))
            link = str(row.get("url", ""))
            digest = hashlib.sha256((link or description).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"brave-{idx}-{digest}",
                    content=f"{title}\n{description}".strip(),
                    metadata={"title": title, "url": link, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
