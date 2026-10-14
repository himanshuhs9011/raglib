"""SearxNG provider implementation for self-hosted instances."""

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


class SearxNGProvider(BaseSearchProvider):
    """Searches using a configurable SearxNG endpoint."""

    PROVIDER_NAME: str = "searxng"

    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        categories: str = "general",
        language: str = "en",
        timeout: int = 10,
    ):
        """Initialize SearxNG credentials and endpoint settings."""

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.categories = categories
        self.language = language
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute SearxNG search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("SearxNGProvider requires a valid api_key")

        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "categories": self.categories,
                "language": self.language,
            }
        )
        url = f"{self.base_url}/search?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("SearxNG request failed")
            raise ProviderError(f"SearxNG request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("results", [])[:num_results]):
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            url = str(row.get("url", ""))
            digest = hashlib.sha256((url or content).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"searxng-{idx}-{digest}",
                    content=f"{title}\n{content}".strip(),
                    metadata={"title": title, "url": url, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
