"""Google Custom Search Engine provider implementation."""

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


class GoogleCSEProvider(BaseSearchProvider):
    """Searches using Google Custom Search JSON API."""

    PROVIDER_NAME: str = "google_cse"
    ENDPOINT: str = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: Optional[str],
        cse_id: Optional[str],
        safe: str = "off",
        timeout: int = 10,
    ):
        """Initialize Google CSE credentials and request options."""

        self.api_key = api_key
        self.cse_id = cse_id
        self.safe = safe
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute Google CSE search and map native results into Documents."""

        if not self.api_key or not self.cse_id:
            raise ProviderNotConfiguredError(
                "GoogleCSEProvider requires both api_key and cse_id"
            )

        params = urllib.parse.urlencode(
            {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(num_results, 10),
                "safe": self.safe,
            }
        )
        url = f"{self.ENDPOINT}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("Google CSE request failed")
            raise ProviderError(f"Google CSE request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("items", [])[:num_results]):
            title = str(row.get("title", ""))
            snippet = str(row.get("snippet", ""))
            link = str(row.get("link", ""))
            digest = hashlib.sha256((link or snippet).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"google-cse-{idx}-{digest}",
                    content=f"{title}\n{snippet}".strip(),
                    metadata={"title": title, "url": link, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
