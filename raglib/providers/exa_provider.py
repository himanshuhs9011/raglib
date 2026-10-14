"""Exa neural search provider implementation."""

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


class ExaProvider(BaseSearchProvider):
    """Searches using the Exa API."""

    PROVIDER_NAME: str = "exa"
    ENDPOINT: str = "https://api.exa.ai/search"

    def __init__(
        self,
        api_key: Optional[str],
        use_autoprompt: bool = True,
        timeout: int = 10,
    ):
        """Initialize Exa credentials and behavior options."""

        self.api_key = api_key
        self.use_autoprompt = use_autoprompt
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute Exa search and map native results into Documents."""

        if not self.api_key:
            raise ProviderNotConfiguredError("ExaProvider requires a valid api_key")

        payload = {
            "query": query,
            "numResults": num_results,
            "useAutoprompt": self.use_autoprompt,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            logger.exception("Exa request failed")
            raise ProviderError(f"Exa request failed: {exc}") from exc

        docs: List[Document] = []
        for idx, row in enumerate(data.get("results", [])[:num_results]):
            title = str(row.get("title", ""))
            text = str(row.get("text", ""))
            url = str(row.get("url", ""))
            digest = hashlib.sha256((url or text).encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    id=f"exa-{idx}-{digest}",
                    content=f"{title}\n{text}".strip(),
                    metadata={"title": title, "url": url, "provider": self.name},
                    source=self.name,
                )
            )

        return docs
