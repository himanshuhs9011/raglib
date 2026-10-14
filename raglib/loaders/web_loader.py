"""Web page loader for URL-based source ingestion."""

from __future__ import annotations

import hashlib
import html
import logging
import re
from html.parser import HTMLParser
from typing import List
from urllib.request import Request, urlopen

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class _VisibleTextParser(HTMLParser):
    """Extracts visible text content from HTML payloads."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


class WebLoader:
    """Fetch URL content and convert it to plain text documents."""

    def load(self, url: str) -> List[Document]:
        """Load URL content and return a single extracted text Document."""

        request = Request(url, headers={"User-Agent": "raglib/0.1"})
        with urlopen(request, timeout=15) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = response.read().decode(charset, errors="ignore")

        text = self._html_to_text(raw_html)
        if not text:
            logger.warning("WebLoader found no text for url=%s", url)
            return []

        stable_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        document = Document(
            id=f"web-{stable_id}",
            content=text,
            metadata={"url": url, "loader": "web"},
            source="web",
        )
        logger.info("WebLoader loaded 1 document from url=%s", url)
        return [document]

    def _html_to_text(self, raw_html: str) -> str:
        """Convert HTML to clean visible plain text."""

        without_scripts = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
        without_styles = re.sub(r"<style[\s\S]*?</style>", " ", without_scripts, flags=re.IGNORECASE)
        without_noscript = re.sub(r"<noscript[\s\S]*?</noscript>", " ", without_styles, flags=re.IGNORECASE)

        parser = _VisibleTextParser()
        try:
            parser.feed(without_noscript)
            extracted = parser.text.strip()
        except Exception:  # noqa: BLE001
            extracted = ""

        if not extracted:
            without_tags = re.sub(r"<[^>]+>", " ", without_noscript)
            extracted = html.unescape(without_tags)

        normalized = re.sub(r"\s+", " ", html.unescape(extracted)).strip()
        return normalized
