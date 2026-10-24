"""Web page loader for URL-based source ingestion."""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import html
import logging
import re
from html.parser import HTMLParser
from typing import List
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

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


class UnsafeURLError(ValueError):
    """Raised when a URL points at a private/internal network address."""


def _check_host_is_public(url: str) -> None:
    """Reject URLs that resolve to loopback, private, link-local or reserved addresses (SSRF guard)."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeURLError(f"Only http(s) URLs can be loaded, got scheme '{parsed.scheme}'")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError(f"URL has no host: {url}")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve host '{host}'") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeURLError(
                f"Refusing to fetch '{host}' ({address}): private/internal address. "
                "Pass WebLoader(allow_private_hosts=True) / DocumentLoader(allow_private_urls=True) "
                "if you trust the source (e.g. an intranet wiki)."
            )


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Re-validate every redirect target so a public URL cannot bounce to an internal one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        _check_host_is_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebLoader:
    """Fetch URL content and convert it to plain text documents.

    By default private/internal addresses are blocked (SSRF protection for servers that
    index user-supplied URLs) and responses are capped at ``max_bytes``.
    """

    def __init__(self, allow_private_hosts: bool = False, max_bytes: int = 20 * 1024 * 1024, timeout: int = 15):
        """Configure network safety limits."""

        self.allow_private_hosts = allow_private_hosts
        self.max_bytes = max_bytes
        self.timeout = timeout

    def load(self, url: str) -> List[Document]:
        """Load URL content and return a single extracted text Document."""

        request = Request(url, headers={"User-Agent": "raglib/2.0"})
        if self.allow_private_hosts:
            opener = build_opener()
        else:
            _check_host_is_public(url)
            opener = build_opener(_SafeRedirectHandler)
        with opener.open(request, timeout=self.timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read(self.max_bytes + 1)
            if len(payload) > self.max_bytes:
                raise ValueError(f"Response from {url} exceeds max_bytes={self.max_bytes}")
            raw_html = payload.decode(charset, errors="ignore")

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
