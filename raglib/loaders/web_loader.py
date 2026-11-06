"""Web page loader for URL-based source ingestion."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import logging
import os
import socket
import tempfile
import urllib.request
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from raglib.loaders.html_text import html_to_text
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class UnsafeURLError(ValueError):
    """Raised when a URL points at a private/internal network address."""


def _is_internal(address: str) -> bool:
    """Return True for loopback, private, link-local, reserved, multicast or unspecified IPs."""

    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_public(host: str, port: int) -> List[Tuple[Any, ...]]:
    """Resolve ``host`` and return its addresses, refusing hosts with any internal address."""

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve host '{host}'") from exc
    for info in infos:
        if _is_internal(str(info[4][0])):
            raise UnsafeURLError(
                f"Refusing to fetch '{host}' ({info[4][0]}): private/internal address. "
                "Pass WebLoader(allow_private_hosts=True) / RAG(allow_private_urls=True) "
                "if you trust the source (e.g. an intranet wiki)."
            )
    return infos


def _check_host_is_public(url: str) -> None:
    """Reject URLs whose scheme is not http(s) or whose host resolves to an internal address."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeURLError(f"Only http(s) URLs can be loaded, got scheme '{parsed.scheme}'")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError(f"URL has no host: {url}")
    _resolve_public(host, parsed.port or (443 if parsed.scheme == "https" else 80))


def _connect_public(host: str, port: int, timeout: Any, source_address: Any) -> socket.socket:
    """Open a TCP connection to a validated public address of ``host``.

    Resolution and connection use the same address list, so a DNS server cannot answer the
    safety check with a public IP and the connection with an internal one (DNS rebinding).
    """

    last_error: Optional[OSError] = None
    for family, socktype, proto, _canon, sockaddr in _resolve_public(host, port):
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise last_error or OSError(f"Could not connect to {host}:{port}")


class _GuardedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that only connects to validated public addresses."""

    def connect(self) -> None:  # noqa: D401
        self.sock = _connect_public(self.host, self.port, self.timeout, getattr(self, "source_address", None))


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):  # noqa: D401
        return self.do_open(_GuardedHTTPConnection, req)


_GUARDED_HANDLERS: List[Any] = [_GuardedHTTPHandler]
if hasattr(http.client, "HTTPSConnection") and hasattr(urllib.request, "HTTPSHandler"):  # no ssl in Pyodide

    class _GuardedHTTPSConnection(http.client.HTTPSConnection):
        """HTTPS connection that only connects to validated public addresses (SNI/cert use the hostname)."""

        def connect(self) -> None:  # noqa: D401
            sock = _connect_public(self.host, self.port, self.timeout, getattr(self, "source_address", None))
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)  # type: ignore[attr-defined]

    class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):  # noqa: D401
            return self.do_open(_GuardedHTTPSConnection, req, context=getattr(self, "_context", None))

    _GUARDED_HANDLERS.append(_GuardedHTTPSHandler)


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Re-validate every redirect target so a public URL cannot bounce to an internal one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        _check_host_is_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _uses_proxy(url: str) -> bool:
    """Return True when an environment proxy applies to ``url``."""

    parsed = urlparse(url)
    proxies = urllib.request.getproxies()
    if parsed.scheme not in proxies:
        return False
    return not urllib.request.proxy_bypass(parsed.hostname or "")


class WebLoader:
    """Fetch URL content and convert it to plain text documents.

    By default private/internal addresses are blocked (SSRF protection for servers that
    index user-supplied URLs): the host is resolved once, every address is checked, and the
    connection is made to a checked address. With an environment proxy (``HTTPS_PROXY``)
    the proxy resolves the host, so only the pre-check applies. Responses are capped at
    ``max_bytes``. HTML keeps its paragraph/heading/table structure; PDF responses are
    parsed with ``pdf_loader`` when one is given.
    """

    def __init__(
        self,
        allow_private_hosts: bool = False,
        max_bytes: int = 20 * 1024 * 1024,
        timeout: int = 15,
        pdf_loader: Optional[Any] = None,
    ):
        """Configure network safety limits."""

        self.allow_private_hosts = allow_private_hosts
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.pdf_loader = pdf_loader

    def _opener(self, url: str):
        """Build an opener that enforces the SSRF policy."""

        if self.allow_private_hosts:
            return build_opener()
        _check_host_is_public(url)
        if _uses_proxy(url):
            return build_opener(_SafeRedirectHandler)
        return build_opener(urllib.request.ProxyHandler({}), *_GUARDED_HANDLERS, _SafeRedirectHandler)

    def load(self, url: str) -> List[Document]:
        """Load URL content and return a single extracted text Document."""

        request = Request(url, headers={"User-Agent": "raglib/2.1"})
        with self._opener(url).open(request, timeout=self.timeout) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read(self.max_bytes + 1)
            final_url = response.geturl() if hasattr(response, "geturl") else url
        if len(payload) > self.max_bytes:
            raise ValueError(f"Response from {url} exceeds max_bytes={self.max_bytes}")

        stable_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        if content_type == "application/pdf" or payload[:5] == b"%PDF-":
            return self._load_pdf(url, payload, stable_id)

        decoded = payload.decode(charset, errors="replace")
        title = ""
        if content_type in {"text/plain", "text/markdown", "text/csv", "application/json"} or (
            content_type.startswith("text/") and "html" not in content_type
        ):
            text = decoded.strip()
        else:
            title, text = html_to_text(decoded)
        if not text:
            logger.warning("WebLoader found no text for url=%s", url)
            return []

        metadata = {"url": url, "loader": "web", "content_type": content_type or "text/html"}
        if title:
            metadata["title"] = title
        if final_url and final_url != url:
            metadata["final_url"] = final_url
        logger.info("WebLoader loaded 1 document from url=%s", url)
        return [Document(id=f"web-{stable_id}", content=text, metadata=metadata, source="web")]

    def _load_pdf(self, url: str, payload: bytes, stable_id: str) -> List[Document]:
        """Parse a PDF response with the configured PDF loader."""

        if self.pdf_loader is None:
            raise ValueError(f"{url} is a PDF; WebLoader needs pdf_loader=PDFLoader() to read it")
        handle, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(payload)
            pages = self.pdf_loader.load(path)
        finally:
            os.remove(path)
        documents = []
        for page in pages:
            metadata = {key: value for key, value in page.metadata.items() if key != "file"}
            metadata.update({"url": url, "loader": "web_pdf"})
            suffix = page.id.split("-page-", 1)[-1]
            documents.append(Document(id=f"web-{stable_id}-page-{suffix}", content=page.content, metadata=metadata, source="web"))
        return documents

    def _html_to_text(self, raw_html: str) -> str:
        """Convert HTML to structured plain text (kept for backward compatibility)."""

        return html_to_text(raw_html)[1]
