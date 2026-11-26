"""Minimal JSON-over-HTTPS transport for provider APIs (standard library only).

raglib talks to every hosted model through three wire protocols (OpenAI Chat Completions,
Anthropic Messages, Gemini ``generateContent``), so a small ``urllib`` client is all it needs:
no SDK and no LangChain. Errors carry the HTTP status, the parsed body and the response
headers, which ``raglib.llm.resilience`` uses to decide whether to retry and how long to wait.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional

if TYPE_CHECKING:  # ssl is imported on first use: Pyodide (the website playground) ships it separately
    import ssl

logger = logging.getLogger(__name__)

_USER_AGENT = "raglib-py"
_SSL_CONTEXT: Optional["ssl.SSLContext"] = None


class ProviderHTTPError(RuntimeError):
    """A provider API answered with a non-2xx status.

    ``status_code``, ``body`` (parsed JSON when possible) and ``response.headers`` follow the
    shape of the official SDK errors, so retry logic can read ``Retry-After`` the same way.
    """

    def __init__(self, status_code: int, body: Any, headers: Optional[Mapping[str, str]] = None, url: str = ""):
        """Store the status, body and headers and build a readable message."""

        self.status_code = status_code
        self.body = body
        self.headers: Dict[str, str] = dict(headers or {})
        self.response = _ResponseInfo(self.headers, status_code)
        self.url = url
        message = f"Error code: {status_code} - {_error_text(body)}"
        if status_code == 410:
            # The provider retired the model (or endpoint): retrying cannot help, a new model name can.
            message += (
                " -- the provider no longer serves this model (HTTP 410 Gone, usually end of life). "
                "Pass another model name (embedding_model=... / llm=...); the provider's model list shows the current ones."
            )
        super().__init__(message)


class ProviderConnectionError(RuntimeError):
    """The provider could not be reached (DNS, refused connection, TLS, timeout)."""


class _ResponseInfo:
    """Just enough of a response object for ``retry_after_seconds``."""

    def __init__(self, headers: Dict[str, str], status_code: int):
        self.headers = headers
        self.status_code = status_code


def _ssl_context() -> "ssl.SSLContext":
    """Default TLS context, using certifi's CA bundle when it is installed."""

    import ssl

    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi  # type: ignore[import-not-found]

            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - certifi missing or unreadable: use the system store
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _error_text(body: Any) -> str:
    """Pull the human-readable message out of an error body."""

    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or json.dumps(error)[:500]
            kind = error.get("type") or error.get("code") or error.get("status")
            return f"{kind}: {message}" if kind and str(kind) not in str(message) else str(message)
        if isinstance(error, str):
            return error
        return json.dumps(body)[:500]
    if isinstance(body, list) and body:
        return _error_text(body[0])
    return str(body)[:500]


def _parse_body(raw: bytes) -> Any:
    """Decode a JSON body, falling back to text."""

    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def _open(url: str, payload: Any, headers: Mapping[str, str], timeout: float, method: str = "POST") -> Any:
    """Send one request and return the open response, raising typed errors."""

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update({key: value for key, value in headers.items() if value is not None})
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    context = _ssl_context() if url.lower().startswith("https://") else None
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=context)  # noqa: S310 - provider URL
    except urllib.error.HTTPError as exc:
        try:
            body = _parse_body(exc.read())
        finally:
            exc.close()
        raise ProviderHTTPError(exc.code, body, dict(exc.headers.items()) if exc.headers else {}, url) from None
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderConnectionError(f"Request to {_host(url)} timed out after {timeout:.0f}s") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ProviderConnectionError(f"Request to {_host(url)} timed out after {timeout:.0f}s") from exc
        raise ProviderConnectionError(f"Connection error reaching {_host(url)}: {reason}") from exc
    except (ConnectionError, OSError) as exc:
        raise ProviderConnectionError(f"Connection error reaching {_host(url)}: {exc}") from exc


def _host(url: str) -> str:
    """Return scheme://host for error messages (never the query string, which may hold a key)."""

    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def post_json(url: str, payload: Any, headers: Mapping[str, str], timeout: float = 60.0) -> Any:
    """POST JSON and return the decoded JSON response."""

    response = _open(url, payload, headers, timeout)
    with response:
        raw = response.read()
    body = _parse_body(raw)
    if not isinstance(body, (dict, list)):
        raise ProviderHTTPError(response.status, {"error": f"Expected JSON, got: {str(body)[:200]}"}, {}, url)
    return body


def post_sse(url: str, payload: Any, headers: Mapping[str, str], timeout: float = 60.0) -> Iterator[Dict[str, Any]]:
    """POST JSON and yield each Server-Sent Event as ``{"event": ..., "data": <json>}``.

    The request is sent when the generator is first advanced, so a caller can retry
    "open the stream and read the first event" as one unit. ``data: [DONE]`` ends the stream.
    """

    response = _open(url, payload, {**headers, "Accept": "text/event-stream"}, timeout)
    with response:
        event_name: Optional[str] = None
        data_lines = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    data_lines = []
                    if data.strip() == "[DONE]":
                        return
                    yield {"event": event_name, "data": _parse_body(data.encode("utf-8"))}
                event_name = None
                continue
            if line.startswith(":"):
                continue  # SSE comment / keep-alive
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value
        if data_lines:
            data = "\n".join(data_lines)
            if data.strip() != "[DONE]":
                yield {"event": event_name, "data": _parse_body(data.encode("utf-8"))}
