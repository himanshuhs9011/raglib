"""Rate limiting and retry policy shared by chat and embedding calls.

Hosted APIs (especially free tiers: OpenRouter ``:free`` models allow ~20 requests/minute and a
daily cap) fail with 429s when a RAG pipeline fans out parallel calls. This module gives:

* ``RateLimiter`` - a thread-safe sliding-window limiter; ``shared_limiter`` returns one instance
  per (provider, key) so chat and embedding calls draw from the same budget.
* ``retry_call`` - retries transient errors, honouring ``Retry-After`` / ``X-RateLimit-Reset``,
  and raises ``QuotaExceededError`` immediately for daily-quota / out-of-credit errors.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Errors worth retrying are transient: rate limits, timeouts, connection resets, 5xx.
_RETRYABLE_MARKERS = (
    "rate limit", "ratelimit", "429", "timeout", "timed out", "temporarily", "overloaded",
    "connection", "503", "502", "500", "504", "529", "unavailable", "resourceexhausted",
)
# Errors that waiting a few seconds cannot fix.
_QUOTA_MARKERS = (
    "per-day", "per day", "daily limit", "free-models-per-day", "insufficient_quota", "insufficient credits",
    "insufficient balance", "error code: 402", "payment required", "exceeded your current quota",
)
MAX_RETRY_WAIT_SECONDS = 90.0


class QuotaExceededError(RuntimeError):
    """The provider quota (daily cap, credits) is exhausted; retrying now will not help."""


class RateLimiter:
    """Allow at most ``rpm`` calls in any rolling 60-second window (thread-safe, blocking)."""

    def __init__(self, rpm: float, window: float = 60.0):
        """Create a limiter for ``rpm`` requests per ``window`` seconds."""

        if rpm <= 0:
            raise ValueError("rpm must be greater than zero")
        self.rpm = rpm
        self.window = window
        self._calls: Deque[float] = deque()
        self._lock = threading.Lock()
        self._paused_until = 0.0

    def acquire(self) -> None:
        """Block until a call is allowed, then record it."""

        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
                wait = max(0.0, self._paused_until - now)
                if not wait and len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return
                if not wait:
                    wait = self.window - (now - self._calls[0])
            time.sleep(min(max(wait, 0.05), 5.0))

    def pause(self, seconds: float) -> None:
        """Hold every caller for ``seconds`` (server said to back off)."""

        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)


_LIMITERS: Dict[Tuple[str, str], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(provider: str, api_key: Optional[str], rpm: float) -> RateLimiter:
    """Return the process-wide limiter for (provider, key); its rpm is the lowest requested."""

    key_id = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get((provider, key_id))
        if limiter is None:
            limiter = _LIMITERS[(provider, key_id)] = RateLimiter(rpm)
        else:
            limiter.rpm = min(limiter.rpm, rpm)
        return limiter


def is_retryable(exc: BaseException) -> bool:
    """Return True for transient provider errors."""

    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def is_quota_exhausted(exc: BaseException) -> bool:
    """Return True when the error is a daily/credit quota that a short wait cannot fix."""

    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _header(headers: Any, name: str) -> Optional[str]:
    """Case-insensitive header lookup on dict-like objects."""

    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is None:
            lowered = {str(k).lower(): v for k, v in dict(headers).items()}
            value = lowered.get(name.lower())
        return None if value is None else str(value)
    except Exception:  # noqa: BLE001
        return None


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract the server-requested wait from an exception (Retry-After, X-RateLimit-Reset, body)."""

    candidates = []
    response = getattr(exc, "response", None)
    candidates.append(getattr(response, "headers", None))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        metadata = error.get("metadata") if isinstance(error, dict) else None
        if isinstance(metadata, dict):
            if metadata.get("retry_after_seconds") is not None:
                try:
                    return max(0.0, float(metadata["retry_after_seconds"]))
                except (TypeError, ValueError):
                    pass
            candidates.append(metadata.get("headers"))
    for headers in candidates:
        value = _header(headers, "Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
        reset = _header(headers, "X-RateLimit-Reset")
        if reset:
            try:
                stamp = float(reset)
                stamp = stamp / 1000.0 if stamp > 1e11 else stamp  # milliseconds epoch
                return max(0.0, stamp - time.time())
            except ValueError:
                pass
    return None


def retry_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 2,
    base_delay: float = 1.0,
    limiter: Optional[RateLimiter] = None,
    label: str = "llm",
    **kwargs: Any,
) -> Any:
    """Call ``fn`` with rate limiting and retries for transient errors."""

    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_quota_exhausted(exc):
                raise QuotaExceededError(f"[{label}] provider quota exhausted: {str(exc)[:300]}") from exc
            if attempt >= max_retries or not is_retryable(exc):
                raise
            server_wait = retry_after_seconds(exc)
            if server_wait is not None and server_wait > MAX_RETRY_WAIT_SECONDS:
                raise QuotaExceededError(
                    f"[{label}] rate limited for {server_wait:.0f}s (likely a daily/account limit): {str(exc)[:200]}"
                ) from exc
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            if server_wait is not None:
                delay = max(delay, server_wait + 0.5)
                if limiter is not None:
                    limiter.pause(delay)
            attempt += 1
            logger.warning(
                "[%s] transient error (%s); retry %d/%d in %.1fs", label, type(exc).__name__, attempt, max_retries, delay
            )
            time.sleep(delay)
