"""6.1: a server's Retry-After never makes a call sleep longer than MAX_RETRY_WAIT_SECONDS."""

from __future__ import annotations

from typing import List

import pytest

from raglib.llm import resilience
from raglib.llm.http import ProviderHTTPError
from raglib.llm.resilience import MAX_RETRY_WAIT_SECONDS, QuotaExceededError, retry_call


def test_a_huge_retry_after_gives_up_at_once_instead_of_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(resilience.time, "sleep", sleeps.append)
    calls = []

    def rate_limited() -> None:
        calls.append(1)
        raise ProviderHTTPError(429, {"error": "slow down"}, headers={"Retry-After": "86400"})

    with pytest.raises(QuotaExceededError, match="rate limited for 86400s"):
        retry_call(rate_limited, max_retries=5, base_delay=0.01)
    assert sleeps == [] and len(calls) == 1


def test_a_short_retry_after_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(resilience.time, "sleep", sleeps.append)
    attempts = []

    def once_limited() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise ProviderHTTPError(429, {"error": "slow down"}, headers={"Retry-After": "2"})
        return "ok"

    assert retry_call(once_limited, max_retries=2, base_delay=0.01) == "ok"
    assert len(sleeps) == 1 and 2.0 <= sleeps[0] <= MAX_RETRY_WAIT_SECONDS
