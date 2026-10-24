"""Shared pytest configuration: the suite runs fully offline (MockLLMClient + MockEmbedding)."""

from __future__ import annotations

import pytest

from raglib.llm.providers import PROVIDERS


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default RAG() to the offline mock and hide real provider keys so no test can call a paid API."""

    monkeypatch.setenv("RAGLIB_OFFLINE", "1")
    for spec in PROVIDERS.values():
        for key in tuple(spec.env_keys) + tuple(spec.base_url_env):
            monkeypatch.delenv(key, raising=False)
