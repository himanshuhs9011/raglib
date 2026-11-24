"""Live vector-store fixtures: PostgreSQL + pgvector and Qdrant (plain and strict mode).

Start the servers with ``docker compose -f docker-compose.test.yml up -d --wait``. The defaults
match that file; override them with ``RAGLIB_TEST_PG_DSN`` and ``RAGLIB_TEST_QDRANT_URL``.
A server that is not running skips its tests, unless ``RAGLIB_REQUIRE_LIVE_STORES=1`` (CI),
which turns the skip into a failure so a broken service cannot pass silently.
"""

from __future__ import annotations

import functools
import os
import uuid
from typing import Any, Callable, Iterator, List

import pytest

PG_DSN = os.getenv("RAGLIB_TEST_PG_DSN", "postgresql://postgres:postgres@localhost:5433/postgres")
QDRANT_URL = os.getenv("RAGLIB_TEST_QDRANT_URL", "http://localhost:6333")
REQUIRED = os.getenv("RAGLIB_REQUIRE_LIVE_STORES", "").strip().lower() in {"1", "true", "yes"}
STORES = ["pgvector", "qdrant", "qdrant_strict"]


def _unavailable(reason: str) -> None:
    if REQUIRED:
        pytest.fail(f"live store required (RAGLIB_REQUIRE_LIVE_STORES=1): {reason}")
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _pg_problem() -> str:
    """Why PostgreSQL cannot be used ("" when it can); probed once per test session."""

    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        return "pip install raglib-py[pgvector] to run the pgvector tests"
    try:
        psycopg.connect(PG_DSN, connect_timeout=2).close()
    except Exception as exc:  # noqa: BLE001
        return f"PostgreSQL not reachable at {PG_DSN.split('@')[-1]}: {type(exc).__name__}"
    return ""


@functools.lru_cache(maxsize=None)
def _qdrant() -> Any:
    """A client, or the reason Qdrant cannot be used; probed once per test session."""

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return "pip install raglib-py[qdrant] to run the Qdrant tests"
    client = QdrantClient(url=QDRANT_URL, timeout=2)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        return f"Qdrant not reachable at {QDRANT_URL}: {type(exc).__name__}"
    return client


def _pg_ready() -> None:
    problem = _pg_problem()
    if problem:
        _unavailable(problem)


def _qdrant_client() -> Any:
    client = _qdrant()
    if isinstance(client, str):
        _unavailable(client)
    return client


def strict_mode_collection(client: Any, name: str, dimension: int) -> None:
    """A collection as Qdrant Cloud creates it with strict mode: filtering on an unindexed payload
    key is rejected (``unindexed_filtering_retrieve=False``)."""

    from qdrant_client import models

    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        strict_mode_config=models.StrictModeConfig(enabled=True, unindexed_filtering_retrieve=False),
    )


@pytest.fixture(params=STORES)
def make_store(request: pytest.FixtureRequest) -> Iterator[Callable[[int], Any]]:
    """``make_store(dimension)`` -> a new, empty store of the parametrized kind (dropped afterwards)."""

    kind = request.param
    created: List[Any] = []
    if kind == "pgvector":
        _pg_ready()
    else:
        client = _qdrant_client()

    def factory(dimension: int) -> Any:
        suffix = uuid.uuid4().hex[:10]
        if kind == "pgvector":
            from raglib.vectorstores.pgvector_store import PGVectorStore

            store = PGVectorStore(PG_DSN, table_name=f"raglib_it_{suffix}")
        else:
            from raglib.vectorstores.qdrant_store import QdrantVectorStore

            name = f"raglib_it_{suffix}"
            if kind == "qdrant_strict":
                strict_mode_collection(client, name, dimension)
            store = QdrantVectorStore(collection_name=name, client=client)
        created.append(store)
        return store

    factory.kind = kind  # type: ignore[attr-defined]
    yield factory
    for store in created:
        try:
            store.clear()
            if hasattr(store, "close"):
                store.close()
        except Exception:  # noqa: BLE001 - cleanup must not hide the test result
            pass
