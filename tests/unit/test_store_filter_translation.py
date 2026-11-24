"""Store filter translation without a server: list-valued fields in pgvector, Qdrant never-match
conditions and strict-mode payload indexes (the live versions are in tests/integration/)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

import pytest

from raglib.vectorstores.pgvector_store import filter_to_sql


def test_pgvector_equality_also_matches_an_element_of_a_list() -> None:
    sql, params = filter_to_sql({"department": "legal"})
    assert sql == "(metadata @> %s::jsonb OR metadata @> %s::jsonb)"
    assert [json.loads(p) for p in params] == [{"department": "legal"}, {"department": ["legal"]}]
    sql, params = filter_to_sql({"department": {"$nin": ["a", "b"]}})
    assert sql.startswith("NOT (") and len(params) == 4
    assert filter_to_sql({"department": {"$in": []}})[0] == "FALSE"


models = pytest.importorskip("qdrant_client.models")


def _store(client: Any = None, payload_indexes: Any = None) -> Any:
    from raglib.vectorstores.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._models = models
    store._client = client
    store._collection_name = "c"
    store._payload_indexes = payload_indexes
    store._needs_indexes = None
    store._indexed = set()
    return store


def test_qdrant_never_match_conditions_name_no_payload_key() -> None:
    store = _store()
    never = models.HasIdCondition(has_id=[])
    assert store.to_qdrant_filter({"$or": []}).must == [never]
    assert store.to_qdrant_filter({"tenant": {"$in": []}}).must == [never]
    assert store.to_qdrant_filter({"tenant": {"$nin": []}}).must_not is None  # excludes nothing


class _Client:
    def __init__(self, strict: bool) -> None:
        mode = SimpleNamespace(enabled=True, unindexed_filtering_retrieve=False) if strict else None
        self.info = SimpleNamespace(config=SimpleNamespace(strict_mode_config=mode), payload_schema={"tenant": "keyword"})
        self.created: List[Any] = []

    def get_collection(self, name: str) -> Any:
        return self.info

    def create_payload_index(self, collection_name: str, field_name: str, field_schema: Any, wait: bool) -> None:
        self.created.append((field_name, field_schema))


def test_strict_mode_collections_get_payload_indexes_for_filtered_keys() -> None:
    client = _Client(strict=True)
    store = _store(client)
    assert store._indexes_wanted() is True
    store._ensure_indexes({"$and": [{"access_level": {"$in": ["public"]}}, {"$or": [{"_acl_any_department": True}]},
                                    {"tenant": "acme"}, {"page": {"$gte": 2}}, {"score": {"$lt": 0.5}}]})
    assert dict(client.created) == {
        "access_level": models.PayloadSchemaType.KEYWORD,
        "_acl_any_department": models.PayloadSchemaType.BOOL,
        "page": models.PayloadSchemaType.INTEGER,
        "score": models.PayloadSchemaType.FLOAT,
    }  # "tenant" was already indexed
    store._ensure_indexes({"access_level": "internal"})
    assert len(client.created) == 4  # each index is created once


def test_plain_collections_get_no_indexes_unless_asked() -> None:
    assert _store(_Client(strict=False))._indexes_wanted() is False
    assert _store(_Client(strict=False), payload_indexes=True)._indexes_wanted() is True
    assert _store(_Client(strict=True), payload_indexes=False)._indexes_wanted() is False
