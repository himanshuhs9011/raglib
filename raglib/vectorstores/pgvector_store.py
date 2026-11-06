"""PostgreSQL + pgvector backend (``pip install raglib-py[pgvector]``; needs the ``vector`` extension).

Vectors, text and JSONB metadata live in one table; search uses cosine distance (``<=>``)
with an HNSW index when the server supports it (pgvector >= 0.5). Metadata filters use
raglib's Mongo-style syntax and are translated to parameterized SQL (values are never
interpolated into the query text).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorHit, VectorRecord

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def filter_to_sql(filters: Optional[MetadataFilter]) -> Tuple[str, List[Any]]:
    """Translate a raglib metadata filter into a SQL condition on the ``metadata`` JSONB column.

    Returns ``(sql, params)``; ``sql`` is ``"TRUE"`` for no filter. Equality uses JSONB
    containment, so ``{"page": 3}`` matches the number 3 and ``{"lang": "en"}`` the string.
    """

    if not filters:
        return "TRUE", []
    clauses: List[str] = []
    params: List[Any] = []
    for key, condition in filters.items():
        if key in {"$and", "$or"}:
            parts = [filter_to_sql(item) for item in condition]
            if not parts:
                continue
            joiner = " AND " if key == "$and" else " OR "
            clauses.append("(" + joiner.join(sql for sql, _ in parts) + ")")
            for _, sub_params in parts:
                params.extend(sub_params)
            continue
        operators = condition if isinstance(condition, dict) else {"$eq": condition}
        for operator, expected in operators.items():
            if operator == "$eq":
                clauses.append("metadata @> %s::jsonb")
                params.append(json.dumps({key: expected}))
            elif operator == "$ne":
                clauses.append("NOT (metadata @> %s::jsonb)")
                params.append(json.dumps({key: expected}))
            elif operator in {"$in", "$nin"}:
                values = list(expected)
                if not values:
                    clauses.append("FALSE" if operator == "$in" else "TRUE")
                    continue
                inner = " OR ".join("metadata @> %s::jsonb" for _ in values)
                clauses.append(f"({inner})" if operator == "$in" else f"NOT ({inner})")
                params.extend(json.dumps({key: value}) for value in values)
            elif operator in _COMPARISONS:
                symbol = _COMPARISONS[operator]
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    clauses.append(
                        f"(CASE WHEN jsonb_typeof(metadata -> %s) = 'number' THEN (metadata ->> %s)::numeric END) {symbol} %s"
                    )
                    params.extend([key, key, expected])
                else:
                    clauses.append(f"(metadata ->> %s) {symbol} %s")
                    params.extend([key, str(expected)])
            else:
                raise ValueError(f"Unsupported metadata filter operator '{operator}'")
    return (" AND ".join(clauses) or "TRUE"), params


def _vector_literal(vector: Iterable[float]) -> str:
    """Render a vector in pgvector's text format."""

    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class PGVectorStore(BaseVectorStore):
    """Vector storage in PostgreSQL with the pgvector extension.

    ``dsn`` is a libpq connection string, e.g. ``"postgresql://user:pass@localhost:5432/db"``.
    The table is created on the first upsert, sized to the embedding dimension.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table_name: str = "raglib_vectors",
        connection: Optional[Any] = None,
        create_extension: bool = True,
        hnsw_index: bool = True,
    ):
        """Connect (or wrap an existing psycopg connection)."""

        if not _IDENTIFIER_RE.match(table_name):
            raise ValueError(f"Invalid table_name '{table_name}' (letters, digits and _ only)")
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("pgvector backend requires: pip install raglib-py[pgvector]") from exc
        if connection is None and not dsn:
            raise ValueError("PGVectorStore needs dsn='postgresql://...' or connection=...")
        self._conn = connection or psycopg.connect(str(dsn), autocommit=True)
        self._table = table_name
        self._create_extension = create_extension
        self._hnsw_index = hnsw_index
        self._dimension: Optional[int] = self._existing_dimension()

    @property
    def dimension(self) -> Optional[int]:
        """Vector size of the table (None until created)."""

        return self._dimension

    def _execute(self, sql: str, params: Optional[List[Any]] = None) -> Any:
        """Run one statement and return its cursor."""

        cursor = self._conn.cursor()
        cursor.execute(sql, params or [])
        return cursor

    def _existing_dimension(self) -> Optional[int]:
        """Read the vector size of an existing table."""

        row = self._execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            [self._table],
        ).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def _ensure_table(self, size: int) -> None:
        """Create the extension, table and index on first use; refuse other vector sizes."""

        if self._dimension is None:
            if self._create_extension:
                self._execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"id TEXT PRIMARY KEY, embedding vector({int(size)}) NOT NULL, content TEXT NOT NULL, "
                "metadata JSONB NOT NULL DEFAULT '{}'::jsonb)"
            )
            if self._hnsw_index:
                try:
                    self._execute(
                        f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw ON {self._table} "
                        "USING hnsw (embedding vector_cosine_ops)"
                    )
                except Exception as exc:  # noqa: BLE001 - pgvector < 0.5 has no HNSW
                    logger.warning("HNSW index not created (%s); searches will scan the table", exc)
            self._dimension = self._existing_dimension() or size
        if size != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: table '{self._table}' holds {self._dimension}-d vectors "
                f"but got {size}-d. The embedding model changed; use another table_name."
            )

    def clear(self) -> None:
        """Drop the table (recreated on the next upsert)."""

        self._execute(f"DROP TABLE IF EXISTS {self._table}")
        self._dimension = None

    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or replace records."""

        if not records:
            return
        self._ensure_table(len(records[0].embedding))
        rows = []
        for record in records:
            if len(record.embedding) != self._dimension:
                raise ValueError(f"Embedding dimension mismatch inside one batch ({len(record.embedding)} != {self._dimension})")
            rows.append(
                (record.document_id, _vector_literal(record.embedding), record.document_text, json.dumps(record.metadata, default=str))
            )
        cursor = self._conn.cursor()
        cursor.executemany(
            f"INSERT INTO {self._table} (id, embedding, content, metadata) VALUES (%s, %s::vector, %s, %s::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, content = EXCLUDED.content, "
            "metadata = EXCLUDED.metadata",
            rows,
        )

    def delete(self, ids: Iterable[str]) -> None:
        """Delete records by id."""

        id_list = list(ids)
        if id_list and self._dimension is not None:
            self._execute(f"DELETE FROM {self._table} WHERE id = ANY(%s)", [id_list])

    def count(self) -> int:
        """Return the number of stored vectors."""

        if self._dimension is None:
            return 0
        return int(self._execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0])

    def existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return the ids already stored."""

        id_list = list(ids)
        if not id_list or self._dimension is None:
            return set()
        return {row[0] for row in self._execute(f"SELECT id FROM {self._table} WHERE id = ANY(%s)", [id_list])}

    def get_embeddings(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        """Return stored vectors by id."""

        id_list = list(ids)
        if not id_list or self._dimension is None:
            return {}
        rows = self._execute(f"SELECT id, embedding::text FROM {self._table} WHERE id = ANY(%s)", [id_list])
        return {row[0]: [float(value) for value in row[1].strip("[]").split(",") if value] for row in rows}

    def query(self, query_embedding: List[float], top_k: int, filters: Optional[MetadataFilter] = None) -> List[VectorHit]:
        """Return the nearest records (score = cosine similarity)."""

        if not query_embedding or top_k <= 0 or self._dimension is None:
            return []
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding has {len(query_embedding)} dimensions but table '{self._table}' holds "
                f"{self._dimension}-d vectors: the query uses a different embedding model."
            )
        where, params = filter_to_sql(filters)
        vector = _vector_literal(query_embedding)
        rows = self._execute(
            f"SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score FROM {self._table} "
            f"WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
            [vector, *params, vector, int(top_k)],
        )
        hits = []
        for doc_id, content, metadata, score in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
            hits.append(VectorHit(document_id=doc_id, score=float(score), document_text=content or "", metadata=meta))
        return hits

    def close(self) -> None:
        """Close the connection."""

        self._conn.close()
