"""Qdrant vector store backend (``pip install raglib-py[qdrant]``)."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorHit, VectorRecord

logger = logging.getLogger(__name__)

_ID_KEY = "_raglib_id"
_TEXT_KEY = "_raglib_text"
_NAMESPACE = uuid.UUID("6f1c3c7e-5f53-4a8e-9a43-3b0d9f2f1a10")


def _point_id(document_id: str) -> str:
    """Qdrant ids must be UUIDs or integers: derive a stable UUID from the document id."""

    return str(uuid.uuid5(_NAMESPACE, document_id))


class QdrantVectorStore(BaseVectorStore):
    """Vector storage in Qdrant (cosine distance).

    * ``location=":memory:"`` (default without ``url``/``path``) - in-process, for tests.
    * ``path="./qdrant"`` - embedded, persisted to disk.
    * ``url="http://localhost:6333"`` (+ ``api_key``) - a Qdrant server or Qdrant Cloud.

    The collection is created on the first upsert, sized to the embedding dimension.
    Metadata filters use raglib's Mongo-style syntax (``$eq``, ``$ne``, ``$in``, ``$nin``,
    ``$gt``/``$gte``/``$lt``/``$lte``, ``$and``, ``$or``) and run inside Qdrant.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        location: Optional[str] = None,
        client: Optional[Any] = None,
        **client_kwargs: Any,
    ):
        """Connect to Qdrant (or wrap an existing ``QdrantClient``)."""

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError("Qdrant backend requires: pip install raglib-py[qdrant]") from exc

        self._models = models
        if client is not None:
            self._client = client
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key, **client_kwargs)
        elif path:
            self._client = QdrantClient(path=path, **client_kwargs)
        else:
            self._client = QdrantClient(location=location or ":memory:", **client_kwargs)
        self._collection_name = collection_name or f"raglib_{uuid.uuid4().hex[:12]}"
        self._dimension: Optional[int] = None
        if self._client.collection_exists(self._collection_name):
            info = self._client.get_collection(self._collection_name)
            vectors = info.config.params.vectors
            self._dimension = getattr(vectors, "size", None)
        logger.info("QdrantVectorStore collection='%s'", self._collection_name)

    @property
    def collection_name(self) -> str:
        """Return the collection name."""

        return self._collection_name

    @property
    def dimension(self) -> Optional[int]:
        """Vector size of the collection (None until the first upsert)."""

        return self._dimension

    def _ensure_collection(self, size: int) -> None:
        """Create the collection on first use, and refuse vectors of another size."""

        if self._dimension is None:
            if not self._client.collection_exists(self._collection_name):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=self._models.VectorParams(size=size, distance=self._models.Distance.COSINE),
                )
            self._dimension = size
        elif size != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: collection '{self._collection_name}' holds {self._dimension}-d "
                f"vectors but got {size}-d. The embedding model changed; use a new collection."
            )

    def _exists(self) -> bool:
        return self._dimension is not None

    def clear(self) -> None:
        """Delete the collection (it is recreated on the next upsert)."""

        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._dimension = None

    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or replace records."""

        if not records:
            return
        self._ensure_collection(len(records[0].embedding))
        points = []
        for record in records:
            if len(record.embedding) != self._dimension:
                raise ValueError(f"Embedding dimension mismatch inside one batch ({len(record.embedding)} != {self._dimension})")
            payload = {**record.metadata, _ID_KEY: record.document_id, _TEXT_KEY: record.document_text}
            points.append(self._models.PointStruct(id=_point_id(record.document_id), vector=list(record.embedding), payload=payload))
        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    def delete(self, ids: Iterable[str]) -> None:
        """Delete records by document id."""

        id_list: List[Any] = [_point_id(doc_id) for doc_id in ids]
        if id_list and self._exists():
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=self._models.PointIdsList(points=id_list),
                wait=True,
            )

    def count(self) -> int:
        """Return the number of stored vectors."""

        if not self._exists():
            return 0
        return int(self._client.count(collection_name=self._collection_name, exact=True).count)

    def existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return the document ids already stored."""

        id_list = list(ids)
        if not id_list or not self._exists():
            return set()
        points = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[_point_id(doc_id) for doc_id in id_list],
            with_payload=[_ID_KEY],
            with_vectors=False,
        )
        return {str(point.payload.get(_ID_KEY)) for point in points if point.payload}

    def get_embeddings(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        """Return stored vectors by document id."""

        id_list = list(ids)
        if not id_list or not self._exists():
            return {}
        points = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[_point_id(doc_id) for doc_id in id_list],
            with_payload=[_ID_KEY],
            with_vectors=True,
        )
        return {str(point.payload.get(_ID_KEY)): list(point.vector) for point in points if point.payload and point.vector is not None}

    def query(self, query_embedding: List[float], top_k: int, filters: Optional[MetadataFilter] = None) -> List[VectorHit]:
        """Return the nearest records (score = cosine similarity)."""

        if not query_embedding or top_k <= 0 or not self._exists():
            return []
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding has {len(query_embedding)} dimensions but collection '{self._collection_name}' "
                f"holds {self._dimension}-d vectors: the query uses a different embedding model."
            )
        query_filter = self.to_qdrant_filter(filters) if filters else None
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_embedding),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            document_id = str(payload.pop(_ID_KEY, point.id))
            text = str(payload.pop(_TEXT_KEY, ""))
            hits.append(VectorHit(document_id=document_id, score=float(point.score), document_text=text, metadata=payload))
        return hits

    def to_qdrant_filter(self, filters: MetadataFilter) -> Any:
        """Translate a Mongo-style raglib filter into a Qdrant ``Filter``."""

        models = self._models
        must: List[Any] = []
        must_not: List[Any] = []
        for key, condition in filters.items():
            if key == "$and":
                must.extend(self.to_qdrant_filter(item) for item in condition)
                continue
            if key == "$or":
                must.append(models.Filter(should=[self.to_qdrant_filter(item) for item in condition]))
                continue
            operators = condition if isinstance(condition, dict) else {"$eq": condition}
            for operator, expected in operators.items():
                if operator == "$eq":
                    must.append(models.FieldCondition(key=key, match=models.MatchValue(value=expected)))
                elif operator == "$ne":
                    must_not.append(models.FieldCondition(key=key, match=models.MatchValue(value=expected)))
                elif operator == "$in":
                    must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(expected))))
                elif operator == "$nin":
                    must_not.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(expected))))
                elif operator in {"$gt", "$gte", "$lt", "$lte"}:
                    must.append(models.FieldCondition(key=key, range=models.Range(**{operator[1:]: expected})))
                else:
                    raise ValueError(f"Unsupported metadata filter operator '{operator}'")
        return models.Filter(must=must or None, must_not=must_not or None)
