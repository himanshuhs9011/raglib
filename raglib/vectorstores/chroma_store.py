"""ChromaDB vector store backend for persistent semantic retrieval."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorHit, VectorRecord

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector storage and search implementation backed by ChromaDB (cosine space).

    Without ``persist_directory`` each instance gets a unique collection name, so two
    ``RAG`` objects in one process never see each other's vectors. Pass an explicit
    ``collection_name`` together with ``persist_directory`` to reopen a saved index.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """Initialize a Chroma client and open (or create) the configured collection."""

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("Chroma backend requires: pip install raglib-py[chroma]") from exc

        settings = Settings(anonymized_telemetry=False)
        if host:
            self._client = chromadb.HttpClient(host=host, port=port or 8000, settings=settings)
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory, settings=settings)
        else:
            self._client = chromadb.EphemeralClient(settings=settings)

        self._collection_name = collection_name or f"raglib_{uuid.uuid4().hex[:12]}"
        self._collection = self._open_collection()
        logger.info(
            "ChromaVectorStore collection='%s' persist_directory=%s",
            self._collection_name,
            persist_directory,
        )

    @property
    def collection_name(self) -> str:
        """Return the active collection name."""

        return self._collection_name

    def _open_collection(self):
        """Get or create the collection using cosine distance."""

        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        """Delete and recreate the active Chroma collection."""

        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Chroma delete_collection ignored: %s", exc)
        self._collection = self._open_collection()

    def upsert(self, records: List[VectorRecord]) -> None:
        """Upsert vector records into the Chroma collection."""

        if not records:
            return
        self._collection.upsert(
            ids=[record.document_id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.document_text for record in records],
            metadatas=[self._normalize_metadata(record.metadata) for record in records],
        )
        logger.debug("ChromaVectorStore upserted %d vectors", len(records))

    def delete(self, ids: Iterable[str]) -> None:
        """Delete vectors by id."""

        id_list = list(ids)
        if id_list:
            self._collection.delete(ids=id_list)

    def count(self) -> int:
        """Return the number of stored vectors."""

        return int(self._collection.count())

    def existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return ids already present in the collection."""

        id_list = list(ids)
        if not id_list:
            return set()
        result = self._collection.get(ids=id_list, include=[])
        return set(result.get("ids") or [])

    def get_embeddings(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        """Return stored embeddings for ids."""

        id_list = list(ids)
        if not id_list:
            return {}
        result = self._collection.get(ids=id_list, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        return {rid: list(vec) for rid, vec in zip(result.get("ids") or [], embeddings)}

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[VectorHit]:
        """Query top_k nearest vectors; score is cosine similarity (1 - cosine distance)."""

        if not query_embedding or top_k <= 0:
            return []
        total_count = self.count()
        if total_count <= 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total_count),
            "include": ["distances", "documents", "metadatas"],
        }
        where = self._to_chroma_where(filters)
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        hits: List[VectorHit] = []
        for index, document_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorHit(
                    document_id=document_id,
                    score=1.0 - distance,
                    document_text=str(documents[index]) if index < len(documents) and documents[index] else "",
                    metadata=self._denormalize_metadata(
                        metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
                    ),
                )
            )
        return hits

    @staticmethod
    def _to_chroma_where(filters: Optional[MetadataFilter]) -> Optional[Dict[str, Any]]:
        """Chroma requires an explicit $and when filtering on several keys."""

        if not filters:
            return None
        if len(filters) == 1:
            return dict(filters)
        return {"$and": [{key: value} for key, value in filters.items()]}

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata into Chroma-supported scalar values (JSON-encode the rest)."""

        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = "__json__" + json.dumps(value, ensure_ascii=False, default=str)
        return normalized

    @staticmethod
    def _denormalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse ``_normalize_metadata`` for JSON-encoded values."""

        restored: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str) and value.startswith("__json__"):
                try:
                    restored[key] = json.loads(value[len("__json__"):])
                    continue
                except ValueError:
                    pass
            restored[key] = value
        return restored
