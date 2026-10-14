"""ChromaDB vector store backend for persistent semantic retrieval."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from raglib.vectorstores.base_store import BaseVectorStore, VectorHit, VectorRecord

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector storage and search implementation backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "raglib",
        persist_directory: Optional[str] = None,
    ):
        """Initialize Chroma client and open the configured collection."""

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("pip install chromadb") from exc

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))

        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        logger.info(
            "ChromaVectorStore initialized collection='%s' persist_directory=%s",
            collection_name,
            persist_directory,
        )

    def clear(self) -> None:
        """Delete and recreate the active Chroma collection."""

        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def upsert(self, records: List[VectorRecord]) -> None:
        """Upsert vector records into Chroma collection."""

        if not records:
            return

        ids = [record.document_id for record in records]
        embeddings = [record.embedding for record in records]
        documents = [record.document_text for record in records]
        metadatas = [self._normalize_metadata(record.metadata) for record in records]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("ChromaVectorStore upserted %d vectors", len(records))

    def query(self, query_embedding: List[float], top_k: int) -> List[VectorHit]:
        """Query top_k nearest vectors from Chroma collection."""

        if not query_embedding or top_k <= 0:
            return []

        try:
            total_count = self._collection.count()
        except Exception:  # noqa: BLE001
            total_count = top_k

        if total_count <= 0:
            return []

        n_results = min(top_k, total_count)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances", "documents", "metadatas"],
        )

        ids_rows = result.get("ids") or [[]]
        distances_rows = result.get("distances") or [[]]
        docs_rows = result.get("documents") or [[]]
        metas_rows = result.get("metadatas") or [[]]

        ids = ids_rows[0] if ids_rows else []
        distances = distances_rows[0] if distances_rows else []
        documents = docs_rows[0] if docs_rows else []
        metadatas = metas_rows[0] if metas_rows else []

        hits: List[VectorHit] = []
        for index, document_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = 1.0 / (1.0 + max(distance, 0.0))
            doc_text = str(documents[index]) if index < len(documents) and documents[index] is not None else ""
            raw_meta = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            hits.append(
                VectorHit(
                    document_id=document_id,
                    score=score,
                    document_text=doc_text,
                    metadata=dict(raw_meta),
                )
            )

        return hits

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata into Chroma-supported scalar values."""

        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = json.dumps(value, ensure_ascii=True)
        return normalized
