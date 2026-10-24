"""Corpus retriever: dense (vector store), sparse (BM25) or hybrid (RRF fusion of both)."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from raglib.components.bm25 import BM25Index
from raglib.core.concurrency import ReadWriteLock
from raglib.core.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.schemas import Document
from raglib.vectorstores.base_store import BaseVectorStore, MetadataFilter, VectorRecord, matches_filter
from raglib.vectorstores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("dense", "sparse", "hybrid")
FUSION_METHODS = ("relative", "rrf")


class Retriever:
    """Retrieves relevant chunks from the corpus.

    * ``mode="hybrid"`` (default when an embedding is configured): dense vector search and
      BM25 run side by side and are fused - robust to both semantic paraphrase and exact
      keyword / ID queries. ``fusion="relative"`` (default) scores every candidate on BOTH
      signals, min-max normalizes each and blends them with ``hybrid_alpha`` (weight of the
      dense score); ``fusion="rrf"`` uses rank-only Reciprocal Rank Fusion.
    * ``mode="dense"``: vector search only.  ``mode="sparse"``: BM25 only (no embedding needed).

    Adding a document whose id is already indexed is a no-op, so ingestion is idempotent.
    """

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embedding: Optional[BaseEmbedding] = None,
        top_k: int = 5,
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        fetch_k: Optional[int] = None,
        rrf_k: int = DEFAULT_RRF_K,
        fusion: str = "relative",
        hybrid_alpha: float = 0.5,
        mmr_lambda: Optional[float] = None,
        neighbor_window: int = 0,
    ):
        """Initialize indexes and ingest the initial documents."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        resolved_mode = (mode or ("hybrid" if embedding is not None else "sparse")).strip().lower()
        if resolved_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Supported: {list(RETRIEVAL_MODES)}")
        if resolved_mode in {"dense", "hybrid"} and embedding is None:
            raise ValueError(f"retrieval mode '{resolved_mode}' requires an embedding model")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if fusion not in FUSION_METHODS:
            raise ValueError(f"Unknown fusion '{fusion}'. Supported: {list(FUSION_METHODS)}")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0 and 1")

        self.top_k = top_k
        self.mode = resolved_mode
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.fusion = fusion
        self.hybrid_alpha = hybrid_alpha
        self.mmr_lambda = mmr_lambda
        self.neighbor_window = max(0, neighbor_window)

        self._embedding = embedding
        self._vector_store: Optional[BaseVectorStore] = None
        if embedding is not None:
            self._vector_store = VectorStoreFactory.build(vector_db=vector_db, vector_db_kwargs=vector_db_kwargs)
            logger.info(
                "Retriever mode=%s embedding=%s vector_store=%s",
                self.mode,
                type(embedding).__name__,
                type(self._vector_store).__name__,
            )

        self._documents: List[Document] = []
        self._doc_index: Dict[str, Document] = {}
        self._bm25 = BM25Index()
        # Queries read concurrently; add/delete mutate under the write side.
        self._lock = ReadWriteLock()
        self.add_documents(documents or [])

    # ------------------------------------------------------------------ properties
    @property
    def embedding(self) -> Optional[BaseEmbedding]:
        """Return the embedding model (None for sparse-only retrieval)."""

        return self._embedding

    @property
    def vector_store(self) -> Optional[BaseVectorStore]:
        """Return the vector store backend (None for sparse-only retrieval)."""

        return self._vector_store

    @property
    def documents(self) -> List[Document]:
        """Return the indexed chunks in insertion order (a snapshot copy)."""

        with self._lock.read():
            return list(self._documents)

    def get(self, doc_id: str) -> Optional[Document]:
        """Return an indexed chunk by id."""

        with self._lock.read():
            return self._doc_index.get(doc_id)

    # ------------------------------------------------------------------ ingestion
    def add_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Index new documents; ids already indexed are skipped. Returns the newly added docs."""

        new_docs: List[Document] = []
        seen = set()
        with self._lock.read():
            for document in documents:
                if not document.id:
                    raise ValueError("Documents must have a non-empty id to be indexed")
                if document.id in self._doc_index or document.id in seen:
                    continue
                seen.add(document.id)
                new_docs.append(document)
        if not new_docs:
            return []

        # Embedding (slow network call) happens outside the lock so queries keep running.
        if self._embedding is not None and self._vector_store is not None:
            self._index_dense(new_docs)

        with self._lock.write():
            new_docs = [document for document in new_docs if document.id not in self._doc_index]
            for document in new_docs:
                self._documents.append(document)
                self._doc_index[document.id] = document
                self._bm25.add(document.id, document.content)

        logger.info("Retriever indexed %d new documents (total=%d)", len(new_docs), len(self._documents))
        return new_docs

    def _index_dense(self, documents: List[Document]) -> None:
        """Embed and upsert documents not already present in the vector store."""

        assert self._embedding is not None and self._vector_store is not None
        already_stored = self._vector_store.existing_ids(doc.id for doc in documents)
        to_embed = [doc for doc in documents if doc.id not in already_stored]
        if not to_embed:
            return
        vectors = self._embedding.embed_documents([doc.content for doc in to_embed])
        if len(vectors) != len(to_embed):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors for {len(to_embed)} documents"
            )
        records = [
            VectorRecord(
                document_id=doc.id,
                embedding=list(vector),
                document_text=doc.content,
                metadata={**doc.metadata, "source": doc.source, "doc_id": doc.id},
            )
            for doc, vector in zip(to_embed, vectors)
        ]
        self._vector_store.upsert(records)

    def delete_documents(self, ids: Iterable[str]) -> int:
        """Remove documents from every index. Returns the number removed."""

        with self._lock.write():
            id_set = {doc_id for doc_id in ids if doc_id in self._doc_index}
            if not id_set:
                return 0
            for doc_id in id_set:
                self._bm25.remove(doc_id)
                del self._doc_index[doc_id]
            self._documents = [doc for doc in self._documents if doc.id not in id_set]
            if self._vector_store is not None:
                self._vector_store.delete(list(id_set))
        logger.info("Retriever deleted %d documents (total=%d)", len(id_set), len(self._documents))
        return len(id_set)

    # ------------------------------------------------------------------ retrieval
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[MetadataFilter] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """Return the top-k chunks for ``query`` (optionally filtered by metadata)."""

        if not isinstance(query, str) or not query.strip():
            return []
        k = top_k or self.top_k
        active_mode = (mode or self.mode).strip().lower()
        if active_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'")
        if active_mode != "sparse" and self._vector_store is None:
            active_mode = "sparse"
        candidate_k = max(self.fetch_k or 0, k * 4 if (self.mmr_lambda is not None or active_mode == "hybrid") else k)

        query_vector = self._embedding.embed_query(query) if active_mode in {"dense", "hybrid"} else None
        with self._lock.read():
            dense = self._dense_search(query_vector, candidate_k, filters) if query_vector is not None else []
            sparse = self._sparse_search(query, candidate_k, filters) if active_mode in {"sparse", "hybrid"} else []

            if active_mode == "hybrid" and self.fusion == "rrf":
                candidates = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k, list_names=["dense", "bm25"])
            elif active_mode == "hybrid":
                candidates = self._relative_score_fusion(query, query_vector, dense, sparse)
            else:
                candidates = dense or sparse

            if self.mmr_lambda is not None and len(candidates) > k:
                candidates = self._mmr(query, candidates, k)

            results = candidates[:k]
            if self.neighbor_window > 0:
                results = [self._expand_neighbors(doc) for doc in results]
        logger.debug("Retriever mode=%s returned %d docs", active_mode, len(results))
        return [
            replace(doc, metadata={**doc.metadata, "retriever": active_mode}, source=doc.source or "vector")
            for doc in results
        ]

    def _dense_search(self, query_vector: List[float], limit: int, filters: Optional[MetadataFilter]) -> List[Document]:
        """Vector search; reconstructs documents from the store when not indexed locally."""

        assert self._vector_store is not None
        hits = self._vector_store.query(query_embedding=query_vector, top_k=limit, filters=filters)
        documents: List[Document] = []
        for hit in hits:
            base = self._doc_index.get(hit.document_id)
            if base is None:
                if not hit.document_text:
                    continue
                meta = dict(hit.metadata or {})
                base = Document(
                    id=str(meta.pop("doc_id", hit.document_id)),
                    content=hit.document_text,
                    source=str(meta.pop("source", "vector")),
                    metadata=meta,
                )
            documents.append(
                replace(base, score=float(hit.score), metadata={**base.metadata, "dense_score": float(hit.score)})
            )
        return documents

    def _sparse_search(self, query: str, limit: int, filters: Optional[MetadataFilter]) -> List[Document]:
        """BM25 search honoring metadata filters."""

        allow = None
        if filters:
            allow = lambda doc_id: matches_filter(  # noqa: E731
                {**self._doc_index[doc_id].metadata, "source": self._doc_index[doc_id].source}, filters
            )
        documents: List[Document] = []
        for doc_id, score in self._bm25.search(query, top_k=limit, allow=allow):
            base = self._doc_index[doc_id]
            documents.append(replace(base, score=float(score), metadata={**base.metadata, "bm25_score": float(score)}))
        return documents

    def _relative_score_fusion(
        self,
        query: str,
        query_vector: Optional[List[float]],
        dense: List[Document],
        sparse: List[Document],
    ) -> List[Document]:
        """Score the union of candidates on both signals, min-max normalize, and blend.

        Unlike rank fusion, a document that is decisive in one signal (e.g. the only chunk
        containing an exact ID) is not out-voted by documents that are mediocre in both.
        """

        union: Dict[str, Document] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        for doc in dense:
            union.setdefault(doc.id, doc)
            dense_scores[doc.id] = doc.score
        for doc in sparse:
            union.setdefault(doc.id, doc)
            bm25_scores[doc.id] = doc.score
        if not union:
            return []

        missing_bm25 = [doc_id for doc_id in union if doc_id not in bm25_scores]
        if missing_bm25:
            bm25_scores.update(self._bm25.score(query, missing_bm25))
        missing_dense = [doc_id for doc_id in union if doc_id not in dense_scores]
        if missing_dense and query_vector is not None and self._vector_store is not None:
            stored = self._vector_store.get_embeddings(missing_dense)
            for doc_id in missing_dense:
                if doc_id in stored:
                    dense_scores[doc_id] = _cosine(query_vector, stored[doc_id])

        dense_norm = _min_max(dense_scores)
        bm25_norm = _min_max(bm25_scores)
        alpha = self.hybrid_alpha
        fused = []
        for doc_id, doc in union.items():
            score = alpha * dense_norm.get(doc_id, 0.0) + (1 - alpha) * bm25_norm.get(doc_id, 0.0)
            fused.append(
                replace(
                    doc,
                    score=round(score, 6),
                    metadata={
                        **doc.metadata,
                        "fusion": "relative",
                        "dense_score": round(dense_scores.get(doc_id, 0.0), 6),
                        "bm25_score": round(bm25_scores.get(doc_id, 0.0), 6),
                    },
                )
            )
        fused.sort(key=lambda doc: doc.score, reverse=True)
        return fused

    def _mmr(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        """Maximal Marginal Relevance: trade relevance for diversity among candidates."""

        if self._embedding is None or self._vector_store is None:
            return candidates
        vectors = self._vector_store.get_embeddings([doc.id for doc in candidates])
        if len(vectors) < len(candidates):
            missing = [doc for doc in candidates if doc.id not in vectors]
            for doc, vector in zip(missing, self._embedding.embed_documents([doc.content for doc in missing])):
                vectors[doc.id] = vector
        query_vector = self._embedding.embed_query(query)
        lam = float(self.mmr_lambda if self.mmr_lambda is not None else 0.5)

        selected: List[Document] = []
        remaining = list(candidates)
        relevance = {doc.id: _cosine(query_vector, vectors[doc.id]) for doc in remaining}
        while remaining and len(selected) < k:
            best_doc = max(
                remaining,
                key=lambda doc: lam * relevance[doc.id]
                - (1 - lam) * max((_cosine(vectors[doc.id], vectors[s.id]) for s in selected), default=0.0),
            )
            selected.append(best_doc)
            remaining.remove(best_doc)
        return selected

    def _expand_neighbors(self, doc: Document) -> Document:
        """Small-to-big: widen a hit with its neighbouring chunks from the same parent."""

        parts = [doc.content]
        current = doc
        for _ in range(self.neighbor_window):
            prev_id = current.metadata.get("prev_id")
            prev_doc = self._doc_index.get(prev_id) if prev_id else None
            if prev_doc is None:
                break
            parts.insert(0, prev_doc.content)
            current = prev_doc
        current = doc
        for _ in range(self.neighbor_window):
            next_id = current.metadata.get("next_id")
            next_doc = self._doc_index.get(next_id) if next_id else None
            if next_doc is None:
                break
            parts.append(next_doc.content)
            current = next_doc
        if len(parts) == 1:
            return doc
        return replace(doc, content="\n".join(parts), metadata={**doc.metadata, "neighbor_expanded": True})


def _min_max(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize to [0, 1]; a constant column carries no signal and maps to 0."""

    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low <= 1e-12:
        return {key: (1.0 if high > 0 and len(scores) == 1 else 0.0) for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for dense vectors."""

    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(length)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(length)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
