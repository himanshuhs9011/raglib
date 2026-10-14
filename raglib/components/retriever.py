"""In-memory vector retriever with optional FAISS acceleration and incremental vector indexing."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.schemas import Document
from raglib.vectorstores.base_store import BaseVectorStore, VectorRecord
from raglib.vectorstores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant documents from a vector store or in-memory corpus."""

    def __init__(
        self,
        documents: List[Document],
        embedding: Optional[BaseEmbedding] = None,
        top_k: int = 5,
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize retriever state and optional FAISS backend."""

        self.top_k = top_k
        self._embedding = embedding
        self._vector_store: Optional[BaseVectorStore] = None
        self._documents: List[Document] = []
        self._vector_document_lookup: Dict[str, Document] = {}
        self._doc_tokens: List[List[str]] = []
        self._doc_tfidf: List[Dict[str, float]] = []
        self._document_frequency: Counter[str] = Counter()
        self._faiss_index = None
        self._faiss_vocabulary: List[str] = []
        self._faiss_ready = False
        self._faiss_import_error: Optional[Exception] = None

        if self._embedding is not None:
            logger.info("Retriever using embedding model=%s", type(self._embedding).__name__)
            self._vector_store = VectorStoreFactory.build(
                vector_db=vector_db,
                vector_db_kwargs=vector_db_kwargs,
            )
            logger.info("Retriever using vector store=%s", type(self._vector_store).__name__)

        self._detect_faiss()
        self.add_documents(documents)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the corpus and incrementally upsert new vectors."""

        if not documents:
            return
        start_index = len(self._documents)
        self._documents.extend(documents)
        logger.info("Retriever added %d documents (total=%d)", len(documents), len(self._documents))
        self._reindex(new_documents=documents, start_index=start_index)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k documents ranked by query similarity."""

        logger.debug("Retriever received query=%s", query)
        if not query or not query.strip():
            logger.info("Retriever returning empty result for empty query")
            return []

        if not self._tokenize(query):
            logger.info("Retriever returning empty result for query with no indexable tokens")
            return []

        if self._embedding is not None and self._vector_store is not None:
            embedding_docs = self._retrieve_with_embedding(query)
            if embedding_docs:
                logger.info("Retriever returned %d documents via embedding search", len(embedding_docs))
                return embedding_docs
            logger.warning("Embedding retrieval produced no results; falling back to sparse search")

        if not self._documents:
            logger.warning("Retriever called with empty corpus")
            return []

        if self._faiss_ready:
            faiss_docs = self._retrieve_with_faiss(query)
            if faiss_docs:
                logger.info("Retriever returned %d documents", len(faiss_docs))
                return faiss_docs

        docs = self._retrieve_with_sparse_cosine(query)
        logger.info("Retriever returned %d documents", len(docs))
        return docs

    def _detect_faiss(self) -> None:
        """Detect FAISS availability without requiring it at runtime."""

        try:
            import faiss  # type: ignore # noqa: F401
            import numpy  # type: ignore # noqa: F401

            logger.info("FAISS backend detected and will be used when possible")
        except Exception as exc:  # noqa: BLE001
            self._faiss_import_error = exc
            logger.info("FAISS backend unavailable; using pure Python retriever")

    def _reindex(
        self,
        new_documents: Optional[List[Document]] = None,
        start_index: int = 0,
    ) -> None:
        """Recompute sparse vectors and incrementally index new dense vectors."""

        self._doc_tokens = [self._tokenize(doc.content) for doc in self._documents]
        self._document_frequency = Counter()

        for tokens in self._doc_tokens:
            for token in set(tokens):
                self._document_frequency[token] += 1

        self._doc_tfidf = []
        for tokens in self._doc_tokens:
            tf = Counter(tokens)
            self._doc_tfidf.append(self._counter_to_tfidf(tf, len(tokens)))

        docs_to_embed = new_documents if new_documents is not None else self._documents
        offset = start_index if new_documents is not None else 0

        if self._embedding is not None and self._vector_store is not None and docs_to_embed:
            try:
                texts = [document.content for document in docs_to_embed]
                vectors = self._embedding.embed_documents(texts)
                records: List[VectorRecord] = []

                limit = min(len(docs_to_embed), len(vectors))
                for index in range(limit):
                    document = docs_to_embed[index]
                    vector = vectors[index]
                    vector_id = self._build_vector_id(document=document, index=offset + index)
                    records.append(
                        VectorRecord(
                            document_id=vector_id,
                            embedding=vector,
                            document_text=document.content,
                            metadata={
                                **document.metadata,
                                "source": document.source,
                                "doc_id": document.id,
                            },
                        )
                    )
                    self._vector_document_lookup[vector_id] = document

                self._vector_store.upsert(records)
                logger.info(
                    "Retriever indexed %d embedding vectors in %s (dim=%d)",
                    len(records),
                    type(self._vector_store).__name__,
                    self._embedding.dimension,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding indexing failed; using sparse fallback only: %s", exc)

        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Build FAISS index if optional dependencies are available."""

        if self._embedding is not None:
            self._faiss_ready = False
            self._faiss_index = None
            return

        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            self._faiss_ready = False
            self._faiss_index = None
            return

        if not self._documents:
            self._faiss_ready = False
            self._faiss_index = None
            return

        vocabulary = sorted(self._document_frequency.keys())
        if not vocabulary:
            self._faiss_ready = False
            self._faiss_index = None
            return

        dim = len(vocabulary)
        matrix = np.zeros((len(self._documents), dim), dtype="float32")
        for row_index, doc_vector in enumerate(self._doc_tfidf):
            for term_index, term in enumerate(vocabulary):
                matrix[row_index, term_index] = float(doc_vector.get(term, 0.0))

        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        self._faiss_index = index
        self._faiss_vocabulary = vocabulary
        self._faiss_ready = True
        logger.debug("FAISS index built with %d docs and dim=%d", len(self._documents), dim)

    def _retrieve_with_faiss(self, query: str) -> List[Document]:
        """Retrieve documents using FAISS inner-product search."""

        if not self._faiss_ready or self._faiss_index is None:
            return []

        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return []

        query_tokens = self._tokenize(query)
        tf = Counter(query_tokens)
        query_vector = self._counter_to_tfidf(tf, len(query_tokens))
        dense = np.zeros((1, len(self._faiss_vocabulary)), dtype="float32")
        for term_index, term in enumerate(self._faiss_vocabulary):
            dense[0, term_index] = float(query_vector.get(term, 0.0))

        faiss.normalize_L2(dense)
        distances, indices = self._faiss_index.search(dense, min(self.top_k, len(self._documents)))

        ranked: List[Tuple[float, int]] = []
        for rank_idx, doc_idx in enumerate(indices[0].tolist()):
            if doc_idx < 0:
                continue
            ranked.append((float(distances[0][rank_idx]), int(doc_idx)))

        return self._to_scored_documents(ranked)

    def _retrieve_with_sparse_cosine(self, query: str) -> List[Document]:
        """Retrieve documents using sparse cosine similarity."""

        query_tokens = self._tokenize(query)
        query_tf = Counter(query_tokens)
        query_vector = self._counter_to_tfidf(query_tf, len(query_tokens))

        scored: List[Tuple[float, int]] = []
        for idx, doc_vector in enumerate(self._doc_tfidf):
            score = self._cosine_similarity(query_vector, doc_vector)
            scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: min(self.top_k, len(scored))]
        return self._to_scored_documents(top)

    def _retrieve_with_embedding(self, query: str) -> List[Document]:
        """Retrieve documents using dense embedding similarity."""

        if self._embedding is None or self._vector_store is None:
            return []

        try:
            query_vector = self._embedding.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding query failed; using sparse fallback: %s", exc)
            return []

        limit = min(self.top_k, len(self._vector_document_lookup)) if self._vector_document_lookup else self.top_k
        hits = self._vector_store.query(
            query_embedding=query_vector,
            top_k=max(1, limit),
        )
        documents: List[Document] = []
        for hit in hits:
            base_document = self._vector_document_lookup.get(hit.document_id)
            if base_document is not None:
                logger.debug("Embedding score doc_id=%s score=%.4f", base_document.id, hit.score)
                documents.append(
                    replace(
                        base_document,
                        score=float(hit.score),
                        source=base_document.source or "vector",
                        metadata={
                            **base_document.metadata,
                            "retriever": "vector_store",
                            "vector_db": type(self._vector_store).__name__,
                        },
                    )
                )
            elif hit.document_text:
                meta = dict(hit.metadata or {})
                doc_id = str(meta.pop("doc_id", hit.document_id))
                source = str(meta.pop("source", "vector"))
                documents.append(
                    Document(
                        id=doc_id,
                        content=hit.document_text,
                        score=float(hit.score),
                        source=source,
                        metadata={
                            **meta,
                            "retriever": "vector_store",
                            "vector_db": type(self._vector_store).__name__,
                        },
                    )
                )
        return documents

    def _to_scored_documents(self, scored_indices: Sequence[Tuple[float, int]]) -> List[Document]:
        """Convert score/index pairs into copied scored Documents."""

        docs: List[Document] = []
        for score, doc_idx in scored_indices:
            base = self._documents[doc_idx]
            docs.append(
                replace(
                    base,
                    score=float(score),
                    source=base.source or "vector",
                    metadata={**base.metadata, "retriever": "in_memory"},
                )
            )
        return docs

    def _counter_to_tfidf(self, term_counter: Counter[str], token_count: int) -> Dict[str, float]:
        """Convert term counts to tf-idf sparse weights."""

        if token_count <= 0:
            return {}
        vector: Dict[str, float] = {}
        doc_total = max(len(self._documents), 1)
        for term, count in term_counter.items():
            tf = count / token_count
            idf = math.log((1 + doc_total) / (1 + self._document_frequency.get(term, 0))) + 1.0
            vector[term] = tf * idf
        return vector

    def _cosine_similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Compute cosine similarity for sparse vectors."""

        if not a or not b:
            return 0.0
        shared_terms = set(a.keys()) & set(b.keys())
        dot = sum(a[t] * b[t] for t in shared_terms)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _build_vector_id(self, document: Document, index: int) -> str:
        """Build a unique vector record id for a document entry."""

        return f"{document.id}__{index}"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric terms."""

        return re.findall(r"[a-zA-Z0-9]+", text.lower())
