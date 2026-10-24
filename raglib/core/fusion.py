"""Result merging helpers shared by every multi-retrieval strategy."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence

from raglib.schemas import Document

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def doc_key(doc: Document) -> str:
    """Return a stable merge key: the document id, or a content hash when id is empty."""

    if doc.id:
        return doc.id
    digest = hashlib.sha256(doc.content.lower().strip().encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def merge_max(documents: Iterable[Document]) -> List[Document]:
    """Merge duplicates keeping the highest-scoring copy, sorted by score descending."""

    merged: Dict[str, Document] = {}
    for doc in documents:
        key = doc_key(doc)
        if key not in merged or doc.score > merged[key].score:
            merged[key] = doc
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    list_names: Optional[Sequence[str]] = None,
) -> List[Document]:
    """Fuse ranked lists with Reciprocal Rank Fusion: score = sum(w / (k + rank)).

    RRF is rank-based, so it is robust to incomparable score scales (cosine vs BM25 vs web).
    """

    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused: Dict[str, float] = {}
    representative: Dict[str, Document] = {}
    contributors: Dict[str, List[str]] = {}

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if weights is not None else 1.0
        name = list_names[list_index] if list_names is not None else str(list_index)
        seen_in_list = set()
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
            representative.setdefault(key, doc)
            contributors.setdefault(key, []).append(name)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    results: List[Document] = []
    for key, score in ordered:
        base = representative[key]
        results.append(
            replace(
                base,
                score=round(score, 6),
                metadata={**base.metadata, "fusion": "rrf", "fused_from": contributors[key]},
            )
        )
    return results
