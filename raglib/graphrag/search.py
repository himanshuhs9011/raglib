"""GraphRAG query engines: local search (entity neighbourhoods) and global search (communities)."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Callable, Dict, List, Optional, Sequence

from raglib.core.concurrency import parallel_map
from raglib.core.intents import is_global_question
from raglib.core.text import token_set
from raglib.graphrag.indexer import GraphIndexer
from raglib.graphrag.schemas import Community
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

ChunkLookup = Callable[[str], Optional[Document]]


class LocalSearch:
    """Entity-centric retrieval. Makes NO LLM calls: 1 query embedding + graph lookups.

    1. Find seed entities similar to the query (entity vector index).
    2. Expand ``hops`` steps through the graph (strongest edges first).
    3. Rank source chunks by how many relevant entities they mention (seed > neighbour).
    4. Return those chunks plus compact entity, relationship and community-report context.
    """

    def __init__(
        self,
        indexer: GraphIndexer,
        chunk_lookup: ChunkLookup,
        top_k_entities: int = 8,
        hops: int = 1,
        max_neighbors: int = 30,
        max_relations: int = 15,
        max_reports: int = 2,
    ):
        """Configure local search."""

        self.indexer = indexer
        self.chunk_lookup = chunk_lookup
        self.top_k_entities = top_k_entities
        self.hops = hops
        self.max_neighbors = max_neighbors
        self.max_relations = max_relations
        self.max_reports = max_reports

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Return graph context documents followed by the best-supported chunks."""

        store = self.indexer.store
        seeds = self.indexer.entity_search(query, top_k=self.top_k_entities)
        if not seeds:
            return []
        seed_scores = {entity.id: score for entity, score in seeds}
        distances = store.neighbors(seed_scores.keys(), hops=self.hops, max_nodes=self.max_neighbors)
        entities = {entity.id: entity for entity in store.get_entities(distances.keys())}

        weights: Dict[str, float] = {}
        for entity_id, distance in distances.items():
            base = seed_scores.get(entity_id, 0.0) if distance == 0 else 0.0
            weights[entity_id] = max(base, 0.1) / (1 + distance)

        chunk_scores: Dict[str, float] = {}
        for entity_id, weight in weights.items():
            entity = entities.get(entity_id)
            if entity is None:
                continue
            for chunk_id in entity.source_chunk_ids:
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight

        documents: List[Document] = []
        ranked_entities = sorted(entities.values(), key=lambda e: weights.get(e.id, 0.0), reverse=True)[:12]
        if ranked_entities:
            documents.append(
                Document(
                    id=f"graph-entities-{_digest(query)}",
                    content="Entities:\n" + "\n".join(f"- {e.name} ({e.type}): {e.description[:250]}" for e in ranked_entities),
                    metadata={"title": "Knowledge graph entities", "graph": "entities"},
                    source="graph",
                    score=1.0,
                )
            )
        relations = store.relations_for(list(seed_scores.keys()), limit=self.max_relations)
        if relations:
            names = {e.id: e.name for e in entities.values()}
            documents.append(
                Document(
                    id=f"graph-relations-{_digest(query)}",
                    content="Relationships:\n"
                    + "\n".join(
                        f"- {names.get(r.source, r.source)} --{r.type}--> {names.get(r.target, r.target)}: {r.description[:200]}"
                        for r in relations
                    ),
                    metadata={"title": "Knowledge graph relationships", "graph": "relations"},
                    source="graph",
                    score=0.9,
                )
            )
        documents.extend(self._community_documents(ranked_entities))

        chunk_docs: List[Document] = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True):
            chunk = self.chunk_lookup(chunk_id)
            if chunk is None:
                continue
            chunk_docs.append(
                Document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={**chunk.metadata, "graph": "chunk", "graph_score": round(score, 4)},
                    source=chunk.source,
                    score=round(score, 4),
                )
            )
            if len(chunk_docs) >= top_k:
                break
        return chunk_docs + documents

    def _community_documents(self, entities: Sequence) -> List[Document]:
        """Reports of the communities the top entities belong to (if already built)."""

        store = self.indexer.store
        counts: Dict[str, int] = {}
        for entity in entities:
            for community_id in entity.community_ids.values():
                counts[community_id] = counts.get(community_id, 0) + 1
        documents: List[Document] = []
        for community_id, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            community = store.get_community(community_id)
            if community is None or not community.summary:
                continue
            documents.append(
                Document(
                    id=f"community-{community.id}",
                    content=f"Community report - {community.report_text()}",
                    metadata={"title": community.title, "graph": "community", "level": community.level},
                    source="graph",
                    score=0.8,
                )
            )
            if len(documents) >= self.max_reports:
                break
        return documents


class GlobalSearch:
    """Corpus-wide sensemaking via map-reduce over community reports.

    Speed: instead of mapping over every community (Microsoft GraphRAG's default),
    reports are pre-filtered by embedding similarity to the query (``top_communities``)
    and mapped in parallel; the reduce step is the regular cited generation.
    """

    def __init__(
        self,
        indexer: GraphIndexer,
        llm_client: BaseLLMClient,
        level: int = 0,
        top_communities: int = 20,
        max_points: int = 15,
        max_workers: int = 8,
    ):
        """Configure global search."""

        self.indexer = indexer
        self.llm_client = llm_client
        self.level = level
        self.top_communities = top_communities
        self.max_points = max_points
        self.max_workers = max(1, max_workers)
        self.last_mapped = 0

    def search(self, query: str) -> List[Document]:
        """Map relevant community reports to scored key points."""

        self.indexer.ensure_communities()
        communities = [c for c in self.indexer.store.communities(self.level) if c.summary]
        if not communities:
            communities = [c for c in self.indexer.store.communities() if c.summary]
        if not communities:
            return []
        selected = self._prefilter(query, communities)
        self.last_mapped = len(selected)
        mapped = parallel_map(lambda c: self._map(query, c), selected, self.max_workers)

        points: List[Document] = [doc for docs in mapped for doc in docs]
        points.sort(key=lambda doc: doc.score, reverse=True)
        return points[: self.max_points]

    def _prefilter(self, query: str, communities: List[Community]) -> List[Community]:
        """Keep the communities most relevant to the query."""

        if len(communities) <= self.top_communities:
            return communities
        embedding = self.indexer.embedding
        if embedding is not None and all(c.embedding for c in communities):
            query_vector = embedding.embed_query(query)
            scored = [(_cosine(query_vector, c.embedding or []), c) for c in communities]
        else:
            terms = token_set(query, remove_stopwords=True)
            scored = [
                (len(terms & token_set(c.report_text())) + c.rating / 10.0, c) for c in communities
            ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [community for _, community in scored[: self.top_communities]]

    def _map(self, query: str, community: Community) -> List[Document]:
        """Extract query-relevant key points from one community report."""

        prompt = (
            "Using only the community report below, list the key points that help answer the question, "
            "each with an importance score from 0 (irrelevant) to 100 (essential). Return an empty list if "
            "the report is irrelevant.\n"
            f"Question: {query}\n\n"
            f"Community report:\n{community.report_text()}\n\n"
            'Return JSON: {"points": [{"description": "...", "score": 0-100}]}'
        )
        data = self.llm_client.complete_json(prompt, default={})
        points = data.get("points") if isinstance(data, dict) else None
        documents: List[Document] = []
        for index, point in enumerate(points or []):
            if not isinstance(point, dict):
                continue
            text = str(point.get("description") or "").strip()
            try:
                score = float(point.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            if not text or score <= 0:
                continue
            documents.append(
                Document(
                    id=f"{community.id}_p{index}",
                    content=text,
                    metadata={"title": community.title or community.id, "graph": "global_point", "community_id": community.id},
                    source="graph",
                    score=score,
                )
            )
        return documents


def _digest(text: str) -> str:
    """Short stable digest for synthetic document ids."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity."""

    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(length))
    norm = math.sqrt(sum(v * v for v in left[:length])) * math.sqrt(sum(v * v for v in right[:length]))
    return dot / norm if norm else 0.0


class GraphRetriever:
    """Single retrieval entry point for the graph (used by GraphRAG, routing and agent tools)."""

    def __init__(self, local: LocalSearch, global_search: GlobalSearch, default_mode: str = "auto", top_k: int = 5):
        """Configure the retriever."""

        if default_mode not in {"auto", "local", "global"}:
            raise ValueError("graph search mode must be 'auto', 'local' or 'global'")
        self.local = local
        self.global_search = global_search
        self.default_mode = default_mode
        self.top_k = top_k

    @property
    def indexer(self) -> GraphIndexer:
        """Return the underlying indexer."""

        return self.local.indexer

    def resolve_mode(self, query: str, mode: Optional[str] = None) -> str:
        """Pick local vs global for a query."""

        chosen = mode or self.default_mode
        if chosen == "auto":
            return "global" if is_global_question(query) else "local"
        return chosen

    def retrieve(self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None, **_: object) -> List[Document]:
        """Retrieve graph context for a query."""

        if self.resolve_mode(query, mode) == "global":
            return self.global_search.search(query)
        return self.local.search(query, top_k=top_k or self.top_k)
