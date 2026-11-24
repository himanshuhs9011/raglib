"""GraphRAG query engines: local search (entity neighbourhoods) and global search (communities)."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from raglib.access import FilterNotSupportedError
from raglib.core.concurrency import parallel_map
from raglib.core.intents import is_global_question
from raglib.core.prompt_safety import data_rule, delimiter_for, wrap
from raglib.core.text import token_set
from raglib.graphrag.indexer import GraphIndexer
from raglib.graphrag.schemas import Community
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document
from raglib.vectorstores.base_store import MetadataFilter, matches_filter

logger = logging.getLogger(__name__)

ChunkLookup = Callable[[str], Optional[Document]]


class ChunkGate:
    """Decides which graph elements a filtered query may see, from their source chunks.

    Graph elements are merged across documents, so:

    * a chunk is visible when it passes the filter (unknown or deleted chunks never are);
    * an entity or relation is visible when at least one of its source chunks is - its name
      then appears in text the user may read;
    * its merged *description* is shown only when every source chunk is visible, because the
      description may quote a restricted document;
    * a community report (an LLM summary over all its members) is shown only when every
      member entity is fully visible.
    """

    def __init__(self, chunk_lookup: ChunkLookup, filters: MetadataFilter):
        """Check chunks with ``chunk_lookup`` against ``filters`` (results cached per query)."""

        self._lookup = chunk_lookup
        self._filters = filters
        self._cache: Dict[str, bool] = {}

    def allowed(self, chunk_id: str) -> bool:
        """True when the chunk exists and passes the filter."""

        if chunk_id not in self._cache:
            chunk = self._lookup(chunk_id)
            self._cache[chunk_id] = chunk is not None and matches_filter(
                {**chunk.metadata, "source": chunk.source, "doc_id": chunk.id}, self._filters
            )
        return self._cache[chunk_id]

    def any_allowed(self, chunk_ids: Iterable[str]) -> bool:
        """True when at least one chunk is visible."""

        return any(self.allowed(chunk_id) for chunk_id in chunk_ids)

    def all_allowed(self, chunk_ids: Iterable[str]) -> bool:
        """True when there is at least one chunk and every chunk is visible."""

        ids = list(chunk_ids)
        return bool(ids) and all(self.allowed(chunk_id) for chunk_id in ids)

    def community_allowed(self, store, community: Community) -> bool:
        """True when every member entity of the community is fully visible."""

        member_ids = set(community.entity_ids)
        entities = store.get_entities(member_ids)
        if community.source_chunk_ids and not self.all_allowed(community.source_chunk_ids):
            return False  # written from a chunk this user may not see (or that no longer exists)
        return len(entities) == len(member_ids) and all(self.all_allowed(e.source_chunk_ids) for e in entities)


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

    def search(self, query: str, top_k: int = 5, filters: Optional[MetadataFilter] = None) -> List[Document]:
        """Return graph context documents followed by the best-supported chunks.

        With ``filters``, only graph elements backed by chunks that pass the filter are used
        (see ``ChunkGate``), and traversal only crosses relations backed by a visible chunk.
        """

        store = self.indexer.store
        gate = ChunkGate(self.chunk_lookup, filters) if filters else None
        seeds = self.indexer.entity_search(query, top_k=self.top_k_entities * (4 if gate else 1))
        if gate is not None:
            seeds = [(entity, score) for entity, score in seeds if gate.any_allowed(entity.source_chunk_ids)]
            seeds = seeds[: self.top_k_entities]
        if not seeds:
            return []
        seed_scores = {entity.id: score for entity, score in seeds}
        with self.indexer.read_lock():  # the graph may be growing in a background build
            return self._search_graph(query, store, seed_scores, top_k, gate)

    def _filtered_neighbors(self, store, seed_ids: Iterable[str], gate: ChunkGate) -> Dict[str, int]:
        """Breadth-first expansion that only crosses relations backed by a visible chunk."""

        distances: Dict[str, int] = {entity_id: 0 for entity_id in seed_ids}
        frontier = list(distances)
        for hop in range(1, self.hops + 1):
            if not frontier or len(distances) >= self.max_neighbors:
                break
            frontier_set = set(frontier)
            next_frontier: List[str] = []
            for relation in store.relations_for(frontier, limit=self.max_neighbors * 4):
                if not gate.any_allowed(relation.source_chunk_ids):
                    continue
                for left, right in ((relation.source, relation.target), (relation.target, relation.source)):
                    if left in frontier_set and right not in distances and len(distances) < self.max_neighbors:
                        distances[right] = hop
                        next_frontier.append(right)
            frontier = next_frontier
        return distances

    def _search_graph(
        self,
        query: str,
        store,
        seed_scores: Dict[str, float],
        top_k: int,
        gate: Optional[ChunkGate] = None,
    ) -> List[Document]:
        if gate is None:
            distances = store.neighbors(seed_scores.keys(), hops=self.hops, max_nodes=self.max_neighbors)
        else:
            distances = self._filtered_neighbors(store, seed_scores.keys(), gate)
        entities = {entity.id: entity for entity in store.get_entities(distances.keys())}
        if gate is not None:
            entities = {eid: e for eid, e in entities.items() if gate.any_allowed(e.source_chunk_ids)}
            distances = {eid: distance for eid, distance in distances.items() if eid in entities}

        def entity_line(entity) -> str:
            if gate is not None and not gate.all_allowed(entity.source_chunk_ids):
                return f"- {entity.name} ({entity.type})"  # merged description may quote a hidden document
            return f"- {entity.name} ({entity.type}): {entity.description[:250]}"

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
                if gate is not None and not gate.allowed(chunk_id):
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight

        def used_chunks(source_chunk_ids: Sequence[str]) -> List[str]:
            """The chunks whose text a line shows: all of them, or only the visible ones (name only)."""

            if gate is None or gate.all_allowed(source_chunk_ids):
                return list(source_chunk_ids)
            return [chunk_id for chunk_id in source_chunk_ids if gate.allowed(chunk_id)]

        documents: List[Document] = []
        ranked_entities = sorted(entities.values(), key=lambda e: weights.get(e.id, 0.0), reverse=True)[:12]
        if ranked_entities:
            sources = [chunk_id for e in ranked_entities for chunk_id in used_chunks(e.source_chunk_ids)]
            documents.append(
                Document(
                    id=f"graph-entities-{_digest(query)}",
                    content="Entities:\n" + "\n".join(entity_line(e) for e in ranked_entities),
                    metadata={"title": "Knowledge graph entities", "graph": "entities", **inherited_access(self, sources)},
                    source="graph",
                    score=1.0,
                )
            )
        relations = store.relations_for(list(seed_scores.keys()), limit=self.max_relations * (4 if gate else 1))
        if gate is not None:
            relations = [
                r
                for r in relations
                if r.source in entities and r.target in entities and gate.any_allowed(r.source_chunk_ids)
            ][: self.max_relations]
        names = {e.id: e.name for e in entities.values()}

        def relation_line(r) -> str:
            line = f"- {names.get(r.source, r.source)} --{r.type}--> {names.get(r.target, r.target)}"
            if gate is not None and not gate.all_allowed(r.source_chunk_ids):
                return line
            return f"{line}: {r.description[:200]}"

        if relations:
            sources = [chunk_id for r in relations for chunk_id in used_chunks(r.source_chunk_ids)]
            documents.append(
                Document(
                    id=f"graph-relations-{_digest(query)}",
                    content="Relationships:\n" + "\n".join(relation_line(r) for r in relations),
                    metadata={"title": "Knowledge graph relationships", "graph": "relations", **inherited_access(self, sources)},
                    source="graph",
                    score=0.9,
                )
            )
        documents.extend(self._community_documents(ranked_entities, gate))

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

    def _community_documents(self, entities: Sequence, gate: Optional[ChunkGate] = None) -> List[Document]:
        """Reports of the communities the top entities belong to (if already built and visible)."""

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
            if gate is not None and not gate.community_allowed(store, community):
                continue
            documents.append(
                Document(
                    id=f"community-{community.id}",
                    content=f"Community report - {community.report_text()}",
                    metadata={
                        "title": community.title, "graph": "community", "level": community.level,
                        **inherited_access(self, community_chunks(store, community)),
                    },
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
        chunk_lookup: Optional[ChunkLookup] = None,
    ):
        """Configure global search (``chunk_lookup`` is required to serve filtered queries)."""

        self.indexer = indexer
        self.chunk_lookup = chunk_lookup
        self.llm_client = llm_client
        self.level = level
        self.top_communities = top_communities
        self.max_points = max_points
        self.max_workers = max(1, max_workers)
        self.last_mapped = 0

    def search(self, query: str, filters: Optional[MetadataFilter] = None) -> List[Document]:
        """Map relevant community reports to scored key points.

        With ``filters``, only communities whose every member entity is backed solely by
        visible chunks are mapped (a report summarizes all members, so a mixed one could leak).
        """

        gate: Optional[ChunkGate] = None
        if filters:
            if self.chunk_lookup is None:
                raise FilterNotSupportedError("GlobalSearch needs chunk_lookup to apply metadata filters")
            gate = ChunkGate(self.chunk_lookup, filters)
        self.indexer.ensure_communities()
        store = self.indexer.store

        def visible(communities: List[Community]) -> List[Community]:
            usable = [c for c in communities if c.summary]
            return usable if gate is None else [c for c in usable if gate.community_allowed(store, c)]

        with self.indexer.read_lock():
            communities = visible(store.communities(self.level))
            if not communities:
                communities = visible(store.communities())
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

        report = community.report_text()
        delimiter = delimiter_for(query, report)
        prompt = (
            "Using only the community report below, list the key points that help answer the question, "
            "each with an importance score from 0 (irrelevant) to 100 (essential). Return an empty list if "
            "the report is irrelevant.\n"
            f"{data_rule(delimiter)}\n"
            f"Question: {query}\n\n"
            f"Community report:\n{wrap(report, delimiter)}\n\n"
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
                    metadata={
                        "title": community.title or community.id, "graph": "global_point", "community_id": community.id,
                        **inherited_access(self, community_chunks(self.indexer.store, community)),
                    },
                    source="graph",
                    score=score,
                )
            )
        return documents


def community_chunks(store: Any, community: Community) -> List[str]:
    """The chunks a community report was built from (recorded at build time; older stores: members' now)."""

    if community.source_chunk_ids:
        return list(community.source_chunk_ids)
    from raglib.graphrag.communities import member_chunk_ids

    return member_chunk_ids(community, store)


MAX_LISTED_SOURCES = 200


def inherited_access(search: Any, chunk_ids: Sequence[str]) -> Dict[str, Any]:
    """Access tags a summary inherits from its source chunks (``AccessControl.strictest``) plus
    ``source_chunk_ids``; empty when access control is off (``search.access is None``).

    A source chunk that no longer exists counts as untagged, so the summary is denied by default.
    """

    access = getattr(search, "access", None)
    ids = sorted(set(chunk_ids))
    lookup = getattr(search, "chunk_lookup", None)
    metadatas = []
    versions: Dict[str, Any] = {}
    for chunk_id in ids:
        chunk = lookup(chunk_id) if lookup is not None else None
        metadata = dict(chunk.metadata) if chunk is not None else {}
        metadatas.append(metadata)
        if chunk is not None:
            versions[str(metadata.get("parent_id") or chunk.id)] = metadata.get("document_version")
    # What the summary was built from (for cache invalidation), with or without access control.
    info: Dict[str, Any] = {"source_chunk_ids": ids[:MAX_LISTED_SOURCES], "source_chunk_count": len(ids),
                            "source_document_versions": versions}
    if access is not None:
        info.update(access.strictest(metadatas))
    return info


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
    """Single retrieval entry point for the graph (used by GraphRAG, routing and agent tools).

    ``supports_filters`` tells pipelines that ``retrieve(..., filters=...)`` is enforced; a
    graph retriever without it makes filtered queries raise ``FilterNotSupportedError``.
    """

    supports_filters = True

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

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        filters: Optional[MetadataFilter] = None,
        **_: object,
    ) -> List[Document]:
        """Retrieve graph context for a query; ``filters`` restrict it to visible source chunks."""

        if self.resolve_mode(query, mode) == "global":
            return self.global_search.search(query, filters=filters)
        return self.local.search(query, top_k=top_k or self.top_k, filters=filters)


def require_filter_support(retriever: object, filters: Optional[MetadataFilter], label: str = "graph retriever") -> None:
    """Raise ``FilterNotSupportedError`` when ``filters`` are set but ``retriever`` cannot enforce them."""

    if filters and not getattr(retriever, "supports_filters", False):
        raise FilterNotSupportedError(
            f"{type(retriever).__name__} ({label}) cannot apply metadata filters; refusing to return "
            "unfiltered results"
        )
