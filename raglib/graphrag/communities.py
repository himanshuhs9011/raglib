"""Hierarchical community detection and incremental community reports."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.graphrag.schemas import Community
from raglib.graphrag.store.base import BaseGraphStore
from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


def _partition(graph: Any, resolution: float, seed: int) -> List[List[str]]:
    """Partition a graph with Leiden (graspologic) when installed, else Louvain (networkx)."""

    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [[node] for node in graph.nodes()]
    try:
        from graspologic.partition import leiden  # type: ignore[import-not-found]

        mapping = leiden(graph, resolution=resolution, random_seed=seed)
        groups: Dict[int, List[str]] = {}
        for node, label in mapping.items():
            groups.setdefault(label, []).append(node)
        return list(groups.values())
    except ImportError:
        pass
    import networkx as nx  # type: ignore[import-not-found]

    communities = nx.algorithms.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
    return [sorted(community) for community in communities]


def _community_id(level: int, members: Sequence[str]) -> str:
    """Stable id derived from membership, so unchanged communities keep their reports."""

    digest = hashlib.sha256("\x00".join(sorted(members)).encode("utf-8")).hexdigest()[:12]
    return f"c{level}_{digest}"


def detect_communities(
    store: BaseGraphStore,
    max_levels: int = 2,
    max_community_size: int = 25,
    min_community_size: int = 2,
    resolution: float = 1.0,
    seed: int = 42,
) -> List[Community]:
    """Build a community hierarchy: level 0 = coarse clusters, deeper levels split large ones."""

    graph = store.to_networkx()
    entities = {entity.id: entity for entity in store.all_entities()}
    results: List[Community] = []

    def signature(members: Sequence[str]) -> str:
        parts = [f"{member}:{entities[member].text_hash()}" for member in sorted(members) if member in entities]
        edges = graph.subgraph(members).number_of_edges()
        return hashlib.sha256(f"{'|'.join(parts)}#{edges}".encode("utf-8")).hexdigest()[:16]

    frontier: List[tuple] = [(None, list(graph.nodes()))]
    for level in range(max_levels):
        next_frontier: List[tuple] = []
        for parent_id, nodes in frontier:
            subgraph = graph.subgraph(nodes)
            groups = _partition(subgraph, resolution=resolution, seed=seed) if level == 0 or len(nodes) > max_community_size else []
            if level > 0 and (not groups or len(groups) == 1):
                continue  # cannot be split further
            for members in groups:
                if len(members) < min_community_size:
                    continue
                community = Community(
                    id=_community_id(level, members),
                    level=level,
                    entity_ids=sorted(members),
                    parent_id=parent_id,
                    signature=signature(members),
                )
                results.append(community)
                if len(members) > max_community_size:
                    next_frontier.append((community.id, members))
        frontier = next_frontier
        if not frontier:
            break
    logger.info("Detected %d communities across %d levels", len(results), len({c.level for c in results}))
    return results


class CommunityReporter:
    """Writes community reports with the LLM, in parallel, reusing unchanged reports."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding: Optional[BaseEmbedding] = None,
        max_workers: int = 8,
        max_entities_in_prompt: int = 30,
        max_relations_in_prompt: int = 40,
        min_llm_report_size: int = 3,
    ):
        """Configure the reporter.

        Communities with fewer than ``min_llm_report_size`` entities get a template summary built
        from their entity descriptions instead of an LLM call (most communities in real corpora
        are tiny pairs, so this removes the bulk of report cost).
        """

        self.llm_client = llm_client
        self.embedding = embedding
        self.max_workers = max(1, max_workers)
        self.max_entities_in_prompt = max_entities_in_prompt
        self.max_relations_in_prompt = max_relations_in_prompt
        self.min_llm_report_size = max(0, min_llm_report_size)
        self.llm_calls = 0

    def report(
        self,
        communities: List[Community],
        store: BaseGraphStore,
        previous: Optional[Dict[str, Community]] = None,
    ) -> List[Community]:
        """Fill title/summary/findings/rating/embedding; only changed communities hit the LLM."""

        previous = previous or {}
        todo: List[Community] = []
        for community in communities:
            old = previous.get(community.id)
            if old is not None and old.summary and old.report_signature == community.signature:
                community.title, community.summary = old.title, old.summary
                community.findings, community.rating = list(old.findings), old.rating
                community.report_signature, community.embedding = old.report_signature, old.embedding
            else:
                todo.append(community)

        small = [c for c in todo if len(c.entity_ids) < self.min_llm_report_size]
        for community in small:
            self._template_report(community, store)
        todo = [c for c in todo if len(c.entity_ids) >= self.min_llm_report_size]
        if todo:
            logger.info("Generating %d community reports (%d reused)", len(todo), len(communities) - len(todo))
            if self.max_workers == 1 or len(todo) == 1:
                for community in todo:
                    self._report_one(community, store)
            else:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(todo))) as pool:
                    list(pool.map(lambda c: self._report_one(c, store), todo))

        needs_embedding = [c for c in communities if c.summary and c.embedding is None]
        if self.embedding is not None and needs_embedding:
            vectors = self.embedding.embed_documents([c.report_text() for c in needs_embedding])
            for community, vector in zip(needs_embedding, vectors):
                community.embedding = list(vector)
        return communities

    def _template_report(self, community: Community, store: BaseGraphStore) -> None:
        """Summarize a small community from entity descriptions and relations (no LLM call)."""

        entities = store.get_entities(community.entity_ids)
        member_ids = set(community.entity_ids)
        names = {entity.id: entity.name for entity in entities}
        relations = [
            r for r in store.relations_for(list(member_ids), limit=20) if r.source in member_ids and r.target in member_ids
        ]
        community.title = ", ".join(e.name for e in entities[:3])[:200]
        parts = [f"{e.name} ({e.type}): {e.description[:200]}".strip() for e in entities]
        parts += [
            f"{names.get(r.source, r.source)} {r.type.replace('_', ' ').lower()} {names.get(r.target, r.target)}."
            + (f" {r.description[:160]}" if r.description else "")
            for r in relations[:5]
        ]
        community.summary = " ".join(parts)
        community.findings = []
        community.rating = 1.0
        community.report_signature = community.signature
        community.embedding = None

    def _report_one(self, community: Community, store: BaseGraphStore) -> None:
        """Generate one report."""

        entities = sorted(store.get_entities(community.entity_ids), key=lambda e: store.degree(e.id), reverse=True)
        entities = entities[: self.max_entities_in_prompt]
        member_ids = set(community.entity_ids)
        relations = [
            relation
            for relation in store.relations_for([e.id for e in entities], limit=self.max_relations_in_prompt * 3)
            if relation.source in member_ids and relation.target in member_ids
        ][: self.max_relations_in_prompt]
        names = {entity.id: entity.name for entity in store.get_entities(member_ids)}

        entity_lines = "\n".join(f"- {e.name}: ({e.type}) {e.description[:300]}" for e in entities)
        relation_lines = "\n".join(
            f"- {names.get(r.source, r.source)} --{r.type}--> {names.get(r.target, r.target)}: {r.description[:200]}"
            for r in relations
        )
        prompt = (
            "Write a community report for this cluster of related entities from a knowledge graph.\n\n"
            f"Entities:\n{entity_lines}\n\nRelations:\n{relation_lines or '(none)'}\n\n"
            'Return JSON: {"title": "short descriptive title", "summary": "3-5 sentence executive summary of what '
            'this community is about", "findings": ["key insight 1", "key insight 2", "..."], '
            '"rating": <importance from 0 to 10>}'
        )
        self.llm_calls += 1
        data = self.llm_client.complete_json(prompt, system="You are an analyst summarizing knowledge graph communities.", default={})
        if not isinstance(data, dict):
            data = {}
        community.title = str(data.get("title") or ", ".join(e.name for e in entities[:3]))[:200]
        community.summary = str(data.get("summary") or "; ".join(e.description[:120] for e in entities[:5]))
        findings = data.get("findings") or []
        community.findings = [str(item) for item in findings if str(item).strip()][:8] if isinstance(findings, list) else []
        try:
            community.rating = float(data.get("rating", 0.0))
        except (TypeError, ValueError):
            community.rating = 0.0
        community.report_signature = community.signature
        community.embedding = None
