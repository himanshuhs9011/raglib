"""Data models for the standalone Graph-Based Context Window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from raglib.schemas import Document


@dataclass
class ConnectedNodeRef:
    """Lightweight signpost to a neighbor node with a preview summary for fast movement."""

    node_id: str
    relation: str  # e.g. "PART_OF" | "HAS_SECTION" | "HAS_CHUNK" | "NEXT_CHUNK" | "PREV_CHUNK" | "SHARED_ENTITY" | "PREV_TURN" | "NEXT_TURN" | "SAME_TOPIC_TURN" | "CITED_CHUNK"
    summary: str
    keywords: List[str] = field(default_factory=list)
    weight: float = 1.0
    target_type: str = "chunk"


@dataclass
class GraphNode:
    """A node in the unified Document + Chat Context Graph."""

    id: str
    node_type: str  # "document" | "section" | "chunk" | "entity" | "chat_turn" | "summary"
    content: str
    summary: str = ""
    connected_nodes: List[ConnectedNodeRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def add_connection(
        self,
        node_id: str,
        relation: str,
        summary: str,
        keywords: Optional[List[str]] = None,
        weight: float = 1.0,
        target_type: str = "chunk",
    ) -> None:
        """Add or update a directional connection to a neighbor node."""

        if node_id == self.id:
            return
        kw_list = list(keywords) if keywords else []
        for existing in self.connected_nodes:
            if existing.node_id == node_id and existing.relation == relation:
                existing.summary = summary
                existing.keywords = kw_list
                existing.weight = max(existing.weight, weight)
                existing.target_type = target_type
                return

        self.connected_nodes.append(
            ConnectedNodeRef(
                node_id=node_id,
                relation=relation,
                summary=summary,
                keywords=kw_list,
                weight=weight,
                target_type=target_type,
            )
        )

    def get_connections(self, relation: Optional[str] = None) -> List[ConnectedNodeRef]:
        """Return connected node references, optionally filtered by relation type."""

        if relation is None:
            return list(self.connected_nodes)
        return [ref for ref in self.connected_nodes if ref.relation == relation]


@dataclass
class GraphContextResult:
    """Result returned when assembling a context window from the graph."""

    query: str
    documents: List[Document]
    selected_nodes: List[GraphNode] = field(default_factory=list)
    chat_nodes: List[GraphNode] = field(default_factory=list)
    traversal_trace: List[str] = field(default_factory=list)
    formatted_context: str = ""
    total_tokens: int = 0
