"""Standalone Graph-Based Context Window engine for documents and multi-turn chat."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from raglib.graph_context.schemas import ConnectedNodeRef, GraphContextResult, GraphNode
from raglib.llm.base_client import BaseLLMClient
from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.schemas import Document

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "Can", "can",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "into", "is", "it", "its", "me", "more", "my", "no", "not",
    "of", "on", "or", "our", "out", "so", "some", "such", "Tell", "tell",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}

_NEXT_CUES = {"next", "after", "following", "subsequent", "later", "then", "continue", "second", "step"}
_PREV_CUES = {"previous", "before", "prior", "earlier", "preceding", "first", "preceded"}
_SECTION_CUES = {"section", "overview", "summary", "chapter", "heading", "structure", "whole", "entire"}
_FOLLOWUP_CUES = {"it", "its", "that", "this", "those", "they", "them", "earlier", "previous", "before", "again", "back"}


class GraphContextWindow:
    """Builds and navigates a unified Hierarchical Document Graph + Chat Turn Graph.

    Nodes store connected node IDs and short neighbor summaries (`ConnectedNodeRef`)
    so multi-hop movement and context assembly happen via fast lookahead without
    scanning full neighbor text.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        chunking_strategy: str = "llm",
        chunk_size: int = 180,
        chunk_overlap: int = 25,
        max_context_tokens: int = 1200,
        include_neighbor_signposts: bool = True,
    ):
        """Initialize the standalone graph context engine."""

        self._llm_client = llm_client
        self._chunking_strategy = chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))
        self._max_context_tokens = max_context_tokens
        self._include_neighbor_signposts = include_neighbor_signposts

        effective_strategy = chunking_strategy
        if effective_strategy in ("llm", "semantic", "llm_semantic") and llm_client is None:
            effective_strategy = "recursive"

        self._splitter = ChunkSplitter(
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
            min_tokens=10,
            strategy=effective_strategy,
            llm_client=llm_client,
        )

        # Central node store & fast metadata inverted indexes
        self.nodes: Dict[str, GraphNode] = {}
        self.type_index: Dict[str, List[str]] = defaultdict(list)
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.doc_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # 1. Document Graph Ingestion (Document -> Section -> Chunk)
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> List[GraphNode]:
        """Parse documents into hierarchical Document, Section, and Chunk nodes and wire links."""

        created_nodes: List[GraphNode] = []
        new_chunk_nodes: List[GraphNode] = []

        for doc in documents:
            doc_id = doc.id or f"doc_{len(self.type_index['document']) + 1}"
            doc_title = (
                doc.metadata.get("title")
                or doc.metadata.get("source")
                or doc.source
                or doc_id
            )

            sections = self._extract_sections(doc.content, default_title=str(doc_title))
            section_nodes: List[GraphNode] = []
            doc_chunk_nodes: List[GraphNode] = []

            doc_node = GraphNode(
                id=doc_id,
                node_type="document",
                content=doc.content.strip(),
                summary="",
                metadata={
                    **doc.metadata,
                    "title": str(doc_title),
                    "source": doc.source or str(doc_title),
                    "token_count": self._count_tokens(doc.content),
                },
            )
            self._register_node(doc_node)
            created_nodes.append(doc_node)

            for sec_idx, (sec_title, sec_body) in enumerate(sections, start=1):
                sec_id = f"{doc_id}_sec_{sec_idx}"
                chunk_specs = self._split_and_summarize_section(
                    section_title=sec_title,
                    section_text=sec_body,
                    doc_id=doc_id,
                    sec_id=sec_id,
                )

                sec_keywords = self._extract_keywords(f"{sec_title} {sec_body}", limit=8)
                sec_entities = self._extract_entities(f"{sec_title} {sec_body}")
                sec_summary = self._build_section_summary(sec_title, chunk_specs, sec_body)

                sec_node = GraphNode(
                    id=sec_id,
                    node_type="section",
                    content=f"{sec_title}\n{sec_body}".strip(),
                    summary=sec_summary,
                    metadata={
                        "doc_id": doc_id,
                        "doc_title": str(doc_title),
                        "section_index": sec_idx,
                        "section_title": sec_title,
                        "keywords": sec_keywords,
                        "entities": sec_entities,
                        "token_count": self._count_tokens(sec_body),
                    },
                )
                self._register_node(sec_node)
                section_nodes.append(sec_node)
                created_nodes.append(sec_node)

                # Connect Section <-> Document
                sec_node.add_connection(
                    node_id=doc_id,
                    relation="PART_OF",
                    summary=f"Document: {doc_title}",
                    keywords=sec_keywords[:5],
                    weight=0.9,
                    target_type="document",
                )
                doc_node.add_connection(
                    node_id=sec_id,
                    relation="HAS_SECTION",
                    summary=sec_summary,
                    keywords=sec_keywords[:5],
                    weight=0.9,
                    target_type="section",
                )

                for chunk_idx_in_sec, spec in enumerate(chunk_specs, start=1):
                    global_chunk_idx = len(doc_chunk_nodes) + 1
                    chunk_id = f"{sec_id}_chunk_{chunk_idx_in_sec}"
                    chunk_text = spec["content"]
                    chunk_summary = spec["summary"]
                    chunk_keywords = spec["keywords"]
                    chunk_entities = spec["entities"]

                    chunk_node = GraphNode(
                        id=chunk_id,
                        node_type="chunk",
                        content=chunk_text,
                        summary=chunk_summary,
                        metadata={
                            **doc.metadata,
                            "doc_id": doc_id,
                            "doc_title": str(doc_title),
                            "section_id": sec_id,
                            "section_title": sec_title,
                            "chunk_index": global_chunk_idx,
                            "section_chunk_index": chunk_idx_in_sec,
                            "keywords": chunk_keywords,
                            "entities": chunk_entities,
                            "token_count": self._count_tokens(chunk_text),
                            "chunking_strategy": self._chunking_strategy,
                            "source": doc.source or str(doc_title),
                        },
                    )
                    self._register_node(chunk_node)
                    doc_chunk_nodes.append(chunk_node)
                    new_chunk_nodes.append(chunk_node)
                    created_nodes.append(chunk_node)
                    self.doc_to_chunks[doc_id].append(chunk_id)

                    # Connect Chunk <-> Section
                    chunk_node.add_connection(
                        node_id=sec_id,
                        relation="PART_OF",
                        summary=sec_summary,
                        keywords=sec_keywords[:5],
                        weight=0.85,
                        target_type="section",
                    )
                    sec_node.add_connection(
                        node_id=chunk_id,
                        relation="HAS_CHUNK",
                        summary=chunk_summary,
                        keywords=chunk_keywords[:5],
                        weight=0.85,
                        target_type="chunk",
                    )

            # Finalize Document summary from its sections
            doc_keywords = self._extract_keywords(doc.content, limit=10)
            doc_entities = self._extract_entities(doc.content)
            doc_summary_parts = [s.summary for s in section_nodes[:3] if s.summary]
            doc_node.summary = (
                f"{doc_title}: " + " | ".join(doc_summary_parts)
                if doc_summary_parts
                else self._summarize_text(doc.content, max_words=18)
            )
            doc_node.metadata["keywords"] = doc_keywords
            doc_node.metadata["entities"] = doc_entities
            self._index_node_terms(doc_node)

            # Update Section -> Document connection summary now that doc_node.summary is ready
            for sec_node in section_nodes:
                sec_node.add_connection(
                    node_id=doc_id,
                    relation="PART_OF",
                    summary=doc_node.summary,
                    keywords=doc_keywords[:5],
                    weight=0.9,
                    target_type="document",
                )

            # Wire PREV_SECTION / NEXT_SECTION with neighbor summaries
            for idx, sec_node in enumerate(section_nodes):
                if idx > 0:
                    prev_sec = section_nodes[idx - 1]
                    sec_node.add_connection(
                        node_id=prev_sec.id,
                        relation="PREV_SECTION",
                        summary=prev_sec.summary,
                        keywords=prev_sec.metadata.get("keywords", [])[:5],
                        weight=0.8,
                        target_type="section",
                    )
                if idx + 1 < len(section_nodes):
                    next_sec = section_nodes[idx + 1]
                    sec_node.add_connection(
                        node_id=next_sec.id,
                        relation="NEXT_SECTION",
                        summary=next_sec.summary,
                        keywords=next_sec.metadata.get("keywords", [])[:5],
                        weight=0.8,
                        target_type="section",
                    )

            # Wire PREV_CHUNK / NEXT_CHUNK across the document with neighbor summaries
            for idx, chunk_node in enumerate(doc_chunk_nodes):
                if idx > 0:
                    prev_chunk = doc_chunk_nodes[idx - 1]
                    chunk_node.add_connection(
                        node_id=prev_chunk.id,
                        relation="PREV_CHUNK",
                        summary=prev_chunk.summary,
                        keywords=prev_chunk.metadata.get("keywords", [])[:5],
                        weight=0.95,
                        target_type="chunk",
                    )
                if idx + 1 < len(doc_chunk_nodes):
                    next_chunk = doc_chunk_nodes[idx + 1]
                    chunk_node.add_connection(
                        node_id=next_chunk.id,
                        relation="NEXT_CHUNK",
                        summary=next_chunk.summary,
                        keywords=next_chunk.metadata.get("keywords", [])[:5],
                        weight=0.95,
                        target_type="chunk",
                    )

        # Wire cross-section and cross-document SHARED_ENTITY connections
        self._wire_shared_entity_links(new_chunk_nodes)
        return created_nodes

    # ------------------------------------------------------------------
    # 2. Chat Context Graph (Conversational Turns + Links to Doc Chunks)
    # ------------------------------------------------------------------

    def add_chat_turn(
        self,
        query: str,
        answer: str,
        cited_documents: Optional[Sequence[Document]] = None,
        turn_id: Optional[str] = None,
    ) -> GraphNode:
        """Add a conversational Q/A turn to the graph and connect it to prior turns and doc chunks."""

        self._turn_counter += 1
        node_id = turn_id or f"turn_{self._turn_counter}"
        combined_text = f"User: {query.strip()}\nAssistant: {answer.strip()}"
        turn_summary = self._build_turn_summary(query, answer)
        keywords = self._extract_keywords(f"{query} {answer}", limit=8)
        entities = self._extract_entities(f"{query} {answer}")

        cited_ids: List[str] = []
        if cited_documents:
            for doc in cited_documents:
                cand_id = doc.metadata.get("graph_node_id") or doc.id
                if cand_id:
                    cited_ids.append(str(cand_id))

        turn_node = GraphNode(
            id=node_id,
            node_type="chat_turn",
            content=combined_text,
            summary=turn_summary,
            metadata={
                "turn_index": self._turn_counter,
                "query": query.strip(),
                "answer": answer.strip(),
                "keywords": keywords,
                "entities": entities,
                "cited_node_ids": cited_ids,
                "token_count": self._count_tokens(combined_text),
            },
        )

        existing_turn_ids = list(self.type_index.get("chat_turn", []))
        self._register_node(turn_node)

        # 1) Temporal link to immediate previous turn (PREV_TURN <-> NEXT_TURN)
        if existing_turn_ids:
            prev_turn = self.nodes.get(existing_turn_ids[-1])
            if prev_turn is not None:
                turn_node.add_connection(
                    node_id=prev_turn.id,
                    relation="PREV_TURN",
                    summary=prev_turn.summary,
                    keywords=prev_turn.metadata.get("keywords", [])[:5],
                    weight=0.85,
                    target_type="chat_turn",
                )
                prev_turn.add_connection(
                    node_id=turn_node.id,
                    relation="NEXT_TURN",
                    summary=turn_node.summary,
                    keywords=keywords[:5],
                    weight=0.85,
                    target_type="chat_turn",
                )

        # 2) Semantic topic link to older turns sharing entities/keywords (SAME_TOPIC_TURN)
        turn_terms = set(entities) | set(keywords[:6])
        for old_turn_id in existing_turn_ids:
            old_turn = self.nodes.get(old_turn_id)
            if old_turn is None:
                continue
            old_terms = set(old_turn.metadata.get("entities", [])) | set(
                old_turn.metadata.get("keywords", [])[:6]
            )
            shared = turn_terms & old_terms
            if shared:
                weight = min(0.95, 0.65 + 0.1 * len(shared))
                turn_node.add_connection(
                    node_id=old_turn.id,
                    relation="SAME_TOPIC_TURN",
                    summary=old_turn.summary,
                    keywords=sorted(shared)[:5],
                    weight=weight,
                    target_type="chat_turn",
                )
                old_turn.add_connection(
                    node_id=turn_node.id,
                    relation="SAME_TOPIC_TURN",
                    summary=turn_node.summary,
                    keywords=sorted(shared)[:5],
                    weight=weight,
                    target_type="chat_turn",
                )

        # 3) Bridge links between Chat Turn and Cited Document Chunks
        for cited_id in cited_ids:
            chunk_node = self.nodes.get(cited_id)
            if chunk_node is None:
                continue
            turn_node.add_connection(
                node_id=chunk_node.id,
                relation="CITED_CHUNK",
                summary=chunk_node.summary,
                keywords=chunk_node.metadata.get("keywords", [])[:5],
                weight=0.9,
                target_type=chunk_node.node_type,
            )
            chunk_node.add_connection(
                node_id=turn_node.id,
                relation="DISCUSSED_IN_TURN",
                summary=turn_node.summary,
                keywords=keywords[:5],
                weight=0.8,
                target_type="chat_turn",
            )

        return turn_node

    # ------------------------------------------------------------------
    # 3. Graph Navigation & Context Window Assembly
    # ------------------------------------------------------------------

    def build_context_window(
        self,
        query: str,
        top_k_seeds: int = 3,
        max_hops: int = 2,
        max_tokens: Optional[int] = None,
        include_chat_history: bool = True,
    ) -> GraphContextResult:
        """Select seed nodes, traverse connected nodes via neighbor summaries, and assemble context."""

        budget = max_tokens if max_tokens is not None else self._max_context_tokens
        traversal_trace: List[str] = []

        query_keywords = self._extract_keywords(query, limit=12)
        query_entities = self._extract_entities(query)
        query_tokens = set(query_keywords) | set(query_entities)
        raw_lower_words = set(re.findall(r"[a-z0-9_]+", query.lower()))

        wants_next = bool(raw_lower_words & _NEXT_CUES)
        wants_prev = bool(raw_lower_words & _PREV_CUES)
        wants_section = bool(raw_lower_words & _SECTION_CUES)
        is_followup = bool(raw_lower_words & _FOLLOWUP_CUES)

        # Step 1: Score and select seed nodes (chunks + chat turns)
        seed_scores: Dict[str, float] = {}
        for node_id, node in self.nodes.items():
            if node.node_type not in ("chunk", "chat_turn"):
                continue
            score = self._score_node_against_query(node, query_tokens, query.lower())
            if score > 0:
                seed_scores[node_id] = score

        # If conversational follow-up, also seed from the most recent chat turn
        chat_turn_ids = self.type_index.get("chat_turn", [])
        if include_chat_history and chat_turn_ids and (is_followup or not seed_scores):
            latest_turn_id = chat_turn_ids[-1]
            seed_scores[latest_turn_id] = max(seed_scores.get(latest_turn_id, 0.0), 0.55)
            traversal_trace.append(f"Seeded latest chat turn '{latest_turn_id}' for follow-up continuity")

        ranked_seeds = sorted(
            seed_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, top_k_seeds * 2)]

        visited_scores: Dict[str, float] = {}
        frontier: List[Tuple[str, float]] = []
        for node_id, score in ranked_seeds[:top_k_seeds]:
            visited_scores[node_id] = score
            frontier.append((node_id, score))
            node_obj = self.nodes[node_id]
            traversal_trace.append(
                f"Seed {node_obj.node_type} '{node_id}' (score={score:.2f}, summary='{node_obj.summary}')"
            )

        # Step 2: Multi-hop lookahead movement guided by ConnectedNodeRef summaries
        for hop in range(1, max_hops + 1):
            next_frontier: List[Tuple[str, float]] = []
            for current_id, current_score in frontier:
                current_node = self.nodes.get(current_id)
                if current_node is None:
                    continue

                for ref in current_node.connected_nodes:
                    target_node = self.nodes.get(ref.node_id)
                    if target_node is None:
                        continue
                    if not include_chat_history and target_node.node_type == "chat_turn":
                        continue

                    hop_score, reason = self._evaluate_neighbor_lookahead(
                        source_node=current_node,
                        ref=ref,
                        query_tokens=query_tokens,
                        wants_next=wants_next,
                        wants_prev=wants_prev,
                        wants_section=wants_section,
                        is_followup=is_followup,
                    )
                    if hop_score <= 0.0:
                        continue

                    propagated_score = round(current_score * (0.65 ** hop) + hop_score, 4)
                    prev_best = visited_scores.get(ref.node_id, 0.0)
                    if propagated_score > prev_best:
                        visited_scores[ref.node_id] = propagated_score
                        next_frontier.append((ref.node_id, propagated_score))
                        traversal_trace.append(
                            f"Hop {hop}: '{current_id}' --[{ref.relation}]--> '{ref.node_id}' "
                            f"({reason}; neighbor_summary='{ref.summary}')"
                        )

            # Keep top frontier candidates per hop to avoid graph explosion
            next_frontier.sort(key=lambda item: item[1], reverse=True)
            frontier = next_frontier[: top_k_seeds * 2]
            if not frontier:
                break

        # Step 3: Separate selected chat nodes and document/chunk/section nodes, then pack into token budget
        selected_doc_nodes: List[GraphNode] = []
        selected_chat_nodes: List[GraphNode] = []

        ranked_all = sorted(visited_scores.items(), key=lambda item: item[1], reverse=True)
        for node_id, score in ranked_all:
            node = self.nodes[node_id]
            if node.node_type == "chat_turn":
                if include_chat_history:
                    selected_chat_nodes.append(node)
            elif node.node_type in ("chunk", "section"):
                selected_doc_nodes.append(node)

        # Fallback: if no chunk matched lexically, include top chunks from document graph
        if not selected_doc_nodes and self.type_index.get("chunk"):
            for fallback_id in self.type_index["chunk"][:top_k_seeds]:
                selected_doc_nodes.append(self.nodes[fallback_id])
                visited_scores[fallback_id] = 0.1

        # Reserve up to 35% of budget for relevant chat turns (if any), 65%+ for document chunks
        chat_budget = int(budget * 0.35) if selected_chat_nodes else 0
        used_tokens = 0

        kept_chat_nodes: List[GraphNode] = []
        for chat_node in selected_chat_nodes:
            turn_tokens = self._count_tokens(chat_node.content) + 10
            if used_tokens + turn_tokens <= max(chat_budget, 120):
                kept_chat_nodes.append(chat_node)
                used_tokens += turn_tokens

        # Order kept chat turns chronologically for coherent reading
        kept_chat_nodes.sort(key=lambda n: n.metadata.get("turn_index", 0))

        documents: List[Document] = []
        kept_doc_nodes: List[GraphNode] = []
        for doc_node in selected_doc_nodes:
            formatted_chunk = self._format_node_with_structure(doc_node)
            node_tokens = self._count_tokens(formatted_chunk)
            if used_tokens + node_tokens > budget and documents:
                continue
            kept_doc_nodes.append(doc_node)
            used_tokens += node_tokens
            documents.append(
                Document(
                    id=doc_node.id,
                    content=formatted_chunk,
                    metadata={
                        **doc_node.metadata,
                        "graph_node_id": doc_node.id,
                        "node_type": doc_node.node_type,
                        "summary": doc_node.summary,
                        "connected_nodes": [
                            {
                                "node_id": ref.node_id,
                                "relation": ref.relation,
                                "summary": ref.summary,
                            }
                            for ref in doc_node.connected_nodes
                        ],
                    },
                    score=visited_scores.get(doc_node.id, 0.0),
                    source=str(doc_node.metadata.get("source", doc_node.id)),
                )
            )

        formatted_context = self._render_full_context(kept_chat_nodes, documents)
        return GraphContextResult(
            query=query,
            documents=documents,
            selected_nodes=kept_doc_nodes,
            chat_nodes=kept_chat_nodes,
            traversal_trace=traversal_trace,
            formatted_context=formatted_context,
            total_tokens=used_tokens,
        )

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Drop-in retrieval helper returning standard `raglib.schemas.Document` objects."""

        result = self.build_context_window(query=query, top_k_seeds=top_k)
        return result.documents[:top_k]

    # ------------------------------------------------------------------
    # Internal Helpers: Section Extraction, LLM Chunking, Scoring
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str, default_title: str) -> List[Tuple[str, str]]:
        """Extract logical sections from headings or paragraphs."""

        cleaned = text.strip()
        if not cleaned:
            return [(default_title, "")]

        lines = cleaned.splitlines()
        heading_pattern = re.compile(
            r"^(?:#{1,6}\s+(.+)|(?:Section|Chapter|Part|Slide|Page)\s+\d+[:.\-\s]*(.*)|(\d+(?:\.\d+)*\s+[A-Z][^\n]{2,60}))$",
            re.IGNORECASE,
        )

        sections: List[Tuple[str, List[str]]] = []
        current_title = default_title
        current_lines: List[str] = []

        for raw_line in lines:
            stripped = raw_line.strip()
            match = heading_pattern.match(stripped)
            if match:
                if current_lines and "\n".join(current_lines).strip():
                    sections.append((current_title, current_lines))
                extracted = match.group(1) or match.group(2) or match.group(3) or stripped
                current_title = extracted.strip(" #:-") or default_title
                current_lines = []
            else:
                current_lines.append(raw_line)

        if current_lines and "\n".join(current_lines).strip():
            sections.append((current_title, current_lines))

        if not sections:
            return [(default_title, cleaned)]

        return [(title, "\n".join(body_lines).strip()) for title, body_lines in sections]

    def _split_and_summarize_section(
        self,
        section_title: str,
        section_text: str,
        doc_id: str,
        sec_id: str,
    ) -> List[Dict[str, Any]]:
        """Split a section into chunks (using LLM chunking when configured) and attach summaries/keywords."""

        if not section_text.strip():
            return []

        # Try enriched LLM graph chunking if strategy is 'llm' and client is provided
        if self._chunking_strategy in ("llm", "semantic", "llm_semantic") and self._llm_client is not None:
            llm_specs = self._try_llm_structured_chunks(section_title, section_text)
            if llm_specs:
                return llm_specs

        # Otherwise use ChunkSplitter (which itself supports fixed/recursive/llm) + fast summary extraction
        temp_doc = Document(id=sec_id, content=section_text, metadata={"doc_id": doc_id})
        split_docs = self._splitter.split([temp_doc])
        if not split_docs:
            split_docs = [temp_doc]

        specs: List[Dict[str, Any]] = []
        for chunk_doc in split_docs:
            content = chunk_doc.content.strip()
            summary = self._summarize_text(content, prefix=section_title, max_words=16)
            keywords = self._extract_keywords(f"{section_title} {content}", limit=8)
            entities = self._extract_entities(f"{section_title} {content}")
            specs.append(
                {
                    "content": content,
                    "summary": summary,
                    "keywords": keywords,
                    "entities": entities,
                }
            )
        return specs

    def _try_llm_structured_chunks(
        self,
        section_title: str,
        section_text: str,
    ) -> List[Dict[str, Any]]:
        """Use LLM to propose chunk sentence ranges and optional 1-line summaries."""

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", section_text.strip())
            if s and s.strip()
        ]
        if len(sentences) <= 1 or self._llm_client is None:
            return []

        numbered = "\n".join(f"{idx}. {sent}" for idx, sent in enumerate(sentences, start=1))
        prompt = (
            f"Split the numbered sentences from section '{section_title}' into coherent semantic chunks.\n"
            f"Keep each chunk under approximately {self._chunk_size} words.\n"
            "Return lines in the format:\n"
            "CHUNK: start-end | SUMMARY: short 10-word summary | KEYWORDS: kw1, kw2, kw3\n\n"
            f"Sentences:\n{numbered}"
        )
        system = "You are a graph document chunking assistant. Output CHUNK lines."

        try:
            raw = self._llm_client.complete(prompt=prompt, system=system)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM structured graph chunking failed (%s); falling back", exc)
            return []

        line_pattern = re.compile(
            r"CHUNK\s*:\s*(\d+)\s*[-:]\s*(\d+)"
            r"(?:\s*\|\s*SUMMARY\s*:\s*([^|\n]+))?"
            r"(?:\s*\|\s*KEYWORDS\s*:\s*([^\n]+))?",
            re.IGNORECASE,
        )

        parsed_items: List[Tuple[int, int, str, List[str]]] = []
        for match in line_pattern.finditer(raw or ""):
            start = int(match.group(1))
            end = int(match.group(2))
            if start < 1 or end < start or end > len(sentences):
                continue
            summary_part = (match.group(3) or "").strip()
            kw_part = (match.group(4) or "").strip()
            kw_list = [k.strip().lower() for k in kw_part.split(",") if k.strip()] if kw_part else []
            parsed_items.append((start, end, summary_part, kw_list))

        if not parsed_items:
            return []

        # Ensure full sentence coverage
        parsed_items.sort(key=lambda item: item[0])
        covered: List[Tuple[int, int, str, List[str]]] = []
        cursor = 1
        for start, end, summ, kws in parsed_items:
            if start > cursor:
                covered.append((cursor, start - 1, "", []))
            covered.append((max(start, cursor), end, summ, kws))
            cursor = end + 1
        if cursor <= len(sentences):
            covered.append((cursor, len(sentences), "", []))

        specs: List[Dict[str, Any]] = []
        for start, end, llm_summary, llm_keywords in covered:
            chunk_text = " ".join(sentences[start - 1 : end]).strip()
            if not chunk_text:
                continue
            summary = llm_summary or self._summarize_text(chunk_text, prefix=section_title, max_words=16)
            keywords = llm_keywords or self._extract_keywords(f"{section_title} {chunk_text}", limit=8)
            entities = self._extract_entities(f"{section_title} {chunk_text}")
            specs.append(
                {
                    "content": chunk_text,
                    "summary": summary,
                    "keywords": keywords,
                    "entities": entities,
                }
            )
        return specs

    def _wire_shared_entity_links(self, new_chunks: Sequence[GraphNode]) -> None:
        """Connect chunks across sections or documents that share salient entities/keywords."""

        all_chunk_ids = self.type_index.get("chunk", [])
        for chunk in new_chunks:
            chunk_entities = set(chunk.metadata.get("entities", []))
            chunk_keywords = set(chunk.metadata.get("keywords", []))
            salient_terms = chunk_entities | chunk_keywords
            if not salient_terms:
                continue

            candidate_scores: List[Tuple[GraphNode, List[str], float]] = []
            for other_id in all_chunk_ids:
                if other_id == chunk.id:
                    continue
                other = self.nodes.get(other_id)
                if other is None:
                    continue
                # Skip immediate consecutive chunks in the same section (already linked via NEXT/PREV_CHUNK)
                if (
                    other.metadata.get("section_id") == chunk.metadata.get("section_id")
                    and abs(
                        int(other.metadata.get("section_chunk_index", 0))
                        - int(chunk.metadata.get("section_chunk_index", 0))
                    )
                    <= 1
                ):
                    continue

                other_terms = set(other.metadata.get("entities", [])) | set(
                    other.metadata.get("keywords", [])
                )
                shared = sorted(salient_terms & other_terms)
                if not shared:
                    continue

                # Higher weight for cross-document entity bridges
                cross_doc_bonus = 0.15 if other.metadata.get("doc_id") != chunk.metadata.get("doc_id") else 0.0
                weight = min(0.9, 0.55 + 0.1 * len(shared) + cross_doc_bonus)
                candidate_scores.append((other, shared, weight))

            candidate_scores.sort(key=lambda item: item[2], reverse=True)
            for other, shared, weight in candidate_scores[:4]:
                chunk.add_connection(
                    node_id=other.id,
                    relation="SHARED_ENTITY",
                    summary=other.summary,
                    keywords=shared[:5],
                    weight=weight,
                    target_type="chunk",
                )
                other.add_connection(
                    node_id=chunk.id,
                    relation="SHARED_ENTITY",
                    summary=chunk.summary,
                    keywords=shared[:5],
                    weight=weight,
                    target_type="chunk",
                )

    def _evaluate_neighbor_lookahead(
        self,
        source_node: GraphNode,
        ref: ConnectedNodeRef,
        query_tokens: Set[str],
        wants_next: bool,
        wants_prev: bool,
        wants_section: bool,
        is_followup: bool,
    ) -> Tuple[float, str]:
        """Inspect a ConnectedNodeRef's summary, keywords, and relation to decide whether to hop."""

        ref_terms = set(self._extract_keywords(ref.summary, limit=10)) | {
            k.lower() for k in ref.keywords
        }
        overlap = sorted(query_tokens & ref_terms)
        overlap_ratio = len(overlap) / max(1, len(query_tokens)) if query_tokens else 0.0

        # 1) Explicit directional navigation (e.g. "what happens in the next step/chunk?")
        if wants_next and ref.relation in ("NEXT_CHUNK", "NEXT_SECTION", "NEXT_TURN"):
            return 0.55 + overlap_ratio * 0.3, f"directional NEXT cue + {ref.relation}"
        if wants_prev and ref.relation in ("PREV_CHUNK", "PREV_SECTION", "PREV_TURN"):
            return 0.55 + overlap_ratio * 0.3, f"directional PREV cue + {ref.relation}"
        if wants_section and ref.relation in ("PART_OF", "HAS_CHUNK"):
            return 0.45 + overlap_ratio * 0.3, f"hierarchical section cue + {ref.relation}"

        # 2) Chat-to-Document or Chat-to-Earlier-Topic bridge
        if source_node.node_type == "chat_turn":
            if ref.relation == "CITED_CHUNK" and (is_followup or overlap):
                return 0.5 + overlap_ratio * 0.35, "chat turn cited chunk bridge"
            if ref.relation == "SAME_TOPIC_TURN" and overlap:
                return 0.5 + overlap_ratio * 0.35, f"same-topic chat turn match ({', '.join(overlap[:3])})"

        # 3) Neighbor summary / keyword match (complementary information in neighbor!)
        if overlap:
            source_terms = set(source_node.metadata.get("keywords", [])) | set(
                source_node.metadata.get("entities", [])
            )
            novel_terms = [t for t in overlap if t not in source_terms]
            novel_bonus = 0.2 if novel_terms else 0.0
            score = (0.35 + overlap_ratio * 0.45 + novel_bonus) * ref.weight
            reason = f"neighbor summary matched {overlap[:4]}"
            return score, reason

        # 4) Immediate structural continuity when source chunk is strongly matched and adjacent
        if ref.relation == "NEXT_CHUNK" and is_followup:
            return 0.28 * ref.weight, "adjacent NEXT_CHUNK follow-up lookahead"

        return 0.0, ""

    def _score_node_against_query(
        self,
        node: GraphNode,
        query_tokens: Set[str],
        raw_query_lower: str,
    ) -> float:
        """Compute fast lexical + entity + summary match score for a node."""

        if not query_tokens:
            return 0.0

        node_keywords = set(node.metadata.get("keywords", []))
        node_entities = set(node.metadata.get("entities", []))
        summary_tokens = set(self._extract_keywords(node.summary, limit=12))
        content_tokens = set(self._extract_keywords(node.content, limit=40))

        entity_hits = len(query_tokens & node_entities)
        keyword_hits = len(query_tokens & node_keywords)
        summary_hits = len(query_tokens & summary_tokens)
        content_hits = len(query_tokens & content_tokens)

        total_hits = entity_hits * 2.0 + keyword_hits * 1.5 + summary_hits * 1.2 + content_hits * 0.8
        if total_hits <= 0:
            return 0.0

        normalized = total_hits / (len(query_tokens) * 2.5)
        # Phrase bonus if exact 2-word query phrase appears in content or summary
        if len(raw_query_lower) > 5 and raw_query_lower in node.content.lower():
            normalized += 0.35

        return round(min(1.5, normalized), 4)

    def _format_node_with_structure(self, node: GraphNode) -> str:
        """Format a chunk/section node with hierarchical breadcrumbs and neighbor summary signposts."""

        if not self._include_neighbor_signposts:
            return node.content

        doc_title = node.metadata.get("doc_title") or node.metadata.get("doc_id") or ""
        sec_title = node.metadata.get("section_title") or ""
        header_parts = [p for p in (doc_title, sec_title) if p]
        header = f"[Context Path: {' > '.join(header_parts)}]" if header_parts else f"[Node: {node.id}]"

        signposts: List[str] = []
        for ref in node.connected_nodes:
            if ref.relation in ("PREV_CHUNK", "NEXT_CHUNK", "SHARED_ENTITY"):
                signposts.append(f"{ref.relation}({ref.node_id}): {ref.summary}")

        signpost_line = f"[Connected: {' | '.join(signposts[:3])}]" if signposts else ""
        lines = [header, node.content]
        if signpost_line:
            lines.append(signpost_line)
        return "\n".join(lines)

    def _render_full_context(
        self,
        chat_nodes: Sequence[GraphNode],
        documents: Sequence[Document],
    ) -> str:
        """Render the combined chat graph history and document graph chunks into a prompt string."""

        blocks: List[str] = []
        if chat_nodes:
            chat_lines = ["=== Relevant Conversation Graph History ==="]
            for turn in chat_nodes:
                chat_lines.append(f"({turn.id} | {turn.summary})\n{turn.content}")
            blocks.append("\n\n".join(chat_lines))

        if documents:
            doc_lines = ["=== Retrieved Document Graph Context ==="]
            for doc in documents:
                doc_lines.append(doc.content)
            blocks.append("\n\n".join(doc_lines))

        return "\n\n".join(blocks)

    def _register_node(self, node: GraphNode) -> None:
        """Store node in the graph and update fast lookup indexes."""

        self.nodes[node.id] = node
        if node.id not in self.type_index[node.node_type]:
            self.type_index[node.node_type].append(node.id)
        self._index_node_terms(node)

    def _index_node_terms(self, node: GraphNode) -> None:
        """Index a node's entities and keywords for fast lookup."""

        for entity in node.metadata.get("entities", []):
            self.entity_index[str(entity).lower()].add(node.id)
        for kw in node.metadata.get("keywords", []):
            self.keyword_index[str(kw).lower()].add(node.id)

    @staticmethod
    def _build_section_summary(
        section_title: str,
        chunk_specs: Sequence[Dict[str, Any]],
        section_body: str,
    ) -> str:
        """Create a compact summary for a SectionNode from its title and chunk summaries."""

        if chunk_specs:
            first_summary = chunk_specs[0].get("summary", "")
            if first_summary and section_title.lower() not in first_summary.lower():
                return f"{section_title}: {first_summary}"
            if first_summary:
                return first_summary
        words = section_body.split()[:14]
        preview = " ".join(words)
        return f"{section_title}: {preview}".strip(": ")

    def _build_turn_summary(self, query: str, answer: str) -> str:
        """Build a short 1-line summary of a conversational Q/A turn."""

        q_clean = " ".join(query.strip().split()[:10])
        a_clean = " ".join(answer.strip().split()[:12])
        return f"Q: {q_clean} -> A: {a_clean}"

    @staticmethod
    def _summarize_text(text: str, prefix: str = "", max_words: int = 15) -> str:
        """Create a deterministic, compact 1-line summary of a chunk."""

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        first_sentence = sentences[0].strip() if sentences else text.strip()
        words = first_sentence.split()[:max_words]
        short = " ".join(words)
        if prefix and prefix.lower() not in short.lower():
            return f"{prefix}: {short}"
        return short

    @staticmethod
    def _extract_keywords(text: str, limit: int = 8) -> List[str]:
        """Extract salient lowercase keywords ordered by frequency and position."""

        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        counts: Dict[str, int] = {}
        first_pos: Dict[str, int] = {}
        for idx, tok in enumerate(tokens):
            if tok in _STOPWORDS:
                continue
            counts[tok] = counts.get(tok, 0) + 1
            if tok not in first_pos:
                first_pos[tok] = idx

        ranked = sorted(
            counts.keys(),
            key=lambda tok: (-counts[tok], first_pos[tok]),
        )
        return ranked[:limit]

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract capitalized named entities, CamelCase identifiers, and acronyms."""

        candidates: List[str] = []
        pattern = re.compile(
            r"\b(?:[A-Z][a-zA-Z0-9_]+(?:\s+[A-Z][a-zA-Z0-9_]+)*|[A-Z]{2,}[a-zA-Z0-9_]*)\b"
        )
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            low = raw.lower()
            if low in _STOPWORDS or len(low) < 2:
                continue
            if low not in candidates:
                candidates.append(low)
        return candidates[:12]

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Estimate token count from whitespace words."""

        return len(text.split())
