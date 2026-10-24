"""Shared schema definitions used across raglib pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a retrievable chunk of information."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""


@dataclass
class QueryResult:
    """Represents retrieval results for a query."""

    query: str
    documents: List[Document]
    rewritten_query: Optional[str] = None
    hop_count: int = 0


@dataclass
class RetrievalContext:
    """Represents accumulated retrieval state across pipeline stages."""

    query: str
    documents: List[Document] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    hop_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    """One timed pipeline stage (retrieve, rerank, generate, ...)."""

    stage: str
    latency_ms: float
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Citation:
    """A source cited by the answer as ``[index]``."""

    index: int
    document_id: str
    source: str = ""
    snippet: str = ""


@dataclass
class GenerationResult:
    """Represents a generated answer with sources, citations, and diagnostics."""

    answer: str
    sources: List[Document]
    reasoning_trace: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    trace_events: List[TraceEvent] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
