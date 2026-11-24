"""Security fix 2.2.1: every retrieval path enforces filters= (no network).

Each test fails on 2.2.0, where the path returned documents outside the filter.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from raglib import RAG
from raglib.components.retriever import Retriever
from raglib.components.router_retriever import RouterRetriever
from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.tracing import Tracer
from raglib.schemas import Document

PUBLIC = Document(id="pub", content="The Orion launch window opens in March for the whole team.", metadata={"tenant": "acme"})
PRIVATE = Document(id="priv", content="The Orion launch budget is 42 million, board only.", metadata={"tenant": "globex"})
QUESTION = "What about the Orion launch window and budget?"
ACME = {"tenant": "acme"}


def _leaks(documents: List[Document]) -> bool:
    return any("42 million" in doc.content for doc in documents)


def _ctx(filters: Any = ACME) -> PipelineContext:
    return PipelineContext(query=QUESTION, search_query=QUESTION, session_id=None, filters=filters, tracer=Tracer())


class _LeakyGraph:
    """A graph retriever that ignores filters, like 2.2.0's."""

    def retrieve(self, query: str, **_: Any) -> List[Document]:
        return [PRIVATE]


# --- web search with the corpus-backed "local" provider --------------------------------------

def test_web_rag_with_the_local_provider_applies_filters() -> None:
    rag = RAG([PUBLIC, PRIVATE], rag_type="web", chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    result = rag.query(QUESTION, filters=ACME)
    assert not _leaks(result.sources)
    assert any("March" in doc.content for doc in result.sources)


def test_web_retriever_passes_filters_to_the_local_provider() -> None:
    rag = RAG([PUBLIC, PRIVATE], chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    web = rag.pipeline.web_retriever
    assert _leaks(web.retrieve(QUESTION))  # unfiltered: unchanged
    assert not _leaks(web.retrieve(QUESTION, filters=ACME))


def test_local_provider_filters_before_taking_the_top_results() -> None:
    """The match is not pushed out of the top results by documents the filter removes."""

    others = [Document(id=f"o{i}", content="Orion launch budget board only figures", metadata={"tenant": "globex"}) for i in range(6)]
    rag = RAG([*others, PUBLIC], chat_llm="mock", embedding_llm="mock", web_search_provider="local", top_k=2)
    docs = rag.pipeline.web_retriever.retrieve(QUESTION, filters=ACME)
    assert [doc.id.split("_chunk_")[0] for doc in docs] == ["pub"]


def test_agent_web_tool_applies_filters() -> None:
    rag = RAG([PUBLIC, PRIVATE], rag_type="agentic", chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    tools = rag.pipeline.available_tools(_ctx())
    assert not _leaks(tools["search_web"].run(QUESTION))
    assert not _leaks(tools["search_docs"].run(QUESTION))


def test_corrective_rag_web_search_applies_filters() -> None:
    rag = RAG([PUBLIC, PRIVATE], rag_type="corrective", chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    assert not _leaks(rag.pipeline._knowledge_search(_ctx(), failed_docs=[]))


# --- decision-engine fallback and HybridRAG without a corpus retriever ------------------------

class _Custom(BaseRAG):
    pipeline_name = "custom"


@pytest.mark.parametrize("mode", ["web", "hybrid"])
def test_decision_engine_web_and_hybrid_modes_apply_filters(mode: str) -> None:
    rag = RAG([PUBLIC, PRIVATE], chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    pipeline = rag.pipeline
    custom = _Custom(retriever=pipeline.retriever, web_retriever=pipeline.web_retriever, hybrid_retriever=pipeline.hybrid_retriever)
    assert not _leaks(custom._retrieve_by_mode(_ctx(), QUESTION, mode))


def test_decision_engine_web_fallback_without_a_corpus_applies_filters() -> None:
    rag = RAG([PUBLIC, PRIVATE], chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    custom = _Custom(retriever=None, web_retriever=rag.pipeline.web_retriever)
    assert not _leaks(custom._retrieve_by_mode(_ctx(), QUESTION, "vector"))


def test_hybrid_rag_without_a_corpus_retriever_applies_filters() -> None:
    from raglib.rag_types.hybrid_rag import HybridRAG

    rag = RAG([PUBLIC, PRIVATE], chat_llm="mock", embedding_llm="mock", web_search_provider="local")
    hybrid = HybridRAG(retriever=None, hybrid_retriever=rag.pipeline.hybrid_retriever)
    assert not _leaks(hybrid.retrieve(_ctx()))


# --- graph: the router falls back to vector, the agent drops the tool, GraphRAG raises -----------

def test_router_graph_route_with_filters_uses_the_filtered_vector_route() -> None:
    rag = RAG([PUBLIC, PRIVATE], chat_llm="mock", embedding_llm="mock")
    router = RouterRetriever(vector_retriever=rag.pipeline.retriever, graph_retriever=_LeakyGraph())
    assert not _leaks(router.retrieve_route("graph", QUESTION, filters=ACME))
    assert _leaks(router.retrieve_route("graph", QUESTION))  # unfiltered: still the graph


def test_agent_graph_tool_is_not_offered_with_filters() -> None:
    rag = RAG([PUBLIC, PRIVATE], rag_type="agentic", chat_llm="mock", embedding_llm="mock")
    rag.pipeline.graph_retriever = _LeakyGraph()
    assert "graph_search" not in rag.pipeline.available_tools(_ctx())
    assert "graph_search" in rag.pipeline.available_tools(_ctx(filters=None))


def test_graph_rag_raises_instead_of_ignoring_filters() -> None:
    from raglib import FilterNotSupportedError

    rag = RAG([PUBLIC, PRIVATE], rag_type="graph", chat_llm="mock", embedding_llm="mock")
    with pytest.raises(FilterNotSupportedError):
        rag.query(QUESTION, filters=ACME)
    with pytest.raises(FilterNotSupportedError):
        rag.pipeline.graph_retriever.retrieve(QUESTION, filters=ACME)
    rag.pipeline.graph_retriever.retrieve(QUESTION)  # unfiltered: unchanged


# --- neighbour expansion ------------------------------------------------------------------

def test_neighbor_expansion_skips_neighbours_that_fail_the_filter() -> None:
    chunks = [
        Document(id="c1", content="Orion page one: launch window in March.", metadata={"page": 1, "next_id": "c2"}),
        Document(id="c2", content="Orion page two: budget is 42 million.", metadata={"page": 2, "prev_id": "c1"}),
    ]
    retriever = Retriever(documents=chunks, neighbor_window=1)
    unfiltered = retriever.retrieve("Orion launch window March", top_k=1)
    assert "42 million" in unfiltered[0].content  # expansion itself still works
    filtered = retriever.retrieve("Orion launch window March", top_k=1, filters={"page": 1})
    assert [doc.id for doc in filtered] == ["c1"]
    assert not _leaks(filtered)


# --- stores: an empty $or matches nothing, as in the in-memory store -------------------------

def test_pgvector_empty_or_matches_nothing() -> None:
    from raglib.vectorstores.base_store import matches_filter
    from raglib.vectorstores.pgvector_store import filter_to_sql

    assert matches_filter({"tenant": "acme"}, {"$or": []}) is False
    sql, params = filter_to_sql({"$or": []})
    assert sql == "FALSE" and params == []
    sql, _ = filter_to_sql({"$and": []})
    assert sql == "TRUE"  # an empty AND still matches everything


def test_qdrant_empty_or_matches_nothing() -> None:
    models = pytest.importorskip("qdrant_client.models")
    from raglib.vectorstores.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._models = models
    translated = store.to_qdrant_filter({"$or": []})
    assert translated.must == [models.HasIdCondition(has_id=[])]
