"""Access-control leak fixes: deleted graph text, "non-public" with departments/roles, memory after a
demotion, and empty $or in every store (offline, no network)."""

from __future__ import annotations

import json

import pytest

from raglib import RAG, UserContext, access_tags
from raglib.access import merge_filters
from raglib.schemas import Document

PUBLIC = [
    "The Orion Program trains new pilots at the Northfield Base every spring.",
    "The Orion Program publishes its flight schedule for the Northfield Base online.",
]
CONFIDENTIAL = "The Orion Program budget for the Northfield Base is 42 million, approved by the Finance Board."
SECRET = "42 million"
PUBLIC_USER = UserContext(user_id="visitor")
CLEARED = UserContext(user_id="cfo", roles=["executive"], departments=["finance"], max_access_level="confidential")
QUESTION = "What does the Orion Program do at the Northfield Base and what is its budget?"


def _graph(access: bool = True) -> RAG:
    options = dict(access_control=True) if access else {}
    rag = RAG(chat_llm="mock", embedding_llm="mock", rag_type="graph", token_counter="units", llm_cache=False,
              graph_kwargs={"min_llm_report_size": 99}, **options)
    for text in PUBLIC:
        rag.add(text, access=access_tags("public")) if access else rag.add(text)
    if access:
        rag.add(CONFIDENTIAL, access=access_tags("confidential", department="finance", allowed_roles=["executive"]))
    else:
        rag.add(CONFIDENTIAL)
    rag.rebuild_graph_communities()
    return rag


def _everything_in_the_graph(rag: RAG) -> str:
    store = rag._graph_indexer.store
    parts = [json.dumps([e.to_dict() for e in store.all_entities()], ensure_ascii=False),
             json.dumps([r.to_dict() for r in store.all_relations()], ensure_ascii=False),
             json.dumps([c.to_dict() for c in store.communities()], ensure_ascii=False, default=str)]
    vectors = rag._graph_indexer.entity_vectors
    if vectors is not None:
        parts.append(" ".join(record.document_text for record in vectors.records()))
    return "\n".join(parts)


@pytest.mark.parametrize("access", [True, False])
def test_deleting_a_document_removes_its_text_from_the_whole_graph(access: bool) -> None:
    rag = _graph(access)
    assert SECRET in _everything_in_the_graph(rag)  # not vacuous: the text was in descriptions and reports
    rag.delete(CONFIDENTIAL)
    assert SECRET not in _everything_in_the_graph(rag)
    users = [PUBLIC_USER, CLEARED] if access else [None]
    for user in users:
        filters = rag.permission_filter(user) if access else None
        for mode in ("local", "global"):
            docs = rag._graph_retriever.retrieve(QUESTION, mode=mode, filters=filters)
            assert all(SECRET not in doc.content for doc in docs), (user, mode)
        result = rag.query(QUESTION, user=user) if access else rag.query(QUESTION)
        assert SECRET not in result.answer and all(SECRET not in d.content for d in result.sources)
    # what remains is still described by the public chunks
    orion = next(e for e in rag._graph_indexer.store.all_entities() if e.name == "Orion Program")
    assert orion.description and "pilots" in orion.description + " ".join(orion.descriptions.values())


def test_graphs_saved_without_fragments_drop_descriptions_instead_of_keeping_them() -> None:
    rag = _graph(access=False)
    store = rag._graph_indexer.store
    for entity in store.all_entities():
        entity.descriptions = {}  # as loaded from a graph saved before this fix
    for relation in store.all_relations():
        relation.descriptions = {}
    rag.delete(CONFIDENTIAL)
    assert SECRET not in _everything_in_the_graph(rag)


def test_non_public_means_any_restriction() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units")

    def private(**tags) -> bool:
        return rag._private_document(Document(id="d", content="x", metadata=tags))

    assert private(access_level="confidential")
    assert private(access_level="public", department="hr")
    assert private(access_level="public", allowed_roles=["hr-admin"])
    assert not private(access_level="public")
    assert not private()  # a web result has no tags
    plain = RAG(chat_llm="mock", embedding_llm="mock", token_counter="units")
    assert not plain._private_document(Document(id="d", content="x", metadata={"department": "hr"}))


def test_agent_web_search_is_off_after_a_department_restricted_document() -> None:
    import re

    from raglib.llm.mock_client import MockLLMClient
    from raglib.providers.base_provider import BaseSearchProvider

    class Agent(MockLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            if "research agent" in prompt.lower():
                steps = len(re.findall(r"Step \d+:", prompt))
                plan = [("search_docs", "payroll calendar"), ("search_web", "payroll calendar"), ("finish", "")]
                action, value = plan[min(steps, 2)]
                return json.dumps({"thought": "next", "action": action, "input": value})
            return super().complete(prompt, system)

    class Web(BaseSearchProvider):
        queries = []

        @property
        def name(self) -> str:
            return "web"

        def search(self, query: str, num_results: int = 5):
            Web.queries.append(query)
            return []

    rag = RAG(rag_type="agentic", chat_llm=Agent(), embedding_llm="mock", web_search_provider="local",
              access_control=True, token_counter="units", llm_cache=False)
    rag._web_retriever.provider = Web()
    rag.add("The HR payroll calendar: salaries are paid on the 25th.", access=access_tags("public", department="hr"))
    result = rag.query("When is the payroll calendar?", user=UserContext(user_id="h", departments=["hr"]))
    assert Web.queries == []  # retrieved an HR-only document: no web search
    assert any(call.get("blocked") for call in result.metadata["tool_calls"] if call["tool"] == "search_web")


def test_memory_does_not_survive_a_demotion() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, memory=True, token_counter="units",
              llm_cache=False)
    rag.add("Board memo: the merger budget is 42 million.", access=access_tags("confidential"))
    rag.add("The office opens at eight.", access=access_tags("public"))
    manager = UserContext(user_id="sam", max_access_level="confidential")
    first = rag.query("What is the merger budget?", user=manager, session_id="s1")
    assert SECRET in first.answer  # the mock answer quotes the confidential source
    demoted = UserContext(user_id="sam", max_access_level="public")
    _, session = rag._scope(None, demoted, "s1")
    assert SECRET not in rag._conversations.get(session).get_context()  # a fresh history
    again = rag.query("What did you just tell me?", user=demoted, session_id="s1")
    assert SECRET not in again.answer
    _, same = rag._scope(None, manager, "s1")
    assert SECRET in rag._conversations.get(same).get_context()  # unchanged access: history kept


# ------------------------------------------------------------------ empty $or


def _empty_or_filter(rag: RAG) -> dict:
    return merge_filters(rag.permission_filter(UserContext(user_id="u", max_access_level="executive")), {"$or": []})


def test_empty_or_matches_nothing_in_memory_and_bm25() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units")
    rag.add("The office opens at eight.", access=access_tags("public"))
    flt = _empty_or_filter(rag)
    assert rag.retriever.retrieve("office opens", filters=flt) == []
    sparse = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units", retrieval_mode="sparse")
    sparse.add("The office opens at eight.", access=access_tags("public"))
    assert sparse.retriever.retrieve("office opens", filters=_empty_or_filter(sparse)) == []


def test_empty_or_matches_nothing_in_pgvector_sql() -> None:
    from raglib.vectorstores.pgvector_store import filter_to_sql

    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units")
    sql, _ = filter_to_sql(_empty_or_filter(rag))
    assert "FALSE" in sql
    assert filter_to_sql({"$and": []})[0] == "TRUE"  # an empty AND still matches everything


@pytest.mark.parametrize("backend", ["qdrant", "chroma"])
def test_empty_or_matches_nothing_in_qdrant_and_chroma(backend: str, tmp_path) -> None:
    pytest.importorskip("qdrant_client" if backend == "qdrant" else "chromadb")
    kwargs = {"location": ":memory:"} if backend == "qdrant" else {"persist_directory": str(tmp_path / "c")}
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units",
              vector_db=backend, vector_db_kwargs=kwargs, retrieval_mode="dense")
    rag.add("The office opens at eight.", access=access_tags("public"))
    assert rag.retriever.retrieve("office opens", filters=rag.permission_filter(None))  # sanity: visible
    assert rag.retriever.retrieve("office opens", filters=_empty_or_filter(rag)) == []
