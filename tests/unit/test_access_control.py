"""Access control and filter enforcement: no retrieval path may leak a restricted document.

Everything runs offline (MockLLMClient + MockEmbedding). The leak checks look at every prompt
sent to the LLM, the answer, the sources and the whole serialized result.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pytest

from raglib import RAG, AccessControl, FilterNotSupportedError, UserContext, access_tags
from raglib.access import merge_filters
from raglib.core.base import PipelineContext
from raglib.core.tracing import Tracer
from raglib.graphrag.search import GlobalSearch
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document
from raglib.vectorstores.base_store import VectorHit, matches_filter
from raglib.vectorstores.chroma_store import ChromaVectorStore
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

SECRET = "ZEPHYR7731"  # only in the confidential document
SECRET_PARTNER = "Globex"  # only in the confidential document; merged into the shared "Acme Corp" entity

PUBLIC_TEXT = (
    "Acme Corp publishes its holiday schedule every December for all staff and visitors. "
    "Acme Corp offices close on Friday afternoons during the summer, and the cafeteria opens at eight."
)
CONFIDENTIAL_TEXT = (
    f"Acme Corp will acquire {SECRET_PARTNER} in Project Zephyr next quarter after the board approves it. "
    f"The Project Zephyr budget code is {SECRET} and only the finance leadership may share it."
)
QUESTION = "What is the budget code of Project Zephyr at Acme Corp?"

LOW_USER = UserContext(user_id="intern", roles=["staff"], departments=["sales"], max_access_level="internal")
HIGH_USER = UserContext(user_id="cfo", roles=["executive"], departments=["finance"], max_access_level="executive")

ALL_RAG_TYPES = [
    "naive", "advanced", "corrective", "self", "agentic", "hybrid", "hybrid_web", "multi_query",
    "multi_hop", "routing", "memory", "web", "tool", "graph",
]


class SpyLLM(MockLLMClient):
    """Mock LLM that records every prompt it receives."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(f"{system}\n{prompt}")
        return super().complete(prompt, system)


def _build(rag_type: str = "advanced", spy: Optional[SpyLLM] = None, **kwargs: Any) -> RAG:
    options: Dict[str, Any] = dict(
        rag_type=rag_type,
        chat_llm=spy or SpyLLM(),
        embedding_llm="mock",
        web_search_provider="local",
        vector_db="memory",
        top_k=3,
        access_control=True,
    )
    if rag_type in {"routing", "agentic"}:
        options["enable_graph"] = True
    options.update(kwargs)
    rag = RAG(**options)
    rag.add(PUBLIC_TEXT, access=access_tags(access_level="public"))
    rag.add(CONFIDENTIAL_TEXT, access=access_tags("confidential", department="finance", allowed_roles=["executive"]))
    if rag.graph is not None:
        # Index-time work (like extraction at ingest) reads every document; it is otherwise done
        # lazily on the first global query. Do it now so the spy only sees query-time prompts.
        rag.rebuild_graph_communities()
    return rag


def _serialized(result: Any) -> str:
    return json.dumps(asdict(result), default=str)


def _assert_no_leak(result: Any, spy: SpyLLM) -> None:
    for marker in (SECRET, SECRET_PARTNER):
        assert marker not in result.answer
        assert all(marker not in doc.content for doc in result.sources)
        assert marker not in _serialized(result)
        assert not any(marker in prompt for prompt in spy.prompts), f"{marker} reached the LLM"


# ---------------------------------------------------------------------------- end-to-end leak test


@pytest.mark.parametrize("retrieval_mode", ["dense", "sparse", "hybrid"])
def test_low_access_user_never_sees_confidential_chunks_in_any_retrieval_mode(retrieval_mode: str) -> None:
    spy = SpyLLM()
    rag = _build("advanced", spy=spy, retrieval_mode=retrieval_mode)
    spy.prompts.clear()
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)
    _assert_no_leak(rag.query(QUESTION), spy)  # anonymous


@pytest.mark.parametrize("rag_type", ALL_RAG_TYPES)
def test_low_access_user_never_sees_confidential_chunks_in_any_rag_type(rag_type: str) -> None:
    spy = SpyLLM()
    rag = _build(rag_type, spy=spy)
    spy.prompts.clear()  # graph extraction at ingest legitimately reads every document
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)
    _assert_no_leak(rag.query("What are the main themes across the whole corpus overall?", user=LOW_USER), spy)


@pytest.mark.parametrize("rag_type", ["advanced", "graph", "hybrid_web", "multi_query"])
def test_cleared_user_does_see_the_confidential_chunk(rag_type: str) -> None:
    """Sanity check: the leak tests above are not vacuous."""

    rag = _build(rag_type)
    result = rag.query(QUESTION, user=HIGH_USER)
    assert any(SECRET in doc.content for doc in result.sources)


@pytest.mark.parametrize("rag_type", ["advanced", "graph", "hybrid_web"])
def test_cached_answer_of_a_cleared_user_is_never_served_to_a_low_access_user(rag_type: str) -> None:
    spy = SpyLLM()
    rag = _build(rag_type, spy=spy)  # llm_cache=True is the default
    high = rag.query(QUESTION, user=HIGH_USER)
    assert SECRET in _serialized(high)
    spy.prompts.clear()
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)
    _assert_no_leak(rag.query(QUESTION), spy)


@pytest.mark.parametrize(
    "options",
    [
        {"answer_cache": True},
        {"answer_cache": True, "cache_personalized": True},
        {"cheap_llm": "spy"},
        {"parent_retrieval": "heading", "chunk_size": 20, "chunk_overlap": 0},
        {"chunking_strategy": "embedding"},
        {"reranker": "llm", "rerank_skip_threshold": -1.0},
    ],
    ids=["answer-cache", "answer-cache-personalized", "cheap-tier", "parent-sections", "semantic-chunks", "rerank-skip"],
)
def test_caches_tiers_and_parent_sections_never_leak(options: Dict[str, Any]) -> None:
    spy = SpyLLM()
    if options.get("cheap_llm") == "spy":
        options = {**options, "cheap_llm": spy}
    rag = _build("advanced", spy=spy, **options)
    rag.query(QUESTION, user=HIGH_USER)  # a cleared user fills every cache first
    rag.query(QUESTION)
    spy.prompts.clear()
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)
    _assert_no_leak(rag.query(QUESTION), spy)


def test_streaming_applies_the_permission_filter() -> None:
    spy = SpyLLM()
    rag = _build("advanced", spy=spy)
    spy.prompts.clear()
    stream = rag.stream(QUESTION, user=LOW_USER)
    text = "".join(stream)
    assert SECRET not in text and not any(SECRET in prompt for prompt in spy.prompts)


def test_every_router_route_and_agent_tool_applies_filters() -> None:
    rag = _build("agentic")
    acl = rag.permission_filter(LOW_USER)
    router = rag._router_retriever
    assert set(router.available_routes()) == {"vector", "web", "hybrid", "graph"}
    for route in router.available_routes():
        docs = router.retrieve_route(route, QUESTION, filters=acl)
        assert all(SECRET not in doc.content and SECRET_PARTNER not in doc.content for doc in docs), route

    ctx = PipelineContext(query=QUESTION, search_query=QUESTION, session_id=None, filters=acl, tracer=Tracer())
    tools = rag.pipeline.available_tools(ctx)
    assert {"search_docs", "search_web", "graph_search"} <= set(tools)
    for name, tool in tools.items():
        output = str(tool.func(QUESTION))
        assert SECRET not in output and SECRET_PARTNER not in output, name


# ------------------------------------------------------------------------------- deny by default


def test_untagged_documents_are_invisible_to_everyone() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True)
    rag.add("The untagged memo mentions the codename Bluebird for the Orion launch window.")
    top = UserContext(roles=["executive"], departments=["finance"], max_access_level="executive")
    assert rag.query("What is the Orion codename?", user=top).sources == []
    assert rag.query("What is the Orion codename?").sources == []


def test_default_access_level_tags_untagged_documents() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=AccessControl(default_access_level="internal"))
    rag.add("The internal wiki says the Orion launch window opens in March.")
    assert rag.query("When does the Orion launch window open?").sources == []  # anonymous = public only
    assert rag.query("When does the Orion launch window open?", user={"max_access_level": "internal"}).sources


def test_anonymous_queries_see_only_the_lowest_level() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True)
    rag.add("The public FAQ says the Orion launch is open to visitors.", access={"access_level": "public"})
    rag.add("The internal note says the Orion launch moved to the north pad.", access={"access_level": "internal"})
    contents = " ".join(doc.content for doc in rag.query("Where is the Orion launch?").sources)
    assert "visitors" in contents and "north pad" not in contents


def test_departments_and_roles_must_both_match() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True)
    rag.add(
        "The Orion payroll budget sheet lists every salary band for the launch team.",
        access=access_tags("internal", department=["finance", "hr"], allowed_roles="manager"),
    )
    question = "What does the Orion payroll budget sheet list?"
    cases = [
        (UserContext(roles=["manager"], departments=["hr"], max_access_level="internal"), True),
        (UserContext(roles=["manager"], departments=["sales"], max_access_level="internal"), False),
        (UserContext(roles=["staff"], departments=["finance"], max_access_level="internal"), False),
        (UserContext(roles=["manager"], departments=["finance"], max_access_level="public"), False),
    ]
    for user, visible in cases:
        assert bool(rag.query(question, user=user).sources) is visible, user


def test_caller_filters_can_only_narrow_the_permission_filter() -> None:
    rag = _build("advanced")
    widen = {"$or": [{"access_level": "confidential"}, {"access_level": "public"}]}
    result = rag.query(QUESTION, user=LOW_USER, filters=widen)
    assert all(SECRET not in doc.content for doc in result.sources)
    assert merge_filters({"a": 1}, None) == {"a": 1}
    assert merge_filters({"a": 1}, {"b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}


def test_permission_changes_apply_on_reingest_of_unchanged_text() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True)
    text = "The Orion travel policy lets the launch team book economy flights only."
    rag.add(text, access={"access_level": "public"})
    assert rag.query("What does the Orion travel policy allow?").sources
    rag.add(text, access={"access_level": "confidential"})
    assert rag.query("What does the Orion travel policy allow?").sources == []
    assert rag.stats()["chunks"] == 1


def test_access_tags_survive_save_and_load(tmp_path) -> None:
    rag = _build("advanced")
    rag.save(str(tmp_path))
    loaded = RAG(chat_llm="mock", embedding_llm="mock", vector_db="memory", access_control=True).load(str(tmp_path))
    assert all(SECRET not in doc.content for doc in loaded.query(QUESTION, user=LOW_USER).sources)
    assert any(SECRET in doc.content for doc in loaded.query(QUESTION, user=HIGH_USER).sources)


def test_memory_is_namespaced_per_user() -> None:
    rag = _build("memory")
    rag.query(QUESTION, session_id="shared", user=HIGH_USER)
    spy = rag._llm_raw
    spy.prompts.clear()
    result = rag.query("And what was that code again?", session_id="shared", user=LOW_USER)
    _assert_no_leak(result, spy)
    assert result.metadata["session_id"] == "shared"


def test_invalid_access_configuration_is_rejected() -> None:
    plain = RAG(chat_llm="mock", embedding_llm="mock")
    with pytest.raises(ValueError, match="access_control"):
        plain.query("anything", user=LOW_USER)
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True)
    with pytest.raises(ValueError, match="Unknown access_level"):
        rag.add("Some text long enough to be indexed as a chunk of the corpus.", access={"access_level": "secret"})
    with pytest.raises(ValueError, match="Unknown access tags"):
        rag.add("Some text long enough to be indexed as a chunk of the corpus.", access={"level": "public"})
    with pytest.raises(ValueError, match="Unknown access_level"):
        rag.query("anything", user={"max_access_level": "top-secret"})
    with pytest.raises(ValueError, match="Unknown user context keys"):
        rag.query("anything", user={"role": "admin"})


# ------------------------------------------------------------------- backends that cannot filter


class _GraphRetrieverWithoutFilters:
    """A custom graph retriever that does not declare filter support."""

    def __init__(self, indexer: Any) -> None:
        self.indexer = indexer

    def resolve_mode(self, query: str, mode: Optional[str] = None) -> str:
        return "local"

    def retrieve(self, query: str, **_: Any) -> List[Document]:
        return [Document(id="everything", content=CONFIDENTIAL_TEXT, metadata={"graph": "chunk"})]


def test_graph_rag_refuses_filters_it_cannot_enforce() -> None:
    rag = _build("graph")
    rag.pipeline.graph_retriever = _GraphRetrieverWithoutFilters(rag.graph)
    with pytest.raises(FilterNotSupportedError):
        rag.query(QUESTION, user=LOW_USER)


def test_global_search_without_chunk_lookup_refuses_filters() -> None:
    rag = _build("graph")
    search = GlobalSearch(indexer=rag.graph, llm_client=MockLLMClient())
    assert search.search("main themes")  # unfiltered still works
    with pytest.raises(FilterNotSupportedError):
        search.search("main themes", filters={"access_level": "public"})


def test_graph_hides_merged_descriptions_and_mixed_communities() -> None:
    rag = _build("graph")
    acl = rag.permission_filter(LOW_USER)
    local = rag._graph_retriever.retrieve("Tell me about Acme Corp", mode="local", filters=acl)
    text = " ".join(doc.content for doc in local)
    assert "Acme Corp" in text  # the shared entity is visible (it is in the public document)...
    assert SECRET_PARTNER not in text and SECRET not in text  # ...but not its merged description
    global_points = rag._graph_retriever.retrieve("main themes overall", mode="global", filters=acl)
    assert all(SECRET not in doc.content and SECRET_PARTNER not in doc.content for doc in global_points)


class _StoreWithoutFilters(InMemoryVectorStore):
    supports_metadata_filters = False


class _StoreIgnoringFilters(InMemoryVectorStore):
    """A buggy custom store that silently ignores filters."""

    def query(self, query_embedding: List[float], top_k: int, filters: Any = None) -> List[VectorHit]:
        return super().query(query_embedding, top_k, filters=None)


def test_vector_store_that_cannot_filter_raises() -> None:
    rag = _build("advanced", vector_db=_StoreWithoutFilters(), retrieval_mode="dense")
    with pytest.raises(FilterNotSupportedError):
        rag.query(QUESTION, user=LOW_USER)
    plain = RAG(chat_llm="mock", embedding_llm="mock", vector_db=_StoreWithoutFilters(), retrieval_mode="dense")
    plain.add(PUBLIC_TEXT)
    assert plain.query("When do Acme Corp offices close?").sources  # unfiltered queries still work


def test_hits_from_a_store_that_ignores_filters_are_dropped() -> None:
    spy = SpyLLM()
    rag = _build("advanced", spy=spy, vector_db=_StoreIgnoringFilters(), retrieval_mode="dense")
    spy.prompts.clear()
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)


def test_corpus_backed_search_provider_without_filter_support_raises() -> None:
    from raglib.components.web_retriever import WebRetriever
    from raglib.providers.base_provider import BaseSearchProvider

    class CorpusProvider(BaseSearchProvider):
        searches_corpus = True

        @property
        def name(self) -> str:
            return "corpus"

        def search(self, query: str, num_results: int = 5) -> List[Document]:
            return [Document(id="c", content=CONFIDENTIAL_TEXT)]

    retriever = WebRetriever(provider=CorpusProvider())
    assert retriever.retrieve("q")  # no filter: fine
    with pytest.raises(FilterNotSupportedError):
        retriever.retrieve("q", filters={"access_level": "public"})


# ------------------------------------------------------------------------- filter translation


def test_chroma_where_is_normalized_recursively() -> None:
    acl = AccessControl().filter_for(None)  # $or lists with a single item
    where = ChromaVectorStore._to_chroma_where(merge_filters(acl, {"lang": "en", "year": {"$gt": 2020}}))

    def check(node: Dict[str, Any]) -> None:
        assert len(node) == 1, node
        key, value = next(iter(node.items()))
        if key in {"$and", "$or"}:
            assert len(value) >= 2, node
            for item in value:
                check(item)

    check(where)
    assert ChromaVectorStore._to_chroma_where({"a": 1}) == {"a": 1}
    assert ChromaVectorStore._to_chroma_where({"a": 1, "b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}


def test_permission_filter_is_pushed_into_a_real_chroma_store() -> None:
    pytest.importorskip("chromadb")
    spy = SpyLLM()
    rag = _build("advanced", spy=spy, vector_db="chroma", retrieval_mode="dense")
    spy.prompts.clear()
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER), spy)
    _assert_no_leak(rag.query(QUESTION, user=LOW_USER, filters={"chunk_index": 1, "chunk_count": 1}), spy)
    assert any(SECRET in doc.content for doc in rag.query(QUESTION, user=HIGH_USER).sources)


def test_matches_filter_array_contains_is_opt_in() -> None:
    meta = {"tags": ["a", "b"]}
    assert not matches_filter(meta, {"tags": "a"})  # default behaviour unchanged
    assert matches_filter(meta, {"tags": "a"}, array_contains=True)
    assert matches_filter(meta, {"tags": {"$in": ["x", "b"]}}, array_contains=True)
    assert not matches_filter(meta, {"tags": {"$nin": ["b"]}}, array_contains=True)
    assert not matches_filter(meta, {"tags": {"$ne": "a"}}, array_contains=True)


def test_acl_keys_are_scalar_and_backend_safe() -> None:
    tagged = AccessControl().tag(access_tags("confidential", department="R&D / Labs", allowed_roles=["lead.eng"]))
    acl_keys = [key for key in tagged if key.startswith("_acl_")]
    assert acl_keys and all(value is True for key, value in tagged.items() if key.startswith("_acl_"))
    assert all(key.replace("_", "").isalnum() for key in acl_keys)
    assert AccessControl().tag({"title": "untagged"}) == {"title": "untagged"}
