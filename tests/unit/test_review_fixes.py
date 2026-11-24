"""Fixes from the review of the regression commits (offline, no network)."""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from raglib import RAG, FilterNotSupportedError, UserContext
from raglib.cache import AnswerCache
from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.tracing import Tracer
from raglib.providers.base_provider import BaseSearchProvider, ProviderChain
from raglib.schemas import Document

QUESTION = "What about the Orion launch?"


def _ctx(filters: Any = None) -> PipelineContext:
    return PipelineContext(query=QUESTION, search_query=QUESTION, session_id=None, filters=filters, tracer=Tracer())


# ------------------------------------------------------- custom retrievers written for 2.2 (hotfix port)


class _OldWeb:
    def retrieve(self, query: str) -> List[Document]:
        return [Document(id="w", content="Orion public news.", source="https://example.com")]


class _OldHybrid:
    def retrieve(self, query: str) -> List[Document]:
        return [Document(id="h", content="Orion corpus and web.")]


class _Custom(BaseRAG):
    pipeline_name = "custom"


def test_a_2_2_style_web_retriever_is_called_without_filters(caplog: pytest.LogCaptureFixture) -> None:
    from raglib.rag_types.web_rag import WebRAG

    web = WebRAG(web_retriever=_OldWeb())
    with caplog.at_level("WARNING", logger="raglib.components.web_retriever"):
        assert [doc.id for doc in web.retrieve(_ctx({"tenant": "acme"}))] == ["w"]  # TypeError before
    assert any("filters were not applied by web retriever _OldWeb" in r.getMessage() for r in caplog.records)


def test_a_2_2_style_hybrid_retriever_works_unfiltered_and_refuses_filters() -> None:
    from raglib.rag_types.hybrid_rag import HybridRAG

    hybrid = HybridRAG(retriever=None, hybrid_retriever=_OldHybrid())
    assert [doc.id for doc in hybrid.retrieve(_ctx())] == ["h"]  # TypeError (filters=None) before
    assert [doc.id for doc in _Custom(hybrid_retriever=_OldHybrid())._retrieve_by_mode(_ctx(), QUESTION, "hybrid")] == ["h"]
    with pytest.raises(FilterNotSupportedError):
        hybrid.retrieve(_ctx({"tenant": "acme"}))


# ------------------------------------------------------------------ #4 AnswerCache subclasses


def test_an_answer_cache_subclass_without_generation_still_caches() -> None:
    class OldCache(AnswerCache):
        def put(self, question: str, result: Any, scope: Any = None, pipeline: str = "") -> bool:  # 2.3-dev signature
            return super().put(question, result, scope, pipeline)

    rag = RAG(chat_llm="mock", embedding_llm="mock", answer_cache=OldCache(), token_counter="units", llm_cache=False)
    rag.add(Document(id="d", content="The office closes at 5pm on Fridays."))
    rag.query("When does the office close on Fridays?")  # TypeError before
    assert rag.query("When does the office close on Fridays?").metadata["answer_cache_hit"] is True


# ------------------------------------------------------------------ #6 / #7 internal providers


class _Intranet(BaseSearchProvider):
    internal = True

    def __init__(self, docs: Optional[List[Document]] = None) -> None:
        self.docs = docs

    @property
    def name(self) -> str:
        return "intranet"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        if self.docs is not None:
            return list(self.docs)
        return [
            Document(id="i1", content="Canteen menu.", source="wiki", metadata={"access_level": "public"}),
            Document(id="i2", content="Odd page.", source="wiki", metadata={"access_level": "top-secret-typo"}),
            Document(id="i3", content="HR page.", source="hr-portal", metadata={"access_level": "public"}),
        ]


class _Public(BaseSearchProvider):
    @property
    def name(self) -> str:
        return "publicweb"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        return [Document(id="p1", content="Orion launch news.", source="https://example.com/news")]


def _web_rag(provider: BaseSearchProvider) -> RAG:
    rag = RAG(rag_type="web", chat_llm="mock", embedding_llm="mock", web_search_provider="local",
              access_control=True, token_counter="units")
    rag._web_retriever.provider = provider
    return rag


def test_provider_chain_results_are_judged_by_the_provider_that_returned_them() -> None:
    rag = _web_rag(ProviderChain([_Intranet(docs=[]), _Public()]))  # the intranet finds nothing, the web answers
    docs = rag.pipeline.web_retriever.retrieve(QUESTION, filters=rag.permission_filter(None))
    assert [doc.id for doc in docs] == ["p1"]  # was [] before: treated as an internal, untagged result
    assert docs[0].metadata["source_type"] == "web" and docs[0].metadata["access_level"] == "public"
    assert "_raglib_chain_index" not in docs[0].metadata
    internal = _web_rag(ProviderChain([_Intranet(docs=[Document(id="u", content="Untagged intranet page.")]), _Public()]))
    assert internal.pipeline.web_retriever.retrieve(QUESTION, filters=internal.permission_filter(None)) == []


def test_one_unreadable_internal_result_does_not_hide_the_others() -> None:
    rag = _web_rag(_Intranet())
    docs = rag.pipeline.web_retriever.retrieve(QUESTION, filters=rag.permission_filter(None))
    assert [doc.id for doc in docs] == ["i1", "i3"]  # was [] before: the ValueError hid every result


def test_source_filters_apply_to_internal_results() -> None:
    from raglib.access import merge_filters

    rag = _web_rag(_Intranet())
    filters = merge_filters(rag.permission_filter(UserContext(user_id="u")), {"source": "wiki"})
    assert [doc.id for doc in rag.pipeline.web_retriever.retrieve(QUESTION, filters=filters)] == ["i1"]


# ------------------------------------------------------------------ #9 Qdrant string ranges


def test_qdrant_does_not_index_string_ranges() -> None:
    models = pytest.importorskip("qdrant_client.models")
    from raglib.vectorstores.qdrant_store import _filter_fields

    fields = _filter_fields({"$and": [{"date": {"$gte": "2024-01-01"}}, {"page": {"$gt": 3}}, {"lang": "en"}]}, models)
    assert "date" not in fields  # a keyword index cannot serve a range
    assert fields == {"page": models.PayloadSchemaType.INTEGER, "lang": models.PayloadSchemaType.KEYWORD}
