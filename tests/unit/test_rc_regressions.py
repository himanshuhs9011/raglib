"""Regressions found in the review of the fix commits, fixed before 2.3.0rc1 (offline, no network)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

from raglib import RAG, UserContext
from raglib.cache import AnswerCache
from raglib.core.runtime import isolated
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.graphrag.schemas import Community, Entity, Relation
from raglib.graphrag.store.base import BaseGraphStore
from raglib.graphrag.store.networkx_store import NetworkXGraphStore
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document

LONG = " ".join(f"word{i}." if i % 9 == 8 else f"word{i}" for i in range(1500))


class Tokenized(BaseEmbedding):
    """A fake model: 1.3 tokens per word, at most ``limit`` tokens; records what it embedded."""

    def __init__(self, limit: int) -> None:
        self.limit, self._model_name = limit, "all-MiniLM-L6-v2"
        self.seen: List[str] = []

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.limit

    def count_input_tokens(self, text: str) -> Optional[int]:
        return math.ceil(1.3 * len(text.split()))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.seen.extend(texts)
        return [[float(len(t) % 7) + 1.0, float(len(t.split()) % 5), 1.0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0, 1.0, 1.0]

    @property
    def dimension(self) -> int:
        return 3


# ------------------------------------------------------------- 1. graph stores written for 2.2


class Store22(BaseGraphStore):
    """A custom store as written against raglib 2.2: exactly the 2.2 abstract methods, nothing more."""

    def __init__(self) -> None:
        self._inner = NetworkXGraphStore()

    def upsert_entities(self, entities: Sequence[Entity]) -> List[str]:
        return self._inner.upsert_entities(entities)

    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        self._inner.upsert_relations(relations)

    def remove_chunks(self, chunk_ids: Iterable[str]) -> List[str]:
        return self._inner.remove_chunks(chunk_ids)

    def set_communities(self, communities: Sequence[Community]) -> None:
        self._inner.set_communities(communities)

    def get_entities(self, ids: Iterable[str]) -> List[Entity]:
        return self._inner.get_entities(ids)

    def all_entities(self) -> List[Entity]:
        return self._inner.all_entities()

    def all_relations(self) -> List[Relation]:
        return self._inner.all_relations()

    def neighbors(self, entity_ids: Iterable[str], hops: int = 1, max_nodes: int = 50) -> Dict[str, int]:
        return self._inner.neighbors(entity_ids, hops, max_nodes)

    def relations_for(self, entity_ids: Iterable[str], limit: int = 50) -> List[Relation]:
        return self._inner.relations_for(entity_ids, limit)

    def communities(self, level: Optional[int] = None) -> List[Community]:
        return self._inner.communities(level)

    def degree(self, entity_id: str) -> int:
        return self._inner.degree(entity_id)


def test_a_2_2_style_graph_store_still_works() -> None:
    store = Store22()  # TypeError before the fix: remove_chunks_detailed was abstract
    rag = RAG(chat_llm="mock", embedding_llm="mock", rag_type="graph", graph_store=store, token_counter="units",
              llm_cache=False, graph_kwargs={"min_llm_report_size": 99})
    rag.add(Document(id="a", content="The Orion Program trains pilots at the Northfield Base."))
    rag.add(Document(id="b", content="The Vega Project builds satellites at the Southfield Lab."))
    assert store.all_entities()
    assert rag.query("What does the Orion Program do?").answer
    assert rag.delete("b") > 0
    assert rag.query("What does the Orion Program do?").answer


def test_set_communities_is_still_required() -> None:
    assert "set_communities" in BaseGraphStore.__abstractmethods__  # as in 2.2
    assert "remove_chunks_detailed" not in BaseGraphStore.__abstractmethods__  # optional, new in 2.3


# ------------------------------------------------- 2. switching oversized_chunks re-chunks on re-add


def test_switching_to_split_and_readding_rechunks(tmp_path: Path) -> None:
    document = Document(id="doc", content=LONG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = RAG(chat_llm="mock", embedding_llm=Tokenized(limit=64), token_counter="units", chunk_size=200,
                   chunk_overlap=0, oversized_chunks="warn")
        kept.add(document)
        assert any(1.3 * len(chunk.content.split()) > 64 for chunk in kept.retriever.documents)  # kept whole
        kept.save(str(tmp_path / "index"))

        model = Tokenized(limit=64)
        split = RAG(chat_llm="mock", embedding_llm=model, token_counter="units", chunk_size=200, chunk_overlap=0,
                    oversized_chunks="split")
        split.load(str(tmp_path / "index"))
        split.add(document)  # before the fix: skipped as unchanged, the oversized chunks stayed
    assert all(model.count_input_tokens(chunk.content) <= 64 for chunk in split.retriever.documents)


def test_readding_with_the_same_policy_is_still_skipped() -> None:
    model = Tokenized(limit=512)
    rag = RAG(chat_llm="mock", embedding_llm=model, token_counter="units", chunk_size=200, chunk_overlap=0)
    rag.add(Document(id="doc", content=LONG))
    embedded = len(model.seen)
    rag.add(Document(id="doc", content=LONG))
    assert len(model.seen) == embedded


# ------------------------------------- 3. production preset: the suggested fix works


def test_production_preset_error_suggests_a_fix_that_works() -> None:
    from raglib import EmbeddingLimitError

    with pytest.raises(EmbeddingLimitError, match="oversized_chunks='split'"):
        RAG(preset="production", chat_llm="mock", embedding_llm=Tokenized(limit=256), reranker="none", chunk_size=500)
    model = Tokenized(limit=256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rag = RAG(preset="production", chat_llm="mock", embedding_llm=model, reranker="none", chunk_size=500,
                  oversized_chunks="split")  # raised before the fix, although the error suggested it
    assert rag._splitter._chunk_size == 500  # your chunk_size is kept; chunks over the limit are split
    rag.add(Document(id="doc", content=LONG))
    assert rag.ingest_usage()["oversized_chunks_split"] > 0
    assert all(model.count_input_tokens(text) <= 256 for text in model.seen)


# -------------------------------------- 4. descriptions survive deletes they do not depend on


def test_a_description_is_kept_while_a_remaining_chunk_supports_it() -> None:
    store = NetworkXGraphStore()
    store.upsert_entities([Entity(id="orion", name="Orion", description="Trains pilots.", source_chunk_ids=["c1"])])
    store.upsert_entities([Entity(id="orion", name="Orion", description="", source_chunk_ids=["c2"])])  # no text
    store.upsert_entities([Entity(id="orion", name="Orion", description="Has a budget.", source_chunk_ids=["c3"])])
    store.remove_chunks_detailed(["c3"])
    assert store.get_entity("orion").description == "Trains pilots."  # was "" before: c2 has no fragment
    store.remove_chunks_detailed(["c1"])
    assert store.get_entity("orion").description == ""  # no remaining chunk supports any text


def test_graphs_without_fragments_still_drop_the_description() -> None:
    store = NetworkXGraphStore()
    store.upsert_entities([Entity(id="orion", name="Orion", description="Budget 42 million.", source_chunk_ids=["c1", "c2"])])
    store.get_entity("orion").descriptions = {}  # loaded from a graph saved before fragments existed
    store.remove_chunks_detailed(["c2"])
    assert store.get_entity("orion").description == ""


# ------------------------------------------- 5. a document added during the run: not cached


def test_answer_cache_put_refuses_an_answer_from_an_older_generation() -> None:
    cache = AnswerCache()
    result = type("R", (), {"sources": [Document(id="x", content="y", metadata={"parent_id": "p"})]})()
    before = cache.generation
    cache.new_documents_added()
    assert cache.put("q", result, generation=before) is False
    assert cache.put("q", result, generation=cache.generation) is True


def test_an_answer_generated_while_a_document_was_added_is_not_cached() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", answer_cache=True, token_counter="units", llm_cache=False)
    rag.add(Document(id="old", content="The office closes at 5pm on Fridays."))
    question = "When does the office close on Fridays?"
    pipeline = rag._rag
    original = pipeline.run

    def run_and_add(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)  # retrieval already happened without the new document
        rag.add(Document(id="new", content="Update: the office closes at 9pm on Fridays."))
        return result

    pipeline.run = run_and_add
    rag.query(question)
    pipeline.run = original
    second = rag.query(question)
    assert second.metadata.get("answer_cache_hit") is not True  # before the fix: the stale answer was served
    assert rag.query(question).metadata.get("answer_cache_hit") is True  # normal caching still works


# ------------------------------------- 6. internal providers: filtered at retrieval, by their tags


class Intranet(BaseSearchProvider):
    internal = True

    @property
    def name(self) -> str:
        return "intranet"

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        return [
            Document(id="i1", content="Canteen menu for the week.", metadata={"access_level": "public"}),
            Document(id="i2", content="Finance forecast: 42 million.",
                     metadata={"access_level": "confidential", "department": "finance"}),
            Document(id="i3", content="Untagged intranet page about salaries."),
        ]


def test_internal_provider_results_are_filtered_at_retrieval() -> None:
    rag = RAG(rag_type="web", chat_llm="mock", embedding_llm="mock", web_search_provider="local",
              access_control=True, token_counter="units")
    rag._web_retriever.provider = Intranet()
    web = rag.pipeline.web_retriever

    def ids(user: Optional[UserContext]) -> List[str]:
        return [doc.id for doc in web.retrieve("menu forecast salaries", filters=rag.permission_filter(user))]

    assert ids(None) == ["i1"]  # before the fix all three came back, untagged and confidential included
    assert ids(UserContext(user_id="cfo", departments=["finance"], max_access_level="confidential")) == ["i1", "i2"]
    assert ids(UserContext(user_id="hr", departments=["hr"], max_access_level="confidential")) == ["i1"]
    result = rag.query("Show the finance forecast", user=UserContext(user_id="visitor"))
    assert all("42 million" not in doc.content for doc in result.sources)


# -------------------------------------------------- 7. role and department order do not matter


def test_memory_key_and_filter_do_not_depend_on_role_order() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", access_control=True, token_counter="units")
    one = UserContext(user_id="u", roles=["analyst", "admin"], departments=["hr", "finance"])
    two = UserContext(user_id="u", roles=["admin", "analyst", "admin"], departments=["finance", "hr"])
    assert rag._scope(None, one, "s") == rag._scope(None, two, "s")
    assert rag.permission_filter(one) == rag.permission_filter(two)


# ----------------------------------------------------- 8. the stream wrapper passes send and throw


def test_isolated_passes_send_and_throw_to_the_wrapped_generator() -> None:
    def inner():
        received = yield "first"
        try:
            yield f"got {received}"
        except ValueError:
            yield "caught"
        return "done"

    stream = isolated(inner())
    assert next(stream) == "first"
    assert stream.send("x") == "got x"  # before the fix: "got None"
    assert stream.throw(ValueError("stop")) == "caught"  # before the fix: raised out of the wrapper
    with pytest.raises(StopIteration) as stop:
        next(stream)
    assert stop.value.value == "done"


def test_isolated_close_still_closes_the_wrapped_generator() -> None:
    closed = []

    def inner():
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = isolated(inner())
    next(stream)
    stream.close()
    assert closed == [True]
