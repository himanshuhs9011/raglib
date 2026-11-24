"""Answer-cache freshness and deletion: versions captured at retrieval, new documents, GraphRAG summaries,
every cache cleared on delete (offline, no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from raglib import RAG
from raglib.cache import AnswerCache, answer_dependencies
from raglib.latency import benchmark
from raglib.llm.base_client import llm_cache_of
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document, GenerationResult

OLD = Document(id="hours", content="The Acme office closes at 5pm on weekdays.")
NEW = Document(id="hours", content="The Acme office closes at 9pm on weekdays.")
QUESTION = "When does the Acme office close?"


def _cached(**kwargs) -> RAG:
    kwargs.setdefault("token_counter", "units")
    return RAG(chat_llm=kwargs.pop("chat_llm", "mock"), embedding_llm="mock", answer_cache=True, **kwargs)


def test_an_update_during_generation_never_leaves_a_stale_answer() -> None:
    class UpdatesWhileAnswering(MockLLMClient):
        rag = None
        done = False

        def complete(self, prompt: str, system: str = "") -> str:
            answer = super().complete(prompt, system)
            if not self.done and "answer with citations" in prompt:
                self.done = True
                self.rag.add(NEW)  # the document changes while this (old) answer is being generated
            return answer

    llm = UpdatesWhileAnswering()
    rag = _cached(chat_llm=llm, llm_cache=False)
    llm.rag = rag
    rag.add(OLD)
    first = rag.query(QUESTION)
    assert "5pm" in first.answer  # built from the old text
    second = rag.query(QUESTION)
    assert second.metadata.get("answer_cache_hit") is not True and "9pm" in second.answer


def test_a_new_document_refreshes_cached_answers_unless_disabled() -> None:
    rag = _cached()
    rag.add(Document(id="policy-2024", content="Travel policy 2024: economy class for all flights."))
    rag.query("What does the travel policy say about flights?")
    assert rag.query("What does the travel policy say about flights?").metadata["answer_cache_hit"] is True
    report = rag.on_document_update(source=[Document(id="policy-2025", content="Travel policy 2025: business class "
                                                     "allowed on flights over six hours.")])
    assert report["reindexed_chunks"] > 0
    assert rag.query("What does the travel policy say about flights?").metadata.get("answer_cache_hit") is not True
    keep = RAG(chat_llm="mock", embedding_llm="mock", token_counter="units",
               answer_cache=AnswerCache(refresh_on_new_documents=False))
    keep.add(Document(id="a", content="Travel policy 2024: economy class for all flights."))
    keep.query("What does the travel policy say?")
    keep.add(Document(id="b", content="The cafeteria opens at eight."))
    assert keep.query("What does the travel policy say?").metadata["answer_cache_hit"] is True
    keep.add(Document(id="a", content="Travel policy 2024: economy class for all flights."))  # unchanged re-add
    assert keep.query("What does the travel policy say?").metadata["answer_cache_hit"] is True


def test_answers_built_from_graph_summaries_depend_on_the_documents_behind_them() -> None:
    summary = Document(id="community-x", content="Community report about the Orion program.",
                       metadata={"graph": "community", "supplementary": True,
                                 "source_document_versions": {"orion-a": "v1", "orion-b": "v7"}})
    chunk = Document(id="c1", content="...", metadata={"parent_id": "orion-a", "document_version": "v1"})
    assert answer_dependencies([summary, chunk]) == {"orion-a": "v1", "orion-b": "v7"}
    current = {"orion-a": "v1", "orion-b": "v7"}
    cache = AnswerCache()
    cache.version_of = current.get
    cache.put("q", GenerationResult(answer="a", sources=[summary]))
    assert cache.get("q") is not None
    current["orion-b"] = "v8"  # a document behind the summary changed
    assert cache.get("q") is None
    cache.put("q", GenerationResult(answer="a", sources=[summary]))
    assert cache.invalidate(["orion-b"]) == 1  # and invalidating it drops the answer


def test_graph_summaries_record_source_versions_without_access_control() -> None:
    rag = RAG(chat_llm="mock", embedding_llm="mock", rag_type="graph", token_counter="units", llm_cache=False,
              graph_kwargs={"min_llm_report_size": 99})
    rag.add(["The Orion Program trains pilots at the Northfield Base.",
             "The Orion Program publishes schedules for the Northfield Base."])
    rag.rebuild_graph_communities()
    docs = rag._graph_retriever.retrieve("What does the Orion Program do at the Northfield Base?", mode="local")
    summaries = [doc for doc in docs if doc.metadata.get("graph") != "chunk"]
    assert summaries and all(doc.metadata.get("source_document_versions") for doc in summaries)
    assert all("access_level" not in doc.metadata for doc in summaries)  # tags only with access control


def test_delete_and_invalidate_all_clear_every_cache() -> None:
    rag = RAG(chat_llm="mock", cheap_llm="mock", embedding_llm="mock", answer_cache=True, token_counter="units")
    rag.add(OLD)
    rag.query(QUESTION)  # cheap tier: the cheap model's cache gets the answer prompt (with the document text)
    rag.query("Please explain in detail and compare when the Acme office closes and why that matters?")
    cheap = llm_cache_of(rag._cheap_llm)
    assert len(rag._answer_cache) and (len(cheap._cache) or len(llm_cache_of(rag._llm)._cache))
    rag.delete(OLD)
    assert len(rag._answer_cache) == 0 and len(cheap._cache) == 0 and len(llm_cache_of(rag._llm)._cache) == 0
    rag.add(OLD)
    rag.query(QUESTION)
    rag.invalidate_all()
    assert len(rag._answer_cache) == 0 and len(cheap._cache) == 0


def test_cold_benchmarks_are_really_cold() -> None:
    rag = _cached()
    rag.add(OLD)
    stats = benchmark(rag, [QUESTION], repeats=3, warmup=0, warm_cache=False)
    assert "answer_cache" not in stats.stages() and "generate" in stats.stages()


def test_bench_latency_needs_yes_for_paid_models() -> None:
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("bench_latency", root / "benchmarks" / "latency" / "bench_latency.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_latency"] = module
    spec.loader.exec_module(module)
    assert module.main(["--llm", "openai/gpt-4.1-mini", "--repeats", "1"]) == 2
    assert module.main(["--repeats", "1", "--warmup", "0"]) == 0  # offline: no confirmation needed
