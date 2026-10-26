"""Regression tests for the v2 production fixes (one test per issue found in the v1 review)."""

from __future__ import annotations

import json
from typing import Dict, List

import pytest

from raglib import (
    RAG,
    BM25Index,
    ConversationStore,
    CorrectiveRAG,
    DecisionEngine,
    Document,
    Generator,
    MockLLMClient,
    Reranker,
    Retriever,
    Tool,
)
from raglib.components.grader import RelevanceGrader
from raglib.core.fusion import reciprocal_rank_fusion
from raglib.core.intents import is_global_question, is_greeting
from raglib.core.text import count_tokens, tokenize
from raglib.embedding.mock_embedding import MockEmbedding
from raglib.eval import evaluate, retrieval_metrics
from raglib.llm.base_client import BaseLLMClient
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

CORPUS = [
    "Photosynthesis converts light energy into chemical energy inside plant chloroplasts.",
    "The Calvin cycle fixes carbon dioxide into glucose using ATP and NADPH.",
    "Mitochondria perform cellular respiration and release ATP for the cell.",
    "Invoice INV-2024-0042 was paid by Acme Corp on 3 March.",
]


class CountingLLM(BaseLLMClient):
    """Wraps the mock LLM and records every prompt."""

    def __init__(self) -> None:
        self.inner = MockLLMClient()
        self.prompts: List[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.inner.complete(prompt, system)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self.inner.chat(messages)


@pytest.mark.parametrize("query", ["What is this?", "Which chip is high end", "Is there a canvas vs paper trend?"])
def test_substring_bug_fixed_questions_are_not_greetings(query: str) -> None:
    """'this' must not match 'hi'; such questions must still retrieve."""

    assert not is_greeting(query)
    assert DecisionEngine().retrieval_type(query) != "none"


@pytest.mark.parametrize("query", ["hi", "Hello there", "thanks a lot", "good morning"])
def test_real_greetings_skip_retrieval(query: str) -> None:
    """Short small talk needs no retrieval."""

    assert is_greeting(query)
    assert DecisionEngine().retrieval_type(query) == "none"


def test_global_question_detection() -> None:
    """Corpus-wide questions route to GraphRAG global search."""

    assert is_global_question("What are the main themes across the documents?")
    assert not is_global_question("Who paid invoice INV-2024-0042?")


@pytest.mark.parametrize("mode", ["sparse", "dense", "hybrid"])
def test_non_english_retrieval_works(mode: str) -> None:
    """Hindi queries must retrieve Hindi documents in every retrieval mode."""

    docs = [
        Document(id="hi1", content="प्रकाश संश्लेषण पौधों में होता है और ऊर्जा बनाता है"),
        Document(id="hi2", content="माइटोकॉन्ड्रिया कोशिका को ऊर्जा देते हैं"),
    ]
    embedding = None if mode == "sparse" else MockEmbedding()
    retriever = Retriever(docs, embedding=embedding, top_k=2, mode=mode, vector_db="memory")
    results = retriever.retrieve("प्रकाश संश्लेषण क्या है")
    assert results and results[0].id == "hi1"
    assert tokenize("प्रकाश संश्लेषण") == ["प्रकाश", "संश्लेषण"]


def test_hybrid_finds_exact_identifier_via_bm25() -> None:
    """Exact IDs are where BM25 beats embeddings; hybrid must surface them first."""

    docs = [Document(id=f"d{i}", content=text) for i, text in enumerate(CORPUS)]
    retriever = Retriever(docs, embedding=MockEmbedding(), top_k=2, mode="hybrid")
    assert retriever.retrieve("INV-2024-0042")[0].id == "d3"


def test_hybrid_keeps_semantic_matches_when_keywords_do_not_overlap() -> None:
    """When BM25 has no signal, the dense score must decide (fusion must not add noise)."""

    docs = [
        Document(id="match", content="electrical generators produce electricity"),
        Document(id="other", content="bananas are yellow tropical fruit"),
    ]
    retriever = Retriever(docs, embedding=MockEmbedding(), top_k=2, mode="hybrid")
    assert retriever.retrieve("electricity")[0].id == "match"
    assert retriever.retrieve("electrically")[0].id == "match"  # no BM25 overlap, trigram-similar


@pytest.mark.parametrize("fusion", ["relative", "rrf"])
def test_hybrid_fusion_methods_are_selectable(fusion: str) -> None:
    """Both fusion strategies are available."""

    docs = [Document(id=f"d{i}", content=text) for i, text in enumerate(CORPUS)]
    retriever = Retriever(docs, embedding=MockEmbedding(), top_k=2, mode="hybrid", fusion=fusion)
    assert retriever.retrieve("Calvin cycle glucose")[0].id == "d1"


def test_rrf_fuses_rankings() -> None:
    """A document ranked well in both lists beats one ranked first in a single list."""

    a, b, c = (Document(id=x, content=x) for x in "abc")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [doc.id for doc in fused][0] == "b"
    assert fused[0].metadata["fusion"] == "rrf"


def test_metadata_filters_apply_to_dense_and_sparse() -> None:
    """Filters restrict both retrieval paths."""

    docs = [
        Document(id="a", content="refund policy for premium customers", metadata={"tenant": "acme"}),
        Document(id="b", content="refund policy for basic customers", metadata={"tenant": "globex"}),
    ]
    retriever = Retriever(docs, embedding=MockEmbedding(), top_k=5)
    results = retriever.retrieve("refund policy", filters={"tenant": "globex"})
    assert [doc.id for doc in results] == ["b"]
    results = retriever.retrieve("refund policy", filters={"tenant": {"$in": ["acme"]}}, mode="sparse")
    assert [doc.id for doc in results] == ["a"]


def test_mmr_promotes_diversity() -> None:
    """With MMR, a near-duplicate of the top hit is pushed down."""

    docs = [
        Document(id="x1", content="solar panels convert sunlight to electricity"),
        Document(id="x2", content="solar panels convert sunlight to electricity efficiently"),
        Document(id="y", content="wind turbines convert wind to electricity"),
    ]
    plain = Retriever(docs, embedding=MockEmbedding(), top_k=2, mode="dense")
    diverse = Retriever(docs, embedding=MockEmbedding(), top_k=2, mode="dense", mmr_lambda=0.3)
    assert {d.id for d in plain.retrieve("solar electricity")} == {"x1", "x2"}
    assert "y" in {d.id for d in diverse.retrieve("solar electricity")}


def test_bm25_incremental_remove() -> None:
    """Removing a document removes it from search."""

    index = BM25Index()
    index.add("a", "apples and oranges")
    index.add("b", "oranges and lemons")
    index.remove("b")
    assert [doc_id for doc_id, _ in index.search("oranges", 5)] == ["a"]
    assert len(index) == 1


def test_reranker_none_keeps_first_stage_order() -> None:
    """The default reranker must not destroy dense/RRF ranking (v1 lexical bug)."""

    docs = [Document(id=str(i), content=f"doc {i} " * (10 - i), score=1 - i / 10) for i in range(5)]
    assert [d.id for d in Reranker(top_k=3).rerank("doc", docs)] == ["0", "1", "2"]


def test_reranker_custom_scorer_reorders() -> None:
    """Custom scorers (or cross-encoders) reorder candidates."""

    docs = [Document(id="short", content="ok"), Document(id="long", content="relevant answer text")]
    reranker = Reranker(top_k=2, score_fn=lambda q, texts: [len(t) for t in texts])
    assert reranker.rerank("q", docs)[0].id == "long"


def test_add_is_idempotent_and_updates_changed_documents(tmp_path) -> None:
    """Re-adding unchanged content is a no-op; changed files replace their old chunks."""

    path = tmp_path / "policy.txt"
    path.write_text("Refunds are allowed within 30 days.", encoding="utf-8")
    rag = RAG(str(path), vector_db="memory")
    assert len(rag.retriever.documents) == 1
    assert rag.add(str(path)) == 0
    assert len(rag.retriever.documents) == 1

    path.write_text("Refunds are allowed within 14 days only.", encoding="utf-8")
    assert rag.update(str(path)) == 1
    contents = [doc.content for doc in rag.retriever.documents]
    assert contents == ["Refunds are allowed within 14 days only."]
    assert rag.retriever.vector_store.count() == 1

    assert rag.delete(str(path)) == 1
    assert rag.retriever.documents == []
    assert rag.retriever.vector_store.count() == 0
    assert rag.retriever.retrieve("refunds") == []


def test_delete_works_after_file_was_removed(tmp_path) -> None:
    """Deleting by path must work even when the file no longer exists."""

    path = tmp_path / "old.txt"
    path.write_text("Legacy pricing was 10 dollars per seat.", encoding="utf-8")
    rag = RAG(str(path))
    path.unlink()
    assert rag.delete(str(path)) == 1
    assert rag.retriever.documents == []


def test_save_and_load_roundtrip_reuses_embeddings(tmp_path) -> None:
    """A saved index reloads without re-embedding."""

    rag = RAG(CORPUS, vector_db="memory")
    rag.save(str(tmp_path / "index"))

    class NoEmbed(MockEmbedding):
        def embed_documents(self, texts):  # noqa: D401
            raise AssertionError("documents must not be re-embedded on load")

    restored = RAG(embedding_llm=NoEmbed(), vector_db="memory").load(str(tmp_path / "index"))
    assert len(restored.retriever.documents) == len(rag.retriever.documents)
    assert restored.query("What does photosynthesis convert?").sources


def test_instances_do_not_share_chroma_collections() -> None:
    """Two RAG objects in one process must never see each other's documents."""

    pytest.importorskip("chromadb")
    alpha = RAG(["Alpha doc about apples.", "Alpha two about orchards."], vector_db="chroma")
    beta = RAG(["Beta doc about rockets.", "Beta two about engines.", "Beta three about fuel."], vector_db="chroma")
    contents = " ".join(doc.content for doc in alpha.retriever.retrieve("rockets engines fuel", top_k=5))
    assert "Beta" not in contents
    assert beta.retriever.vector_store.count() == 3


def test_no_silent_mock_without_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a model and without offline mode, RAG must fail loudly."""

    monkeypatch.delenv("RAGLIB_OFFLINE", raising=False)
    with pytest.raises(ValueError, match="No chat model configured"):
        RAG("text")
    assert RAG("text", offline=True).stats()["llm"] == "MockLLMClient"


def test_unknown_rag_type_and_tools_validation() -> None:
    """Invalid configuration fails fast."""

    with pytest.raises(ValueError, match="Unknown rag_type"):
        RAG("text", rag_type="nope")
    with pytest.raises(ValueError, match="tools"):
        RAG("text", rag_type="advanced", tools=[Tool("x", "y", lambda q: q)])


def test_answers_carry_citations_and_trace_events() -> None:
    """Generated answers are cited and every stage is timed."""

    rag = RAG(CORPUS)
    result = rag.query("What does the Calvin cycle fix?")
    assert result.citations and result.citations[0].index == 1
    assert {event.stage for event in result.trace_events} >= {"retrieve", "generate"}
    assert result.usage["latency_ms"] >= 0


def test_generator_trims_instead_of_dropping() -> None:
    """A document that overflows the budget is truncated, not silently dropped."""

    generator = Generator(llm_client=MockLLMClient(), max_context_tokens=80, min_doc_tokens=10)
    docs = [Document(id="a", content="short fact."), Document(id="b", content="long " * 400)]
    packed, prompt = generator.build_prompt("q", docs)
    assert [d.id for d in packed] == ["a", "b"]
    assert packed[1].metadata.get("trimmed")
    assert count_tokens(prompt) < 200


def test_stream_yields_tokens_and_returns_result() -> None:
    """Streaming yields text and returns the final GenerationResult."""

    rag = RAG(CORPUS)
    stream = rag.stream("What releases ATP?")
    chunks = []
    try:
        while True:
            chunks.append(next(stream))
    except StopIteration as stop:
        result = stop.value
    assert "".join(chunks) == result.answer
    assert result.sources


def test_sessions_are_isolated_and_follow_ups_are_condensed() -> None:
    """Session A's history never reaches session B's prompt; follow-ups get a condense step."""

    llm = CountingLLM()
    rag = RAG(CORPUS, chat_llm=llm, embedding_llm="mock", rag_type="memory", llm_cache=False)
    rag.query("Tell me about photosynthesis", session_id="alice")
    llm.prompts.clear()
    rag.query("What about mitochondria?", session_id="bob")
    assert not any("photosynthesis" in prompt.lower() and "conversation" in prompt.lower() for prompt in llm.prompts)
    llm.prompts.clear()
    rag.query("Where does it happen?", session_id="alice")
    assert any("standalone question" in prompt for prompt in llm.prompts)


def test_conversation_store_token_budget() -> None:
    """History is capped by tokens, keeping the most recent turns."""

    store = ConversationStore(max_turns=50, max_tokens=30)
    memory = store.get("s")
    for index in range(20):
        memory.add(f"question {index}", "answer " * 5)
    context = memory.get_context()
    assert "question 19" in context and "question 0" not in context


def test_crag_never_calls_web_without_provider() -> None:
    """Default CRAG keeps queries private: no web retriever exists unless configured."""

    rag = RAG(CORPUS, rag_type="corrective")
    assert rag.pipeline.web_retriever is None
    result = rag.query("Who won the football world cup in 1930?")
    assert not any(step.startswith("web_search") for step in result.reasoning_trace)


def test_crag_verdict_follows_paper() -> None:
    """CORRECT if any doc is correct, INCORRECT only if all are incorrect."""

    class FixedGrader(RelevanceGrader):
        def __init__(self, labels):
            super().__init__()
            self.labels = labels

        def grade(self, query, documents):
            return self.labels[: len(documents)]

    docs = [Document(id=f"d{i}", content=f"sentence number {i} is here.") for i in range(3)]
    rag = CorrectiveRAG(grader=FixedGrader(["incorrect", "correct", "incorrect"]))
    assert rag._evaluate("q", docs)[0] == "correct"
    rag.grader = FixedGrader(["incorrect", "ambiguous", "incorrect"])
    assert rag._evaluate("q", docs)[0] == "ambiguous"
    rag.grader = FixedGrader(["incorrect"] * 3)
    assert rag._evaluate("q", docs)[0] == "incorrect"


def test_grader_uses_one_llm_call_for_all_documents() -> None:
    """Batched grading: N documents -> 1 LLM call (v1 made N sequential calls)."""

    llm = CountingLLM()
    docs = [Document(id=str(i), content=text) for i, text in enumerate(CORPUS)]
    labels = RelevanceGrader(llm_client=llm).grade("What does photosynthesis convert?", docs)
    assert len(llm.prompts) == 1 and len(labels) == len(docs)
    assert labels[0] == "correct"


def test_agentic_rag_uses_custom_tools() -> None:
    """The agent loop can call user tools and cite their output."""

    class ToolPickingLLM(MockLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            if "research agent" in prompt:
                if "Step 1:" in prompt:
                    return json.dumps({"action": "finish", "input": ""})
                return json.dumps({"action": "fx_rate", "input": "USD to INR"})
            return super().complete(prompt, system)

    tool = Tool("fx_rate", "Get a currency exchange rate", lambda q: "1 USD = 83.2 INR today.")
    rag = RAG(CORPUS, chat_llm=ToolPickingLLM(), embedding_llm="mock", rag_type="agentic", tools=[tool])
    result = rag.query("What is the USD to INR rate?")
    assert any(doc.source == "tool:fx_rate" for doc in result.sources)
    assert any("fx_rate" in step for step in result.reasoning_trace)


def test_multi_hop_plans_next_hop_from_evidence() -> None:
    """Hop 2 must be generated from hop-1 evidence (true multi-hop)."""

    class HopLLM(MockLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            if "planning multi-hop retrieval" in prompt:
                if "mitochondria ATP" not in prompt:
                    return json.dumps({"done": False, "next_query": "mitochondria ATP"})
                return json.dumps({"done": True})
            return super().complete(prompt, system)

    rag = RAG(CORPUS, chat_llm=HopLLM(), embedding_llm="mock", rag_type="multi_hop", top_k=2)
    result = rag.query("What does the Calvin cycle use?")
    assert "hops:2" in result.reasoning_trace


def test_self_rag_respects_llm_call_budget() -> None:
    """Self-RAG never exceeds max_llm_calls judge calls, even if judges keep failing it."""

    class HarshLLM(MockLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            if "relevance evaluator" in prompt:
                return "REWRITE"
            return super().complete(prompt, system)

    rag = RAG(CORPUS, chat_llm=HarshLLM(), embedding_llm="mock", rag_type="self", max_llm_calls=4)
    result = rag.query("What is photosynthesis?")
    assert result.metadata["self_rag_llm_calls"] <= 4


GRAPH_CORPUS = [
    "Marie Curie worked at the University of Paris. Marie Curie discovered Polonium with Pierre Curie.",
    "Pierre Curie was married to Marie Curie. Pierre Curie studied Piezoelectricity.",
    "Albert Einstein developed Relativity. Albert Einstein worked at Princeton University.",
    "Princeton University hosted Albert Einstein and Kurt Godel.",
]


def test_graph_rag_local_search_makes_no_llm_calls_before_generation() -> None:
    """Local search = entity vector search + graph walk, zero LLM calls."""

    llm = CountingLLM()
    rag = RAG(GRAPH_CORPUS, chat_llm=llm, embedding_llm="mock", rag_type="graph", llm_cache=False, chunk_size=40, chunk_overlap=5)
    stats = rag.stats()["graph"]
    assert stats["entities"] >= 5 and stats["relations"] >= 3

    llm.prompts.clear()
    docs = rag._graph_retriever.retrieve("Who did Pierre Curie marry?", mode="local")
    assert llm.prompts == []
    assert any(doc.metadata.get("graph") == "entities" for doc in docs)
    assert any("Pierre" in doc.content for doc in docs if doc.metadata.get("graph") == "chunk")

    result = rag.query("Who did Pierre Curie marry?")
    assert "graph_mode:local" in result.reasoning_trace
    assert result.citations


def test_graph_rag_global_search_uses_prefiltered_communities() -> None:
    """Global search maps at most top_communities reports."""

    rag = RAG(
        GRAPH_CORPUS,
        embedding_llm="mock",
        rag_type="graph",
        chunk_size=40, chunk_overlap=5,
        graph_kwargs={"top_communities": 1},
    )
    result = rag.query("What are the main themes across all the documents?")
    assert "graph_mode:global" in result.reasoning_trace
    assert rag._graph_retriever.global_search.last_mapped <= 1
    assert rag.stats()["graph"]["communities"] >= 1


def test_graph_extraction_is_cached_and_persisted(tmp_path) -> None:
    """Re-ingesting unchanged chunks costs no LLM calls; graphs survive save/load."""

    llm = CountingLLM()
    rag = RAG(GRAPH_CORPUS, chat_llm=llm, embedding_llm="mock", rag_type="graph", llm_cache=False, chunk_size=40, chunk_overlap=5)
    calls = rag.graph.extractor.llm_calls
    rag.add(GRAPH_CORPUS)
    assert rag.graph.extractor.llm_calls == calls
    rag.rebuild_graph_communities()
    rag.save(str(tmp_path / "g"))

    llm2 = CountingLLM()
    restored = RAG(chat_llm=llm2, embedding_llm="mock", rag_type="graph", llm_cache=False).load(str(tmp_path / "g"))
    assert restored.stats()["graph"]["entities"] == rag.stats()["graph"]["entities"]
    assert not any("knowledge graph extraction" in prompt for prompt in llm2.prompts)


def test_graph_delete_removes_orphan_entities() -> None:
    """Deleting a source removes entities only mentioned there."""

    rag = RAG(GRAPH_CORPUS, embedding_llm="mock", rag_type="graph", chunk_size=40, chunk_overlap=5)
    assert rag.graph.store.get_entity("kurt godel") is not None
    rag.delete(GRAPH_CORPUS[3])
    assert rag.graph.store.get_entity("kurt godel") is None


def test_graph_not_built_for_other_rag_types() -> None:
    """Non-graph pipelines pay nothing for GraphRAG."""

    llm = CountingLLM()
    RAG(GRAPH_CORPUS, chat_llm=llm, embedding_llm="mock", rag_type="advanced")
    assert not any("knowledge graph extraction" in prompt for prompt in llm.prompts)


def test_retrieval_metrics() -> None:
    """hit_rate / mrr / recall are computed against gold ids."""

    docs = [Document(id="x_chunk_1_aa", content="", metadata={"parent_id": "x"}), Document(id="y", content="")]
    metrics = retrieval_metrics(docs, ["y"], k=2)
    assert metrics["hit_rate@2"] == 1.0 and metrics["mrr@2"] == 0.5
    assert retrieval_metrics(docs, ["x"], k=2)["mrr@2"] == 1.0


def test_evaluate_end_to_end() -> None:
    """evaluate() runs questions and aggregates retrieval + judge metrics."""

    rag = RAG(CORPUS)
    parent = rag.retriever.documents[0].metadata["parent_id"]
    report = evaluate(
        rag,
        [{"question": "What does photosynthesis convert?", "relevant_ids": [parent]}],
        judge_llm=MockLLMClient(),
        k=5,
    )
    assert report.summary["hit_rate@5"] == 1.0
    assert 0.0 <= report.summary["faithfulness"] <= 1.0
    assert report.summary["errors"] == 0


def test_in_memory_store_delete_and_filter() -> None:
    """Vector store supports delete, count and filters."""

    store = InMemoryVectorStore()
    from raglib.vectorstores.base_store import VectorRecord

    store.upsert([VectorRecord("a", [1.0, 0.0], "a", {"k": 1}), VectorRecord("b", [0.9, 0.1], "b", {"k": 2})])
    assert [hit.document_id for hit in store.query([1.0, 0.0], 5, filters={"k": {"$gte": 2}})] == ["b"]
    store.delete(["a"])
    assert store.count() == 1
