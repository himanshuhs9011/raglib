"""graph_llm: build the knowledge graph with a different (cheaper) model than the answers."""

from __future__ import annotations

from typing import Dict, List

import pytest

from raglib import RAG
from raglib.llm.base_client import BaseLLMClient, unwrap_llm


class _Recorder(BaseLLMClient):
    def __init__(self, name: str) -> None:
        self.name = name
        self.prompts: List[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        if "knowledge graph extraction" in prompt:
            return (
                '{"entities": [{"name": "Flexport", "type": "ORGANIZATION", "description": "Logistics company."},'
                ' {"name": "Ryan Petersen", "type": "PERSON", "description": "Flexport CEO."}],'
                ' "relations": [{"source": "Ryan Petersen", "target": "Flexport", "type": "LEADS",'
                ' "description": "Petersen runs Flexport.", "weight": 8}]}'
            )
        return "Ryan Petersen leads Flexport [1]."

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return ""


def test_graph_is_built_by_graph_llm_and_answers_by_chat_llm() -> None:
    answers, indexer = _Recorder("answers"), _Recorder("indexer")
    rag = RAG(
        ["Ryan Petersen returned as CEO of Flexport in 2023."],
        chat_llm=answers,
        graph_llm=indexer,
        embedding_llm="mock",
        rag_type="graph",
        llm_cache=False,
    )
    rag.query("Who leads Flexport?")
    assert any("knowledge graph extraction" in prompt for prompt in indexer.prompts)
    assert not any("knowledge graph extraction" in prompt for prompt in answers.prompts)
    assert answers.prompts, "the answer still comes from the chat model"
    assert rag.stats()["graph"]["entities"] >= 2


def test_graph_llm_key_is_not_sent_to_another_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        RAG(llm="openrouter", chat_api_key="sk-or-v1-secret", graph_llm="nvidia", rag_type="graph", embedding_llm="none")
    rag = RAG(llm="openrouter", chat_api_key="sk-or-v1-secret", graph_llm="openrouter/openai/gpt-4.1-nano",
              rag_type="graph", embedding_llm="none")
    graph_client = unwrap_llm(rag._graph_llm)
    assert graph_client.model_name == "openai/gpt-4.1-nano" and graph_client.api_key == "sk-or-v1-secret"


def test_graph_llm_without_graph_is_an_error() -> None:
    with pytest.raises(ValueError, match="graph_llm needs"):
        RAG(["x"], llm="mock", graph_llm="mock")


class _FlakyExtractor(_Recorder):
    """Fails extraction for chunks mentioning 'Kelce' until healed, like a rate-limited provider."""

    def __init__(self) -> None:
        super().__init__("flaky")
        self.healthy = False

    def complete(self, prompt: str, system: str = "") -> str:
        if "knowledge graph extraction" in prompt and "Kelce" in prompt and not self.healthy:
            raise RuntimeError("Error code: 429 - Too Many Requests")
        if "knowledge graph extraction" in prompt and "Kelce" in prompt:
            return ('{"entities": [{"name": "Travis Kelce", "type": "PERSON", "description": "Chiefs tight end."}],'
                    ' "relations": []}')
        return super().complete(prompt, system)


def test_failed_extraction_is_not_cached_and_can_be_retried(tmp_path) -> None:
    flaky = _FlakyExtractor()
    rag = RAG(
        ["Ryan Petersen returned as CEO of Flexport in 2023.", "Travis Kelce plays for the Kansas City Chiefs."],
        chat_llm=_Recorder("answers"), graph_llm=flaky, embedding_llm="mock", rag_type="graph", llm_cache=False,
        llm_max_retries=0,
    )
    graph = rag.stats()["graph"]
    assert graph["failed_extractions"] == 1, "the failed chunk is tracked, not silently stored as empty"
    names = {entity.name for entity in rag._graph_indexer.store.all_entities()}
    assert "Travis Kelce" not in names

    flaky.healthy = True
    assert rag.retry_graph_extraction()["failed"] == 0
    assert rag.stats()["graph"]["failed_extractions"] == 0
    assert "Travis Kelce" in {entity.name for entity in rag._graph_indexer.store.all_entities()}


def test_failed_extraction_is_retried_on_load(tmp_path) -> None:
    flaky = _FlakyExtractor()
    rag = RAG(["Travis Kelce plays for the Kansas City Chiefs."], chat_llm=_Recorder("answers"), graph_llm=flaky,
              embedding_llm="mock", rag_type="graph", llm_cache=False, llm_max_retries=0)
    rag.save(str(tmp_path / "index"))
    flaky.healthy = True
    restored = RAG(chat_llm=_Recorder("answers"), graph_llm=flaky, embedding_llm="mock", rag_type="graph",
                   llm_cache=False).load(str(tmp_path / "index"))
    assert restored.stats()["graph"]["failed_extractions"] == 0
    assert "Travis Kelce" in {entity.name for entity in restored._graph_indexer.store.all_entities()}


class _EmptyReplies(_Recorder):
    """A model that never returns usable JSON for one chunk (e.g. an empty reply at max_tokens)."""

    def complete(self, prompt: str, system: str = "") -> str:
        if "knowledge graph extraction" in prompt and "Kelce" in prompt:
            self.prompts.append(prompt)
            return ""
        return super().complete(prompt, system)


def test_a_chunk_that_always_fails_is_skipped_after_max_attempts(tmp_path) -> None:
    empty = _EmptyReplies("empty")
    texts = ["Ryan Petersen returned as CEO of Flexport in 2023.", "Travis Kelce plays for the Kansas City Chiefs."]
    kwargs = dict(chat_llm=_Recorder("answers"), graph_llm=empty, embedding_llm="mock", rag_type="graph",
                  llm_cache=False, llm_max_retries=0, graph_kwargs={"max_extraction_attempts": 2})
    rag = RAG(texts, **kwargs)
    assert rag.stats()["graph"]["failed_extractions"] == 1  # attempt 1
    rag.save(str(tmp_path / "index"))
    restored = RAG(**kwargs).load(str(tmp_path / "index"))  # attempt 2 on load, then give up
    assert restored.stats()["graph"]["failed_extractions"] == 0
    assert restored.stats()["graph"]["skipped_extractions"] == 1
    restored.save(str(tmp_path / "index"))

    def kelce_calls() -> int:
        return sum("Kelce" in prompt for prompt in empty.prompts)

    before = kelce_calls()
    again = RAG(**kwargs).load(str(tmp_path / "index"))
    again.retry_graph_extraction()
    assert kelce_calls() == before, "a skipped chunk is not paid for again on load or retry"
    assert again.stats()["graph"]["skipped_extractions"] == 1
    skipped_id = next(iter(again._graph_indexer.skipped_chunk_ids))
    assert "Kelce" in again._retriever.get(skipped_id).content  # the chunk text stays in the index

    again.retry_graph_extraction(include_skipped=True)  # explicit opt-in tries it once more
    assert kelce_calls() == before + 1


def test_graph_summaries_never_push_evidence_out_of_the_context() -> None:
    """GraphRAG's entity/relation/community summaries have their own budget (MultiHop-RAG finding:
    they used ~60% of the context, so answers saw half as many evidence chunks)."""

    from raglib.components.generator import Generator
    from raglib.schemas import Document

    words = " ".join(["word"] * 400)  # roughly 400+ tokens each
    summaries = [Document(id=f"graph-{i}", content=words, metadata={"supplementary": True}) for i in range(4)]
    chunks = [Document(id=f"chunk-{i}", content=words) for i in range(5)]
    generator = Generator(_Recorder("g"), max_context_tokens=2200, max_supplementary_tokens=900, reorder="none")
    packed = generator._pack_documents(summaries + chunks)
    packed_chunks = [d.id for d in packed if d.id.startswith("chunk")]
    packed_summaries = [d for d in packed if d.id.startswith("graph")]
    assert len(packed_chunks) == 5, "every evidence chunk that fits max_context_tokens is kept"
    assert 1 <= len(packed_summaries) <= 3 and packed[0].id == "graph-0", "summaries come first, within their own budget"

    rag = RAG(["Ryan Petersen returned as CEO of Flexport in 2023."], chat_llm=_Recorder("answers"),
              graph_llm=_Recorder("graph"), embedding_llm="mock", rag_type="graph", llm_cache=False)
    result = rag.query("Who leads Flexport?")
    graph_sources = [s for s in result.sources if s.metadata.get("graph") in {"entities", "relations"}]
    assert graph_sources and all(s.metadata.get("supplementary") for s in graph_sources)


def test_graph_llm_replies_are_capped() -> None:
    rag = RAG(llm="openrouter", chat_api_key="sk-or-v1-x", graph_llm="ollama/qwen2.5:7b", rag_type="graph",
              embedding_llm="none")
    assert unwrap_llm(rag._graph_llm).max_tokens == 4096
    assert unwrap_llm(rag._llm).max_tokens is None, "answers are not capped by the graph setting"
    rag = RAG(llm="openrouter", chat_api_key="sk-or-v1-x", graph_llm="ollama/qwen2.5:7b", rag_type="graph",
              embedding_llm="none", graph_kwargs={"extraction_max_tokens": 1500})
    assert unwrap_llm(rag._graph_llm).max_tokens == 1500


def test_extraction_prompt_states_the_limit() -> None:
    from raglib.graphrag.extractor import GraphExtractor

    prompt = GraphExtractor(_Recorder("x"), max_entities_per_chunk=12)._prompt("Some text.")
    assert "at most 12 entities and 24 relations" in prompt
