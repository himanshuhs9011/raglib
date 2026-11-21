"""Corrective RAG without web search must not answer from an empty context (found on MultiHop-RAG)."""

from __future__ import annotations

from typing import Dict, List

from raglib import RAG
from raglib.llm.base_client import BaseLLMClient


class _StrictGrader(BaseLLMClient):
    """Grades every chunk INCORRECT, cannot refine queries, answers with whatever it is given."""

    def __init__(self) -> None:
        self.answer_prompts: List[str] = []

    def complete(self, prompt: str, system: str = "") -> str:
        if "Grade each retrieved document" in prompt:
            count = prompt.count("\n[")
            return '{"grades": [%s]}' % ", ".join(['"INCORRECT"'] * max(count, 1))
        if "<sources>" in prompt:
            self.answer_prompts.append(prompt)
            return "Answer from the sources [1]."
        return ""  # query refinement: no better query

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return ""


def test_incorrect_verdict_without_web_falls_back_to_retrieved_chunks() -> None:
    llm = _StrictGrader()
    rag = RAG(
        ["TechCrunch reported on 7 October that Flexport changed CEOs.", "The Verge covered the FTX trial."],
        chat_llm=llm,
        embedding_llm="mock",
        rag_type="corrective",
    )
    result = rag.query("Did TechCrunch and The Verge both report leadership changes?")
    assert result.metadata.get("crag_verdict") == "incorrect"
    assert result.metadata.get("crag_fallback") is True
    assert result.sources, "the retrieved chunks are used instead of an empty context"
    assert "Flexport" in llm.answer_prompts[-1] or "FTX" in llm.answer_prompts[-1]


def test_grader_prompt_counts_partial_multi_document_evidence() -> None:
    from raglib.components.grader import RelevanceGrader
    from raglib.schemas import Document

    seen: List[str] = []

    class _Recorder(BaseLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            seen.append(prompt)
            return '{"grades": ["AMBIGUOUS"]}'

        def chat(self, messages):
            return ""

    labels = RelevanceGrader(llm_client=_Recorder()).grade("Compare A and B", [Document(id="d", content="About A.")])
    assert labels == ["ambiguous"]
    assert "any one of the entities" in seen[0]
