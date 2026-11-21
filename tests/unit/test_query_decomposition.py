"""Multi-query retrieval keeps every part of a compound question covered (found on MultiHop-RAG)."""

from __future__ import annotations

from raglib.components.multi_query_retriever import cover_each_list
from raglib.core.fusion import reciprocal_rank_fusion
from raglib.schemas import Document


def _docs(*ids: str):
    return [Document(id=i, content=i) for i in ids]


def test_each_query_keeps_its_best_hit() -> None:
    # Three queries: the first two favour article A, the third (one part of the question) finds B.
    lists = [_docs("a1", "a2", "a3", "a4"), _docs("a1", "a3", "a2", "a4"), _docs("b1", "a1", "a2", "a3")]
    rrf_only = [doc.id for doc in reciprocal_rank_fusion(lists, weights=[1.5, 1.0, 1.0])[:3]]
    assert "b1" not in rrf_only  # the bug: plain RRF drops the part only one query found
    covered = [doc.id for doc in cover_each_list(lists, reciprocal_rank_fusion(lists, weights=[1.5, 1.0, 1.0]), 3)]
    assert covered[:2] == ["a1", "b1"] and len(covered) == len(set(covered))


def test_paraphrases_keep_rrf_order() -> None:
    lists = [_docs("a", "b", "c"), _docs("a", "c", "b"), _docs("a", "b", "c")]
    fused = reciprocal_rank_fusion(lists)
    assert [d.id for d in cover_each_list(lists, fused, 3)] == [d.id for d in fused[:3]]


def test_expander_prompt_asks_for_one_query_per_part() -> None:
    from raglib.components.query_expander import QueryExpander
    from raglib.llm.base_client import BaseLLMClient

    seen = []

    class _LLM(BaseLLMClient):
        def complete(self, prompt: str, system: str = "") -> str:
            seen.append(prompt)
            return "Wired Sony headphones Cyber Monday value\nMusic Business Worldwide artists deals"

        def chat(self, messages):
            return ""

    variants = QueryExpander(_LLM()).expand("Does Wired praise Sony headphones while MBW reports artists seek deals?")
    assert len(variants) == 3 and "one focused query for EACH part" in seen[0]
