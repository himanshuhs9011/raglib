"""Evaluation for RAG pipelines: retrieval metrics + LLM-judged answer quality."""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from raglib.core.concurrency import parallel_map
from raglib.core.parsing import parse_json
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

__all__ = ["EvalSample", "EvalReport", "evaluate", "retrieval_metrics"]


@dataclass
class EvalSample:
    """Per-question evaluation result."""

    question: str
    answer: str
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    samples: List[EvalSample]
    summary: Dict[str, float]

    def __str__(self) -> str:
        """Readable summary table."""

        lines = [f"{name:<22} {value:.3f}" for name, value in sorted(self.summary.items())]
        return "RAG evaluation\n" + "\n".join(lines)


def _matches(doc: Document, relevant: Sequence[str]) -> bool:
    """A retrieved chunk matches a gold id if ids are equal or the chunk came from that parent/file."""

    candidates = {doc.id, str(doc.metadata.get("parent_id", "")), str(doc.metadata.get("file", "")), doc.source}
    return any(item and (item in candidates or doc.id.startswith(f"{item}_chunk_")) for item in relevant)


def retrieval_metrics(sources: Sequence[Document], relevant: Sequence[str], k: int) -> Dict[str, float]:
    """hit_rate@k, MRR@k, recall@k and nDCG@k against gold document/chunk ids."""

    top = list(sources)[:k]
    hits = [_matches(doc, relevant) for doc in top]
    first = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    covered = sum(1 for item in relevant if any(_matches(doc, [item]) for doc in top))
    dcg = sum(1.0 / math.log2(index + 1) for index, hit in enumerate(hits, start=1) if hit)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
    return {
        f"hit_rate@{k}": 1.0 if first else 0.0,
        f"mrr@{k}": 1.0 / first if first else 0.0,
        f"recall@{k}": covered / len(relevant) if relevant else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }


def _judge(judge: BaseLLMClient, instruction: str) -> Optional[float]:
    """Ask the judge for a 0..1 score."""

    data = parse_json(
        judge.complete(
            "You are an evaluation judge for a retrieval-augmented QA system.\n"
            f"{instruction}\n"
            'Reply with JSON only: {"score": <number between 0 and 1>, "reason": "<short>"}'
        )
    )
    try:
        score = float(data.get("score")) if isinstance(data, dict) else None
    except (TypeError, ValueError):
        return None
    return None if score is None else max(0.0, min(1.0, score))


def _answer_metrics(judge: BaseLLMClient, question: str, result: GenerationResult, reference: Optional[str]) -> Dict[str, float]:
    """LLM-judged faithfulness, answer relevance, context precision and (with a reference) correctness."""

    context = "\n\n".join(f"[{i}] {doc.content[:800]}" for i, doc in enumerate(result.sources[:8], start=1)) or "(no context)"
    metrics: Dict[str, float] = {}
    checks = {
        "faithfulness": f"Context:\n{context}\n\nAnswer:\n{result.answer}\n\n"
        "Score the fraction of factual claims in the answer that are supported by the context.",
        "answer_relevance": f"Question: {question}\n\nAnswer:\n{result.answer}\n\n"
        "Score how directly and completely the answer addresses the question.",
        "context_precision": f"Question: {question}\n\nContext:\n{context}\n\n"
        "Score the fraction of context passages that are relevant to the question.",
    }
    if reference:
        checks["correctness"] = (
            f"Question: {question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{result.answer}\n\n"
            "Score how well the candidate matches the facts of the reference answer."
        )
    for name, instruction in checks.items():
        score = _judge(judge, instruction)
        if score is not None:
            metrics[name] = score
    return metrics


def evaluate(
    rag: Any,
    dataset: Sequence[Dict[str, Any]],
    judge_llm: Optional[BaseLLMClient] = None,
    k: int = 5,
    max_workers: int = 4,
) -> EvalReport:
    """Run every question through ``rag`` and score it.

    Each dataset row: ``{"question": str, "relevant_ids": [...] (optional), "reference": str (optional)}``.
    Retrieval metrics need ``relevant_ids`` (chunk ids, parent document ids, or file paths).
    Answer-quality metrics need ``judge_llm`` (use a strong model, ideally different from the generator).
    """

    def run_one(row: Dict[str, Any]) -> EvalSample:
        question = str(row["question"])
        try:
            result = rag.query(question)
        except Exception as exc:  # noqa: BLE001
            return EvalSample(question=question, answer="", error=f"{type(exc).__name__}: {exc}")
        sample = EvalSample(question=question, answer=result.answer, retrieved_ids=[doc.id for doc in result.sources])
        relevant = row.get("relevant_ids") or []
        if relevant:
            sample.metrics.update(retrieval_metrics(result.sources, relevant, k))
        if judge_llm is not None:
            sample.metrics.update(_answer_metrics(judge_llm, question, result, row.get("reference")))
        sample.metrics["latency_ms"] = float(result.usage.get("latency_ms", 0.0))
        return sample

    samples = parallel_map(run_one, dataset, max_workers)

    names = sorted({name for sample in samples for name in sample.metrics})
    summary = {
        name: statistics.fmean(sample.metrics[name] for sample in samples if name in sample.metrics) for name in names
    }
    summary["errors"] = float(sum(1 for sample in samples if sample.error))
    return EvalReport(samples=samples, summary=summary)
