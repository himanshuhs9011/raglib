"""Phase 6: USD cost tracking (price table, per-stage cost, report, ingest cost) and precision@K."""

from __future__ import annotations

import io
import sys
import types
from typing import List

import pytest

from raglib import RAG, CostReport, ModelPrice, PriceTable
from raglib.components.reranker import Reranker
from raglib.components.web_retriever import WebRetriever
from raglib.core.runtime import usage_scope
from raglib.eval import retrieval_metrics
from raglib.providers.base_provider import BaseSearchProvider
from raglib.schemas import Document

CORPUS = [
    "Acme Corp offices close on Friday afternoons in summer and the cafeteria opens at eight.",
    "The Orion launch window opens in March and closes in April for the whole launch team.",
]


def test_price_table_prices_tokens_cache_and_searches() -> None:
    prices = PriceTable(
        llm={"anthropic/claude-opus-5-5": (3.0, 15.0)},
        embedding={"text-embedding-3-large": 0.13},
        search={"cohere": 2.0},
    )
    usage = {
        "llm": {"anthropic/claude-opus-5-5": {"prompt_tokens": 1000, "completion_tokens": 500,
                                              "cache_read_tokens": 200, "cache_write_tokens": 100}},
        "embedding": {"text-embedding-3-large": {"tokens": 1000}},
        "search": {"cohere": {"calls": 2}, "web:tavily": {"calls": 1}},
    }
    cost = prices.cost(usage)
    # 700 uncached * 3 + 200 read * 0.3 + 100 write * 3.75 + 500 out * 15 = 10035 per 1M
    assert cost["llm"] == pytest.approx(0.010035)
    assert cost["embedding"] == pytest.approx(0.00013)
    assert cost["search"] == pytest.approx(0.004)
    assert cost["total"] == pytest.approx(0.010035 + 0.00013 + 0.004)
    assert cost["unpriced"] == ["web:tavily"]


def test_price_lookup_is_most_specific_first() -> None:
    prices = PriceTable(llm={"openai/gpt-4.1": (2, 8), "gpt-4.1-mini": (0.4, 1.6), "openai/*": (1, 1), "*": (9, 9)})
    assert prices.llm_price("openai/gpt-4.1") == ModelPrice(2, 8)
    assert prices.llm_price("azure/gpt-4.1-mini") == ModelPrice(0.4, 1.6)
    assert prices.llm_price("openai/o9") == ModelPrice(1, 1)
    assert prices.llm_price("groq/llama") == ModelPrice(9, 9)
    explicit = PriceTable(llm={"m": {"input_per_1m": 4, "output_per_1m": 20, "cache_read_per_1m": 0.2}})
    cost = explicit.cost({"llm": {"m": {"prompt_tokens": 1_000_000, "cache_read_tokens": 1_000_000}}})
    assert cost["total"] == pytest.approx(0.2)  # explicit cache-read price wins over the multiplier
    assert PriceTable.coerce({"llm": {"*": 1}}).llm_price("x") == ModelPrice(1, 1)


def _priced_rag(**kwargs) -> RAG:
    prices = PriceTable(llm={"mock/mock": (1.0, 2.0)}, embedding={"*": 0.5}, vector_db_monthly=58.0)
    return RAG(CORPUS, offline=True, prices=prices, **kwargs), prices


def test_query_usage_reports_cost_per_stage_and_model() -> None:
    rag, _ = _priced_rag(rewrite_query=True)
    usage = rag.query("When do Acme offices close?").usage
    assert usage["cost_usd"] > 0
    assert {"generate", "query_embedding", "query_rewrite"} <= set(usage["cost_by_stage"])
    assert sum(usage["cost_by_stage"].values()) == pytest.approx(usage["cost_usd"], rel=1e-6)
    assert set(usage["cost_by_model"]) >= {"mock/mock", "MockEmbedding"}
    assert usage["cost_by_category"]["llm"] > usage["cost_by_category"]["embedding"] > 0
    assert usage["cost_unpriced"] == []
    assert usage["usage_by_model"]["llm"]["mock/mock"]["calls"] == usage["llm_calls"]


def test_cached_repeat_costs_less_and_report_aggregates() -> None:
    costs = CostReport()
    rag, prices = _priced_rag(on_event=costs)
    costs.prices = prices
    first = rag.query("When do Acme offices close?").usage["cost_usd"]
    second = rag.query("When do Acme offices close?").usage["cost_usd"]
    assert second < first  # LLM and embedding cache hits are free
    third = rag.query("When does the Orion launch window open?").usage["cost_usd"]
    report = costs.report()
    assert report["queries"] == 3 and report["total_usd"] == pytest.approx(first + second + third)
    assert sum(report["by_stage"].values()) == pytest.approx(report["total_usd"], rel=1e-6)
    assert report["avg_usd_per_query"] == pytest.approx(report["total_usd"] / 3)
    monthly = costs.projection(queries_per_day=10_000)
    assert monthly["vector_db"] == 58.0 and monthly["total"] == pytest.approx(sum(v for k, v in monthly.items() if k != "total"))
    buffer = io.StringIO()
    costs.print_report(queries_per_day=10_000, file=buffer)
    assert "monthly projection" in buffer.getvalue() and "generate" in buffer.getvalue()


def test_without_prices_no_cost_fields_and_unpriced_models_are_listed() -> None:
    rag = RAG(CORPUS, offline=True)
    assert "cost_usd" not in rag.query("When do Acme offices close?").usage
    partial = RAG(CORPUS, offline=True, prices=PriceTable(llm={"mock/mock": (1, 1)}))
    assert partial.query("When do Acme offices close?").usage["cost_unpriced"] == ["MockEmbedding"]


def test_ingest_usage_counts_embedding_tokens_and_cost() -> None:
    rag, _ = _priced_rag()
    info = rag.ingest_usage()
    assert info["embedding_tokens"] > 0 and info["cost_usd"] > 0
    before = info["embedding_tokens"]
    rag.add(CORPUS)  # unchanged: nothing re-embedded
    assert rag.ingest_usage()["embedding_tokens"] == before


def test_cohere_rerank_and_web_search_calls_are_billable_searches(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCohere:
        def __init__(self, api_key: str) -> None:
            pass

        def rerank(self, model: str, query: str, documents: List[str]):
            return types.SimpleNamespace(
                results=[types.SimpleNamespace(index=i, relevance_score=1.0 / (i + 1)) for i in range(len(documents))]
            )

    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(ClientV2=FakeCohere))
    docs = [Document(id=f"d{i}", content=f"passage {i}") for i in range(150)]

    class FakeWeb(BaseSearchProvider):
        @property
        def name(self) -> str:
            return "fakeweb"

        def search(self, query: str, num_results: int = 5) -> List[Document]:
            return [Document(id="w", content="web result")]

    with usage_scope() as meter:
        Reranker(top_k=3, backend="cohere", api_key="test").rerank("q", docs)
        WebRetriever(provider=FakeWeb()).retrieve("q")
    searches = meter.breakdown()["search"]
    assert searches == {"cohere": {"calls": 2}, "web:fakeweb": {"calls": 1}}  # 150 docs = 2 search units


def test_precision_at_k_next_to_recall() -> None:
    sources = [Document(id=f"doc{i}_chunk_1_x", content="", metadata={"parent_id": f"doc{i}"}) for i in range(5)]
    metrics = retrieval_metrics(sources, relevant=["doc0", "doc3", "doc9"], k=5)
    assert metrics["precision@5"] == pytest.approx(2 / 5)
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert retrieval_metrics(sources[:2], relevant=["doc0"], k=5)["precision@5"] == pytest.approx(1 / 5)
