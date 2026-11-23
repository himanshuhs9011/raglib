"""Cost tracking in USD: a configurable price table, per-query costs per stage, and an aggregate report.

::

    from raglib import RAG, CostReport, PriceTable

    prices = PriceTable(
        llm={"openai/gpt-4.1-mini": (0.40, 1.60)},          # USD per 1M input / output tokens
        embedding={"text-embedding-3-large": 0.13},          # USD per 1M tokens
        search={"cohere": 2.00},                             # USD per 1K searches
        vector_db_monthly=58.0,                              # fixed hosting cost, for projections
    )
    costs = CostReport(prices)
    rag = RAG("docs/", llm="openai/gpt-4.1-mini", prices=prices, on_event=costs)
    result = rag.query("...")
    result.usage["cost_usd"], result.usage["cost_by_stage"]  # e.g. {"generate": 0.0011, "query_embedding": 2e-06}
    costs.print_report(queries_per_day=10_000)               # totals, per stage / model, monthly projection

raglib ships **no built-in prices**: they change often and differ by contract, and a silently
wrong default is worse than an explicit "unpriced". Anything without a price is listed in
``cost_unpriced`` and counted as 0. Token counts are the provider's when it reports them, and
estimates otherwise (``usage_estimated``); embedding tokens are always estimated locally.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import IO, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from raglib.core.tracing import STAGE_USAGE_KEY, TOTAL_STAGE
from raglib.schemas import TraceEvent

PriceInput = Union["ModelPrice", Tuple[float, float], Sequence[float], Mapping[str, float], float]


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. Prompt-cache prices default to the table's multipliers of ``input_per_1m``."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cache_read_per_1m: Optional[float] = None
    cache_write_per_1m: Optional[float] = None

    @classmethod
    def coerce(cls, value: PriceInput) -> "ModelPrice":
        """Accept a ModelPrice, ``(input, output)``, a dict of the fields, or one number (input = output)."""

        if isinstance(value, ModelPrice):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if isinstance(value, (int, float)):
            return cls(float(value), float(value))
        items = list(value)
        if len(items) not in (2, 4):
            raise ValueError("an LLM price is (input_per_1m, output_per_1m) or 4 values with cache read/write")
        return cls(*[float(item) for item in items])


class PriceTable:
    """Prices for LLM tokens, embedding tokens and billable searches.

    Keys are matched most specific first: ``"provider/model"``, then ``"model"`` without the
    provider, then ``"provider/*"``, then ``"*"``. ``search`` keys are ``"cohere"`` (rerank) and
    ``"web:<provider>"`` (e.g. ``"web:tavily"``). ``cache_read_multiplier`` / ``cache_write_multiplier``
    price prompt-cache tokens when a model has no explicit cache prices (Anthropic: ~0.1x / 1.25x).
    """

    def __init__(
        self,
        llm: Optional[Mapping[str, PriceInput]] = None,
        embedding: Optional[Mapping[str, float]] = None,
        search: Optional[Mapping[str, float]] = None,
        vector_db_monthly: float = 0.0,
        cache_read_multiplier: float = 0.1,
        cache_write_multiplier: float = 1.25,
    ):
        """Build the table (all prices in USD)."""

        self.llm = {str(key): ModelPrice.coerce(value) for key, value in (llm or {}).items()}
        self.embedding = {str(key): float(value) for key, value in (embedding or {}).items()}
        self.search = {str(key): float(value) for key, value in (search or {}).items()}
        self.vector_db_monthly = float(vector_db_monthly)
        self.cache_read_multiplier = cache_read_multiplier
        self.cache_write_multiplier = cache_write_multiplier

    @classmethod
    def coerce(cls, value: Union["PriceTable", Mapping[str, Any], None]) -> Optional["PriceTable"]:
        """Accept a PriceTable, a dict of its arguments, or None."""

        if value is None or isinstance(value, PriceTable):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"prices must be a PriceTable or a dict, got {type(value).__name__}")

    @staticmethod
    def _lookup(table: Mapping[str, Any], name: str) -> Optional[Any]:
        if name in table:
            return table[name]
        provider, _, model = name.partition("/")
        if model and model in table:
            return table[model]
        if model and f"{provider}/*" in table:
            return table[f"{provider}/*"]
        return table.get("*")

    def llm_price(self, model: str) -> Optional[ModelPrice]:
        """The price of an LLM (``"provider/model"``), or None."""

        return self._lookup(self.llm, model)

    def cost(self, usage: Mapping[str, Any]) -> Dict[str, Any]:
        """Price a usage breakdown (``UsageMeter.breakdown()`` or a stage's ``meta["_usage"]``).

        Returns ``total``, ``llm``, ``embedding``, ``search`` (USD), ``by_model`` and ``unpriced``.
        """

        by_model: Dict[str, float] = {}
        unpriced: List[str] = []
        totals = {"llm": 0.0, "embedding": 0.0, "search": 0.0}
        for model, row in (usage.get("llm") or {}).items():
            price = self.llm_price(model)
            if price is None:
                unpriced.append(model)
                continue
            read = row.get("cache_read_tokens", 0)
            write = row.get("cache_write_tokens", 0)
            uncached = max(0, row.get("prompt_tokens", 0) - read - write)
            read_price = price.cache_read_per_1m if price.cache_read_per_1m is not None else price.input_per_1m * self.cache_read_multiplier
            write_price = price.cache_write_per_1m if price.cache_write_per_1m is not None else price.input_per_1m * self.cache_write_multiplier
            usd = (
                uncached * price.input_per_1m
                + read * read_price
                + write * write_price
                + row.get("completion_tokens", 0) * price.output_per_1m
            ) / 1e6
            by_model[model] = by_model.get(model, 0.0) + usd
            totals["llm"] += usd
        for model, row in (usage.get("embedding") or {}).items():
            price = self._lookup(self.embedding, model)
            if price is None:
                unpriced.append(model)
                continue
            usd = row.get("tokens", 0) * price / 1e6
            by_model[model] = by_model.get(model, 0.0) + usd
            totals["embedding"] += usd
        for provider, row in (usage.get("search") or {}).items():
            price = self._lookup(self.search, provider)
            if price is None:
                unpriced.append(provider)
                continue
            usd = row.get("calls", 0) * price / 1e3
            by_model[provider] = by_model.get(provider, 0.0) + usd
            totals["search"] += usd
        return {"total": sum(totals.values()), **totals, "by_model": by_model, "unpriced": sorted(set(unpriced))}


def price_run(prices: PriceTable, breakdown: Mapping[str, Any], events: Sequence[TraceEvent]) -> Dict[str, Any]:
    """Cost fields for ``result.usage``: total, by stage (innermost stage owns each call), by model, by category."""

    total = prices.cost(breakdown)
    by_stage: Dict[str, float] = {}
    for event in events:
        usage = event.meta.get(STAGE_USAGE_KEY)
        if usage:
            by_stage[event.stage] = by_stage.get(event.stage, 0.0) + prices.cost(usage)["total"]
        elif event.stage.startswith(("guard.", "judge.")):
            by_stage.setdefault(event.stage, 0.0)  # every guard / judge gets its own line, free ones at $0
    other = total["total"] - sum(by_stage.values())
    if other > 1e-12:
        by_stage["other"] = other  # calls made outside any traced stage
    return {
        "cost_usd": round(total["total"], 8),
        "cost_by_stage": {stage: round(usd, 8) for stage, usd in by_stage.items()},
        "cost_by_model": {model: round(usd, 8) for model, usd in total["by_model"].items()},
        "cost_by_category": {key: round(total[key], 8) for key in ("llm", "embedding", "search")},
        "cost_unpriced": total["unpriced"],
    }


class CostReport:
    """Aggregate query costs over many runs; pass it as ``on_event`` (needs ``RAG(prices=...)``).

    ``report()`` sums the ``total`` events' cost fields; ``projection(queries_per_day)`` turns the
    average cost per query into a monthly estimate, with the vector database's fixed monthly cost
    kept separate - in a typical RAG the LLM context is ~99% of the bill and the vector DB ~1%.
    """

    def __init__(self, prices: Optional[PriceTable] = None):
        """``prices`` is only needed for ``vector_db_monthly`` in projections."""

        self.prices = prices
        self.queries = 0
        self.total_usd = 0.0
        self.by_stage: Dict[str, float] = {}
        self.by_model: Dict[str, float] = {}
        self.by_category: Dict[str, float] = {}
        self.unpriced: set = set()
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        """``on_event`` callback: counts every finished query that carries cost fields."""

        if event.stage != TOTAL_STAGE or "cost_usd" not in event.meta:
            return
        meta = event.meta
        with self._lock:
            self.queries += 1
            self.total_usd += float(meta["cost_usd"])
            for target, key in ((self.by_stage, "cost_by_stage"), (self.by_model, "cost_by_model"), (self.by_category, "cost_by_category")):
                for name, usd in (meta.get(key) or {}).items():
                    target[name] = target.get(name, 0.0) + float(usd)
            self.unpriced.update(meta.get("cost_unpriced") or [])

    def report(self) -> Dict[str, Any]:
        """Totals and averages so far (USD)."""

        with self._lock:
            queries = self.queries
            return {
                "queries": queries,
                "total_usd": self.total_usd,
                "avg_usd_per_query": self.total_usd / queries if queries else 0.0,
                "by_stage": dict(self.by_stage),
                "by_model": dict(self.by_model),
                "by_category": dict(self.by_category),
                "unpriced": sorted(self.unpriced),
            }

    def projection(self, queries_per_day: float, days: float = 30.0) -> Dict[str, float]:
        """Monthly estimate per category from the average cost per query, plus vector DB hosting."""

        report = self.report()
        queries = report["queries"] or 1
        monthly = {key: usd / queries * queries_per_day * days for key, usd in report["by_category"].items()}
        monthly["vector_db"] = self.prices.vector_db_monthly if self.prices else 0.0
        monthly["total"] = sum(monthly.values())
        return monthly

    def format_report(self, queries_per_day: Optional[float] = None) -> str:
        """Readable text report."""

        report = self.report()
        if not report["queries"]:
            return "No priced queries recorded (pass RAG(prices=...) and this report as on_event)."
        lines = [
            f"queries: {report['queries']}   total: ${report['total_usd']:.6f}   per query: ${report['avg_usd_per_query']:.6f}",
            "",
            "by stage:",
            *[f"  {name:<20} ${usd:.6f}" for name, usd in sorted(report["by_stage"].items(), key=lambda item: -item[1])],
            "by model:",
            *[f"  {name:<40} ${usd:.6f}" for name, usd in sorted(report["by_model"].items(), key=lambda item: -item[1])],
        ]
        if report["unpriced"]:
            lines.append(f"unpriced (counted as $0): {', '.join(report['unpriced'])}")
        if queries_per_day:
            monthly = self.projection(queries_per_day)
            lines += ["", f"monthly projection at {queries_per_day:,.0f} queries/day:"]
            lines += [f"  {name:<12} ${usd:,.2f}" for name, usd in monthly.items()]
        return "\n".join(lines)

    def print_report(self, queries_per_day: Optional[float] = None, file: Optional[IO[str]] = None) -> None:
        """Print ``format_report()``."""

        print(self.format_report(queries_per_day), file=file or sys.stdout)
