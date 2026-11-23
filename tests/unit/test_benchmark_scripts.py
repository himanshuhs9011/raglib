"""The chunking benchmark scripts' scoring helpers (no network, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from raglib.schemas import Document

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / "chunking" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


real_eval = _load("real_eval")


def test_golden_rows_are_validated(tmp_path) -> None:
    good = tmp_path / "golden.jsonl"
    good.write_text('{"question": "Q?", "relevant_text": "a snippet"}\n\n{"question": "R?", "relevant_ids": ["a.md"]}\n', encoding="utf-8")
    rows = real_eval.load_golden(good)
    assert rows[0]["relevant_text"] == ["a snippet"] and rows[1]["relevant_ids"] == ["a.md"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "Q?"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_text"):
        real_eval.load_golden(bad)
    assert len(real_eval.load_golden(ROOT / "benchmarks" / "chunking" / "golden.example.jsonl")) == 3


def test_scoring_by_text_snippet_and_by_relative_file_path() -> None:
    chunks = [
        Document(id="c1", content="Refunds are   paid within FOURTEEN days.", metadata={"file": "C:/docs\\refunds.md"}),
        Document(id="c2", content="Shipping takes a week.", metadata={"file": "C:/docs\\policies\\travel.md"}),
    ]
    recall, precision = real_eval.score(chunks, {"relevant_text": ["refunds are paid within fourteen days"], "relevant_ids": []}, k=2)
    assert (recall, precision) == (1.0, 0.5)
    recall, precision = real_eval.score(chunks, {"relevant_text": [], "relevant_ids": ["policies/travel.md"]}, k=2)
    assert (recall, precision) == (1.0, 0.5)
    recall, _ = real_eval.score(chunks, {"relevant_text": [], "relevant_ids": ["missing.md"]}, k=2)
    assert recall == 0.0


def test_configurations_cover_every_threshold_type_and_minimum() -> None:
    args = real_eval.main.__globals__["argparse"].Namespace(chunk_size=500, overlap=100, number_of_chunks=8, min_tokens=[32, 64, 125])
    labels = [label for label, _ in real_eval.configurations(args)]
    assert labels[0].startswith("fixed") and len(labels) == 1 + 5 * 3
    for kind in ("percentile", "interquartile", "gradient", "number_of_chunks", "standard_deviation"):
        assert sum(kind in label for label in labels) == 3


def test_guardrail_cost_benchmark_runs_offline(tmp_path) -> None:
    import json

    spec = importlib.util.spec_from_file_location("cost_bench", ROOT / "benchmarks" / "guardrails" / "cost_bench.py")
    cost_bench = importlib.util.module_from_spec(spec)
    sys.modules["cost_bench"] = cost_bench
    spec.loader.exec_module(cost_bench)
    out = tmp_path / "bench.json"
    assert cost_bench.main(["--no-sleep", "--json", str(out)]) == 0
    setups = {row["setup"]: row for row in json.loads(out.read_text(encoding="utf-8"))["setups"]}
    assert set(setups) == {"off", "production", "all / local", "all / api"}
    assert setups["off"]["blocked"] == {} and setups["off"]["attacks_blocked"] == 0
    assert setups["all / api"]["attacks_blocked"] == setups["all / api"]["attacks_total"]
    assert setups["all / api"]["llm_calls_saved"].get("topic", 0) > 0
    assert setups["all / api"]["cheap_llm_calls"] > setups["all / local"]["cheap_llm_calls"]
    assert any(stage.startswith("guard.") for stage in setups["production"]["stages"])
    assert cost_bench.main(["--llm", "openai/gpt-4.1-mini"]) == 2  # paid: needs --yes, nothing sent
