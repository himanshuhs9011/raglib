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


def _public_datasets():
    spec = importlib.util.spec_from_file_location("public_datasets", ROOT / "benchmarks" / "chunking" / "public_datasets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["public_datasets"] = module
    spec.loader.exec_module(module)
    return module


def test_miracl_subset_keeps_judged_documents_and_samples_distractors(tmp_path, monkeypatch) -> None:
    import gzip
    import json

    public = _public_datasets()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "topics.tsv").write_text("q1#0\tभारत की राजधानी क्या है?\nq2#0\tगंगा कहाँ बहती है?\nq3#0\tबिना जवाब\n", encoding="utf-8")
    (raw / "qrels.tsv").write_text("q1#0\tQ0\td1#0\t1\nq1#0\tQ0\td2#0\t0\nq2#0\tQ0\td3#0\t1\n", encoding="utf-8")
    rows = [{"docid": f"d{i}#0", "title": f"T{i}", "text": f"पाठ {i}।"} for i in range(1, 40)]
    with gzip.open(raw / "docs.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    empty = raw / "empty.gz"
    with gzip.open(empty, "wt", encoding="utf-8") as handle:
        handle.write("")
    files = {"topics": raw / "topics.tsv", "qrels": raw / "qrels.tsv", "docs-0": raw / "docs.gz", "docs-1": empty}

    def fake_download(url, target):
        return next(path for key, path in files.items() if key in url)

    monkeypatch.setattr(public, "download", fake_download)
    docs_dir, golden, info = public.prepare_miracl("miracl-hi", tmp_path / "cache", max_queries=5, max_docs=10, seed=1)
    names = {path.name for path in docs_dir.glob("*.txt")}
    assert {"d1_0.txt", "d2_0.txt", "d3_0.txt"} <= names and len(names) == 10  # judged + 7 distractors
    questions = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines()]
    assert sorted(tuple(q["relevant_ids"]) for q in questions) == [("d1_0.txt",), ("d3_0.txt",)]  # positives only
    assert all(q["relevant_ids"] for q in questions) and info["language"] == "Hindi" and info["queries"] == 2


def test_beir_subset_from_parquet(tmp_path, monkeypatch) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    public = _public_datasets()
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.table({"_id": ["a", "b", "c"], "title": ["A", "B", "C"], "text": ["alpha", "beta", "gamma"]}), raw / "c.parquet")
    pq.write_table(pa.table({"_id": ["1", "2"], "title": ["", ""], "text": ["what is alpha?", "no labels"]}), raw / "q.parquet")
    (raw / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n1\ta\t1\n", encoding="utf-8")
    files = {"corpus": raw / "c.parquet", "queries": raw / "q.parquet", "test.tsv": raw / "qrels.tsv"}
    monkeypatch.setattr(public, "download", lambda url, target: next(p for k, p in files.items() if k in url))
    docs_dir, golden, info = public.prepare_beir("scifact", tmp_path / "cache", max_queries=10, max_docs=2, seed=1)
    assert info["queries"] == 1 and len(list(docs_dir.glob("*.txt"))) == 2 and "a.txt" in golden.read_text(encoding="utf-8")


def test_a_snippet_straddling_a_chunk_boundary_counts_when_both_chunks_are_retrieved() -> None:
    text = "Intro words here. The launch window opens in March and closes in April. Outro words follow."
    cut = text.index("in March")  # the snippet is split across two chunks
    first = Document(id="c1", content=text[:cut], metadata={"parent_id": "d", "start_char": 0, "end_char": cut})
    second = Document(id="c2", content=text[cut:], metadata={"parent_id": "d", "start_char": cut, "end_char": len(text)})
    row = {"relevant_text": ["The launch window opens in March and closes in April."], "relevant_ids": []}
    assert real_eval.score([first, second], row, 2, {"d": text}) == (1.0, 1.0)  # covered together: a hit
    assert real_eval.score([first], row, 1, {"d": text})[0] == 0.0  # half of it: not found
    assert real_eval.score([first, second], row, 2)[0] == 0.0  # without offsets: the old containment rule


def test_counting_embedding_passes_the_model_tokenizer_and_limit_through() -> None:
    from raglib.embedding.mock_embedding import MockEmbedding

    class Model(MockEmbedding):
        @property
        def max_input_tokens(self):
            return 512

        def count_input_tokens(self, text):
            return 2 * len(text.split())

    counting = real_eval.CountingEmbedding(Model())
    assert counting.max_input_tokens == 512 and counting.count_input_tokens("a b c") == 6
