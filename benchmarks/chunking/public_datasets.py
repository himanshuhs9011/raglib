"""Run the chunking comparison (real_eval.py) on public retrieval datasets with relevance labels.

No hand-made golden set and no paid API: datasets come with relevance judgments (qrels), and the
default embedding models run locally (pip install raglib-py[hf] pyarrow; a CUDA build of torch for GPU).

    python benchmarks/chunking/public_datasets.py                       # all four datasets
    python benchmarks/chunking/public_datasets.py --datasets scifact fiqa --max-queries 50

Datasets (downloaded once into --cache, default ~/.cache/raglib/bench):

| name      | domain                    | language | source (Hugging Face)                    | queries / judged docs |
| scifact   | scientific claims         | English  | BeIR/scifact + BeIR/scifact-qrels (test) | 300 / 5k corpus       |
| nfcorpus  | nutrition / medicine      | English  | BeIR/nfcorpus + qrels (test)             | 323 / 3.6k corpus     |
| fiqa      | financial opinion Q&A     | English  | BeIR/fiqa + qrels (test)                 | 648 / 57k corpus      |
| miracl-hi | Wikipedia                 | Hindi    | miracl/miracl + miracl/miracl-corpus (dev) | 350 / 506k passages |

Each run uses a subset: --max-queries questions (default 100) and a corpus of every document judged
for them plus random distractors, up to --max-docs (default 2000). MIRACL's judged negatives are kept,
so the Hindi subset is hard, not padded with easy noise. Embedding: --embedding (default BAAI/bge-base-en-v1.5)
for the English sets and --multilingual-embedding (default BAAI/bge-m3) for MIRACL, because an English-only
model cannot rank Hindi text.

Note: BEIR documents and MIRACL passages are short (often below 300 tokens), so with the default
--chunk-size 500 most of them stay one chunk and the configs differ little; add --chunk-size 128 to see
how the chunkers split. Results: one JSON per dataset in --out plus a combined summary table.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
USER_AGENT = "raglib-benchmark/1.0"

BEIR = {
    "scifact": ("BeIR/scifact", "BeIR/scifact-qrels", "science"),
    "nfcorpus": ("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", "nutrition / medicine"),
    "fiqa": ("BeIR/fiqa", "BeIR/fiqa-qrels", "finance"),
}
MIRACL = {"miracl-hi": "hi"}
ALL = [*BEIR, *MIRACL]


# ------------------------------------------------------------------ downloads


def download(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` once (atomic: a partial download is never mistaken for a file)."""

    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"  downloading {url}")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:  # noqa: S310 - fixed public URLs
        total, done, next_report = int(response.headers.get("Content-Length") or 0), 0, 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            done += len(block)
            if total and done >= next_report:
                print(f"    {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                next_report += max(total // 5, 1 << 20)
    partial.replace(target)
    return target


def _read_parquet(path: Path, columns: List[str]) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit("BEIR datasets are Parquet files: pip install pyarrow") from None
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def safe_name(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", doc_id) + ".txt"


# ------------------------------------------------------------------ dataset preparation


def _write(folder: Path, docs: Dict[str, Tuple[str, str]], questions: List[Tuple[str, Set[str]]]) -> Tuple[Path, Path]:
    docs_dir = folder / "docs"
    if docs_dir.exists():
        for old in docs_dir.glob("*.txt"):
            old.unlink()
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, (title, text) in docs.items():
        body = f"{title}\n\n{text}" if title else text
        (docs_dir / safe_name(doc_id)).write_text(body, encoding="utf-8")
    golden = folder / "golden.jsonl"
    with golden.open("w", encoding="utf-8") as handle:
        for question, relevant in questions:
            ids = [safe_name(doc_id) for doc_id in sorted(relevant) if doc_id in docs]
            if ids:
                handle.write(json.dumps({"question": question, "relevant_ids": ids}, ensure_ascii=False) + "\n")
    return docs_dir, golden


def prepare_beir(name: str, cache: Path, max_queries: int, max_docs: int, seed: int) -> Tuple[Path, Path, Dict[str, Any]]:
    repo, qrels_repo, domain = BEIR[name]
    raw = cache / name / "raw"
    corpus_file = download(HF.format(repo=repo, path="corpus/corpus-00000-of-00001.parquet"), raw / "corpus.parquet")
    queries_file = download(HF.format(repo=repo, path="queries/queries-00000-of-00001.parquet"), raw / "queries.parquet")
    qrels_file = download(HF.format(repo=qrels_repo, path="test.tsv"), raw / "qrels-test.tsv")

    relevant: Dict[str, Set[str]] = {}
    for line in qrels_file.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and float(parts[2]) > 0:
            relevant.setdefault(parts[0], set()).add(parts[1])
    queries = {row["_id"]: row["text"] for row in _read_parquet(queries_file, ["_id", "text"]) if row["_id"] in relevant}
    rng = random.Random(seed)
    chosen = sorted(queries)
    chosen = rng.sample(chosen, min(max_queries, len(chosen)))
    needed = set().union(*(relevant[qid] for qid in chosen))
    corpus = {row["_id"]: (row.get("title") or "", row.get("text") or "") for row in _read_parquet(corpus_file, ["_id", "title", "text"])}
    others = sorted(set(corpus) - needed)
    extra = rng.sample(others, max(0, min(len(others), max_docs - len(needed))))
    docs = {doc_id: corpus[doc_id] for doc_id in sorted(needed | set(extra)) if doc_id in corpus}
    docs_dir, golden = _write(cache / name, docs, [(queries[qid], relevant[qid]) for qid in chosen])
    return docs_dir, golden, {"dataset": name, "domain": domain, "language": "English", "queries": len(chosen),
                              "docs": len(docs), "relevant_docs": len(needed)}


def _miracl_corpus(paths: List[Path]) -> Iterator[Dict[str, str]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def prepare_miracl(name: str, cache: Path, max_queries: int, max_docs: int, seed: int) -> Tuple[Path, Path, Dict[str, Any]]:
    lang = MIRACL[name]
    raw = cache / name / "raw"
    base = f"miracl-v1.0-{lang}"
    topics = download(HF.format(repo="miracl/miracl", path=f"{base}/topics/topics.{base}-dev.tsv"), raw / "topics-dev.tsv")
    qrels = download(HF.format(repo="miracl/miracl", path=f"{base}/qrels/qrels.{base}-dev.tsv"), raw / "qrels-dev.tsv")
    shards = [download(HF.format(repo="miracl/miracl-corpus", path=f"miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz"),
                       raw / f"docs-{i}.jsonl.gz") for i in (0, 1)]

    relevant: Dict[str, Set[str]] = {}
    judged: Dict[str, Set[str]] = {}
    for line in qrels.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            judged.setdefault(parts[0], set()).add(parts[2])
            if int(parts[3]) > 0:
                relevant.setdefault(parts[0], set()).add(parts[2])
    queries = {}
    for line in topics.read_text(encoding="utf-8").splitlines():
        qid, _, text = line.partition("\t")
        if qid in relevant:
            queries[qid] = text
    rng = random.Random(seed)
    chosen = rng.sample(sorted(queries), min(max_queries, len(queries)))
    needed = set().union(*(judged[qid] for qid in chosen))  # positives + judged negatives
    budget = max(0, max_docs - len(needed))
    docs: Dict[str, Tuple[str, str]] = {}
    reservoir: List[Dict[str, str]] = []
    for seen, row in enumerate(_miracl_corpus(shards)):
        if row["docid"] in needed:
            docs[row["docid"]] = (row.get("title") or "", row.get("text") or "")
        elif budget:  # reservoir sampling: uniform distractors in one pass
            if len(reservoir) < budget:
                reservoir.append(row)
            else:
                slot = rng.randint(0, seen)
                if slot < budget:
                    reservoir[slot] = row
    for row in reservoir:
        docs.setdefault(row["docid"], (row.get("title") or "", row.get("text") or ""))
    docs_dir, golden = _write(cache / name, docs, [(queries[qid], relevant[qid]) for qid in chosen])
    return docs_dir, golden, {"dataset": name, "domain": "Wikipedia", "language": "Hindi", "queries": len(chosen),
                              "docs": len(docs), "relevant_docs": len(set().union(*(relevant[q] for q in chosen)))}


# ------------------------------------------------------------------ run


def _real_eval():
    spec = importlib.util.spec_from_file_location("real_eval", HERE / "real_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["real_eval"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "raglib" / "bench"))
    parser.add_argument("--out", default="bench-results", help="folder for the per-dataset JSON results")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--embedding", default="huggingface/BAAI/bge-base-en-v1.5", help="for the English datasets")
    parser.add_argument("--multilingual-embedding", default="huggingface/BAAI/bge-m3", help="for MIRACL")
    parser.add_argument("--prepare-only", action="store_true", help="download and build the subsets, do not embed")
    args, passthrough = parser.parse_known_args(argv)  # e.g. --chunk-size 128 --k 10 go to real_eval.py

    cache, out = Path(args.cache).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real_eval = _real_eval()
    summary = []
    for name in args.datasets:
        print(f"\n=== {name}")
        prepare = prepare_beir if name in BEIR else prepare_miracl
        docs_dir, golden, info = prepare(name, cache, args.max_queries, args.max_docs, args.seed)
        embedding = args.multilingual_embedding if name in MIRACL else args.embedding
        info["embedding"] = embedding
        print(f"  {info['queries']} questions, {info['docs']} documents ({info['relevant_docs']} relevant), embedding {embedding}")
        if args.prepare_only:
            summary.append(info)
            continue
        result_file = out / f"{name}.json"
        code = real_eval.main(["--docs", str(docs_dir), "--golden", str(golden), "--embedding", embedding,
                               "--json", str(result_file), *passthrough])
        if code != 0:
            return code
        rows = json.loads(result_file.read_text(encoding="utf-8"))
        info["results"] = rows
        summary.append(info)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.prepare_only:
        print("\n| dataset | domain | language | embedding | baseline recall | best config | best recall | its chunks |")
        print("| :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: |")
        for info in summary:
            rows = info["results"]
            baseline, best = rows[0], max(rows, key=lambda row: row["recall"])
            print(f"| {info['dataset']} | {info['domain']} | {info['language']} | {info['embedding'].split('/', 1)[-1]} | "
                  f"{baseline['recall']:.3f} | {best['config']} | {best['recall']:.3f} | {best['chunks']} |")
        print(f"\nPer-dataset tables above; JSON in {out}/ (send summary.json back for the docs update).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
