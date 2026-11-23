"""Per-stage latency benchmark: run test queries N times and print P50/P95/P99 per stage.

Offline by default (mock models, built-in sample corpus), so it runs anywhere::

    python benchmarks/latency/bench_latency.py
    python benchmarks/latency/bench_latency.py --source docs/ --queries questions.txt \\
        --llm openai/gpt-4.1-mini --embedding openai --preset production --repeats 5

``--queries`` is a text file with one question per line. By default the LLM and embedding
caches are cleared before every query, so repeats measure real work; ``--warm-cache`` measures
the cached path instead. ``--slowest STAGE`` prints the slowest samples of a stage with their tags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SAMPLE_CORPUS = [
    "Acme Corp offices close on Friday afternoons in summer and the cafeteria opens at eight every morning.",
    "The Orion launch window opens in March and closes in April; the launch team meets every Monday.",
    "Refunds are issued within fourteen days of a return when the receipt is included with the package.",
    "The security policy requires two-factor authentication for every employee account and contractor laptop.",
    "Quarterly revenue grew thirty-four percent, driven by enterprise contracts and lower operating costs.",
]
SAMPLE_QUERIES = [
    "When do Acme offices close?",
    "When does the Orion launch window open?",
    "How long do refunds take?",
    "What does the security policy require?",
    "What drove quarterly revenue growth?",
]


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="*", help="files, folders or URLs to index (default: a built-in sample corpus)")
    parser.add_argument("--queries", help="text file with one question per line (default: built-in sample questions)")
    parser.add_argument("--repeats", type=int, default=5, help="times each query is run (default 5)")
    parser.add_argument("--warmup", type=int, default=1, help="unrecorded warm-up runs (default 1)")
    parser.add_argument("--llm", help="provider or provider/model (default: offline mock)")
    parser.add_argument("--embedding", help="embedding provider (default: chosen from --llm, or mock offline)")
    parser.add_argument("--rag-type", default="advanced")
    parser.add_argument("--preset", help="configuration preset, e.g. production")
    parser.add_argument("--reranker", help="none | cross_encoder | cohere | llm")
    parser.add_argument("--warm-cache", action="store_true", help="keep caches between queries")
    parser.add_argument("--slowest", metavar="STAGE", help="also print the slowest samples of this stage")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    from raglib import RAG, benchmark

    args = _parse_args(argv)
    queries = (
        [line.strip() for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.queries
        else SAMPLE_QUERIES
    )
    options = {"rag_type": args.rag_type}
    if args.llm:
        options["llm"] = args.llm
    else:
        options["offline"] = True
    for key in ("embedding", "preset", "reranker"):
        value = getattr(args, key)
        if value:
            options["embedding_llm" if key == "embedding" else key] = value
    source = (args.source[0] if len(args.source) == 1 else args.source) if args.source else SAMPLE_CORPUS
    rag = RAG(source, **options)

    print(f"Running {len(queries)} queries x {args.repeats} ({'warm' if args.warm_cache else 'cold'} cache), "
          f"rag_type={args.rag_type}, llm={args.llm or 'offline mock'}\n")
    stats = benchmark(rag, queries, repeats=args.repeats, warmup=args.warmup, warm_cache=args.warm_cache)
    stats.print_report()
    if args.slowest:
        print(f"\nSlowest '{args.slowest}' samples:")
        for sample in stats.slowest(args.slowest, 5):
            print(f"  {sample.duration_ms:9.2f} ms  {sample.tags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
