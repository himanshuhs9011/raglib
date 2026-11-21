"""Benchmark raglib's RAG types on MultiHop-RAG (Tang & Yang, 2024) and score them with Open RAG Eval.

Every configuration gets the same questions, corpus, chunking, embedding model, generator model and
judge, so differences come from the pipeline alone.

Steps (each is resumable; outputs go to ``--work``, default ``benchmarks/multihop_rag/work``)::

    python benchmarks/multihop_rag/bench.py download          # dataset from Hugging Face (ODC-BY)
    python benchmarks/multihop_rag/bench.py sample --n 30     # stratified by question type, seed 42
    python benchmarks/multihop_rag/bench.py index             # chunk + embed the 609 articles once
    python benchmarks/multihop_rag/bench.py generate          # answer every question with every config
    python benchmarks/multihop_rag/bench.py judge             # LLM-judged correctness vs the gold answer
    python benchmarks/multihop_rag/bench.py export            # Open RAG Eval CSVs + judge configs
    open-rag-eval eval --config work/ore/<config>.yaml        # LLM-judged metrics (see README.md)
    python benchmarks/multihop_rag/bench.py summarize         # deterministic metrics + judge metrics -> JSON

Needs ``NVIDIA_API_KEY`` (generator, embeddings and judge all run on build.nvidia.com).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
# RAGLIB_SRC runs another copy of the library (e.g. a pre-fix snapshot) with this harness.
sys.path.insert(0, os.environ.get("RAGLIB_SRC") or str(HERE.parents[1]))  # run from a checkout without installing

DATASET_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/{name}"
QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")

GENERATOR = "nvidia/nemotron-3.5-lightning-30b-a3b"
EMBEDDING = "nvidia/nemotron-3-embed-1b"
JUDGE = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
REQUESTS_PER_MINUTE = 35  # stays under build.nvidia.com's free-tier limit
# GraphRAG's ~3,600 extraction calls are too slow on NVIDIA's free tier (about 2.5 successful calls/min),
# so the graph is built with a cheap paid model through an OpenAI-compatible gateway. Answers and judging
# stay on NVIDIA like every other config.
GRAPH_MODEL = "inclusionai/ling-3.0-flash"
GRAPH_LLM = f"velona/{GRAPH_MODEL}"
VELONA_BASE_URL = "https://velona.in/v1"
GRAPH_INDEX = "graph_index_ling"

# name -> (rag_type, extra RAG(...) settings, one-line description)
CONFIGS: Dict[str, Dict[str, Any]] = {
    "naive": {"rag_type": "naive", "kwargs": {"retrieval_mode": "dense"},
              "about": "Dense vector search -> answer (textbook baseline)"},
    "advanced": {"rag_type": "advanced", "kwargs": {},
                 "about": "Library default: BM25 + dense hybrid search -> dedupe -> cited answer"},
    "advanced_rerank": {"rag_type": "advanced",
                        "kwargs": {"reranker": "cross_encoder", "reranker_model": "BAAI/bge-reranker-base", "fetch_k": 20},
                        "about": "Default + cross-encoder reranking of 20 candidates (bge-reranker-base, local)"},
    "multi_query": {"rag_type": "multi_query", "kwargs": {},
                    "about": "LLM paraphrases the question, retrieves for each, fuses with RRF"},
    "multi_hop": {"rag_type": "multi_hop", "kwargs": {},
                  "about": "Retrieve, ask what is still missing, retrieve again (bridge entities)"},
    "corrective": {"rag_type": "corrective", "kwargs": {},
                   "about": "CRAG: grade retrieved chunks, keep relevant strips (no web fallback in this run)"},
    "self": {"rag_type": "self", "kwargs": {},
             "about": "Self-RAG: relevance, support and usefulness checks with revision"},
    "agentic": {"rag_type": "agentic", "kwargs": {},
                "about": "ReAct agent that searches the documents until it has enough evidence"},
    "graph": {"rag_type": "graph", "kwargs": {"graph_llm": GRAPH_LLM}, "index": GRAPH_INDEX,
              "about": f"GraphRAG: LLM-extracted knowledge graph + communities, local/global search (graph built by {GRAPH_MODEL})"},
    "graph_context": {"rag_type": "graph_context", "kwargs": {},
                      "about": "GraphContextWindow retrieval (document/section/chunk graph, keyword + embedding seeds, 2 hops)"},
}
EXCLUDED = {
    "hybrid": "same retrieval as advanced (BM25 + dense) when an embedding model is configured",
    "web / hybrid_web": "live web search is not reproducible and can leak answers from the internet",
    "routing": "routes between corpus, web and graph; with web and graph off it reduces to advanced",
    "memory": "built for multi-turn chat; single-turn questions make it equal to advanced",
    "tool": "needs application-specific tools",
}

ABSTAIN_RE = re.compile(
    r"\b(i don'?t know|do not know|insufficient information|not enough information|"
    r"(?:does|do) not (?:contain|provide|mention|include|state)|no information|not mentioned|"
    r"cannot (?:be )?(?:determine|answer|find)|unable to (?:determine|answer|find))",
    re.IGNORECASE,
)

log = logging.getLogger("bench")


# ----------------------------------------------------------------------------- helpers


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"\[\d+\]", " ", text)  # drop citation markers
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def article_text(article: Dict[str, Any]) -> str:
    """The article as indexed: metadata header (questions refer to source and date) + body."""

    header = (
        f"Title: {article['title']}\nSource: {article['source']}\n"
        f"Published: {article['published_at'][:10]}\nAuthor: {article.get('author') or 'unknown'}\n\n"
    )
    return header + article["body"]


def is_correct(question: Dict[str, Any], answer: str) -> bool:
    """MultiHop-RAG accuracy: abstain on null queries, else the gold answer must appear in the response.

    Yes/No answers are compared on the first yes/no word of the response, so "No, ... yes" is not
    counted as "Yes".
    """

    gold = question["answer"].strip()
    if question["question_type"] == "null_query":
        return bool(ABSTAIN_RE.search(answer))
    if ABSTAIN_RE.search(answer) and len(answer) < 200:
        return False
    norm_answer = normalize(answer)
    norm_gold = normalize(gold)
    if norm_gold in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", norm_answer)
        return bool(match) and match.group(1) == norm_gold
    return bool(norm_gold) and norm_gold in norm_answer


def evidence_recall(question: Dict[str, Any], sources: List[Dict[str, Any]]) -> Optional[float]:
    """Share of the gold evidence articles that appear among the sources given to the model."""

    gold_titles = {normalize(item["title"]) for item in question["evidence_list"]}
    if not gold_titles:
        return None
    seen = {normalize(source.get("title") or "") for source in sources}
    return len(gold_titles & seen) / len(gold_titles)


# ----------------------------------------------------------------------------- steps


def cmd_download(args: argparse.Namespace) -> None:
    data = args.work / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("corpus.json", "MultiHopRAG.json"):
        target = data / name
        if target.exists():
            print(f"{target} exists")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(DATASET_URL.format(name=name), target)  # noqa: S310 - fixed HF URL
    print("MultiHop-RAG (Tang & Yang, 2024), license ODC-BY: https://huggingface.co/datasets/yixuantt/MultiHopRAG")


def cmd_sample(args: argparse.Namespace) -> None:
    questions = read_json(args.work / "data" / "MultiHopRAG.json")
    rng = random.Random(args.seed)
    excluded = {q["dataset_index"] for q in read_json(args.exclude)} if args.exclude else set()
    per_type = {
        kind: [i for i, q in enumerate(questions) if q["question_type"] == kind and i not in excluded]
        for kind in QUESTION_TYPES
    }
    base, extra = divmod(args.n, len(QUESTION_TYPES))
    picked: List[int] = []
    for position, kind in enumerate(QUESTION_TYPES):
        picked += rng.sample(per_type[kind], base + (1 if position < extra else 0))
    sample = [{"id": f"q{index}", "dataset_index": index, **questions[index]} for index in sorted(picked)]
    write_json(args.work / "questions.json", sample)
    counts = {kind: sum(q["question_type"] == kind for q in sample) for kind in QUESTION_TYPES}
    print(f"sampled {len(sample)} questions (seed {args.seed}): {counts}")


def register_fast_model() -> None:
    """The generator with thinking off: a cheap, fast model for the thousands of graph-indexing calls."""

    from raglib import register_provider
    from raglib.llm.providers import PROVIDERS

    if "velona" not in PROVIDERS:
        # json_mode off: Ling loops until max_tokens in JSON mode, but returns clean JSON without it
        register_provider(
            "velona", name="Velona gateway", base_url=VELONA_BASE_URL, api_key="{env:VELONA_API_KEY}",
            models=[{"id": GRAPH_MODEL, "json_mode": False}, {"id": "anthropic/claude-haiku-5.5"},
                    {"id": "openai/gpt-6-luna"}, {"id": "deepseek/deepseek-v4-flash"}],
        )
    if "nvidia-fast" not in PROVIDERS:
        register_provider(
            "nvidia-fast", name="NVIDIA NIM, thinking off", base_url=NVIDIA_BASE_URL, api_key="{env:NVIDIA_API_KEY}",
            models=[GENERATOR], stream_usage=True, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )


def _rag(rag_type: str, **kwargs: Any):
    from raglib import RAG

    register_fast_model()

    settings: Dict[str, Any] = {
        "llm": f"nvidia/{GENERATOR}",
        "embedding_llm": f"nvidia/{EMBEDDING}",
        "llm_rpm": REQUESTS_PER_MINUTE,
        "llm_timeout": 120,
        "llm_max_retries": 4,
        "llm_cache": False,
    }
    settings.update(kwargs)  # a configuration may override the shared defaults (e.g. a lower llm_rpm)
    return RAG(rag_type=rag_type, **settings)


def corpus_documents(work: Path) -> list:
    """The 609 articles as raglib Documents (identical for every configuration)."""

    from raglib import Document

    corpus = read_json(work / "data" / "corpus.json")
    return [
        Document(
            id=f"article-{i}",
            content=article_text(article),
            metadata={"title": article["title"], "source_name": article["source"],
                      "published_at": article["published_at"], "url": article["url"]},
        )
        for i, article in enumerate(corpus)
    ]


class GraphContextAnswerer:
    """GraphContextWindow as the retriever, raglib's standard Generator for the answer.

    Same corpus, chunk size (400 words, 50 overlap), embeddings, generator, prompt and top_k=5
    as the other configurations; only retrieval differs.
    """

    def __init__(self, work: Path):
        from raglib import GraphContextWindow
        from raglib.components.generator import Generator

        rag = _rag("advanced")  # provides the shared LLM client and embedding model
        self.graph = GraphContextWindow(
            chunking_strategy="fixed", chunk_size=400, chunk_overlap=50, max_context_tokens=3000,
            embedding=rag._embedding,
        )
        started = time.time()
        self.graph.add_documents(corpus_documents(work))
        print(f"[graph_context] graph built in {time.time() - started:.0f}s", flush=True)
        self.generator = Generator(llm_client=rag._llm)

    def query(self, question: str):
        from raglib.core.runtime import usage_scope

        with usage_scope() as meter:
            documents = self.graph.retrieve(question, top_k=5)
            result = self.generator.generate(query=question, documents=documents)
        result.usage = {**result.usage, **meter.as_dict()}
        return result


def cmd_graph_index(args: argparse.Namespace) -> None:
    """Build the GraphRAG knowledge graph over the corpus, in batches, saving after each (resumable)."""

    from raglib.core.runtime import usage_scope

    out = args.work / GRAPH_INDEX
    stats_path = args.work / f"{GRAPH_INDEX}_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {"seconds": 0.0, "llm_calls": 0, "tokens": 0, "articles_done": 0}
    documents = corpus_documents(args.work)
    if stats["articles_done"] >= len(documents) and stats.get("communities") is not None:
        print(f"graph index complete: {stats}")
        return
    # A 300s timeout: a call that times out client-side is still billed by the gateway, so retrying it pays
    # twice. 4000 tokens caps the cost of a reply that loops (normal replies are 2,300-3,900 tokens).
    rag = _rag("graph", graph_llm=GRAPH_LLM, llm_rpm=args.rpm, llm_timeout=300,
               graph_kwargs={"max_workers": args.workers, "extraction_max_tokens": 4000,
                             # some chunks always come back empty (the model loops to max_tokens): give up
                             # after two paid attempts instead of paying again on every batch and load
                             "max_extraction_attempts": 2})
    if out.exists() and any(out.iterdir()):
        rag.load(str(out))
    for start in range(stats["articles_done"], len(documents), args.batch):
        if not _credit_ok(args.min_balance):
            return
        batch = documents[start : start + args.batch]
        began = time.time()
        with usage_scope() as meter:
            rag.add(batch)
            for _ in range(1):  # failed extractions are retried once (a looping reply loops again), never stored as empty
                if not rag.stats()["graph"]["failed_extractions"]:
                    break
                time.sleep(20)
                rag.retry_graph_extraction()
        rag.save(str(out))
        stats["failed_extractions"] = rag.stats()["graph"]["failed_extractions"]
        stats["seconds"] += time.time() - began
        stats["llm_calls"] += meter.llm_calls
        stats["tokens"] += meter.prompt_tokens + meter.completion_tokens
        stats["articles_done"] = start + len(batch)
        write_json(stats_path, stats)
        graph = rag.stats().get("graph", {})
        print(f"[graph-index] {stats['articles_done']}/{len(documents)} articles, {stats['llm_calls']} LLM calls, "
              f"{stats['seconds'] / 60:.0f} min, graph={graph}", flush=True)
    if not _credit_ok(args.min_balance + 4):  # community summaries cost about Rs 4
        return
    began = time.time()
    with usage_scope() as meter:
        communities = rag.rebuild_graph_communities()
    rag.save(str(out))
    stats["seconds"] += time.time() - began
    stats["llm_calls"] += meter.llm_calls
    stats["tokens"] += meter.prompt_tokens + meter.completion_tokens
    stats["communities"] = len(communities) if hasattr(communities, "__len__") else communities
    stats["graph"] = rag.stats().get("graph", {})
    write_json(stats_path, stats)
    print(f"graph index complete: {stats}")


def _credit_ok(min_balance: float) -> bool:
    """Spending stop: True while the gateway balance is above ``min_balance`` rupees."""

    request = urllib.request.Request(
        "https://velona.in/gateway/v1/account/credits",
        headers={"Authorization": f"Bearer {os.environ['VELONA_API_KEY']}", "User-Agent": "raglib-bench"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            balance = float(json.load(response)["data"]["balance_inr"])
    except Exception as exc:  # noqa: BLE001 - unknown balance: stop rather than overspend
        print(f"[graph-index] could not read the balance ({exc}); stopping", flush=True)
        return False
    print(f"[graph-index] balance Rs {balance:.2f}", flush=True)
    if balance < min_balance:
        print(f"[graph-index] STOP: balance Rs {balance:.2f} < Rs {min_balance:.2f}; progress is saved, "
              "top up and rerun graph-index to resume", flush=True)
        return False
    return True


def cmd_index(args: argparse.Namespace) -> None:
    index_dir = args.work / "index"
    if (index_dir / "manifest.json").exists() or any(index_dir.glob("*")):
        print(f"index exists at {index_dir}")
        return
    documents = corpus_documents(args.work)
    started = time.time()
    rag = _rag("advanced")
    chunks = rag.add(documents)
    rag.save(str(index_dir))
    print(f"indexed {len(documents)} articles into {chunks} chunks in {time.time() - started:.0f}s -> {index_dir}")


def _answer(rag, question: Dict[str, Any]) -> Dict[str, Any]:
    from raglib.core.runtime import usage_scope  # noqa: F401 - usage is reported on the result

    started = time.time()
    try:
        result = rag.query(question["query"])
    except Exception as exc:  # noqa: BLE001 - recorded as a failed answer
        return {"id": question["id"], "error": f"{type(exc).__name__}: {exc}"[:500], "latency_s": time.time() - started}
    sources = [
        {"n": n, "doc_id": doc.id, "title": doc.metadata.get("title") or doc.metadata.get("doc_title"),
         "text": doc.content}
        for n, doc in enumerate(result.sources, start=1)
    ]
    return {
        "id": question["id"],
        "answer": result.answer,
        "sources": sources,
        "citations": [c.index for c in result.citations],
        "usage": result.usage,
        "latency_s": round(time.time() - started, 2),
        "degraded": result.metadata.get("degraded") or [],
    }


def cmd_generate(args: argparse.Namespace) -> None:
    questions = read_json(args.work / "questions.json")[: args.limit or None]
    names = args.configs or list(CONFIGS)
    out_dir = args.work / "answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def run_config(name: str) -> str:
        config = CONFIGS[name]
        path = out_dir / f"{args.save_as or name}.jsonl"
        done = {row["id"] for row in read_jsonl(path) if "error" not in row}
        todo = [q for q in questions if q["id"] not in done]
        if not todo:
            return f"{name}: complete"
        if config["rag_type"] == "graph_context":
            rag = GraphContextAnswerer(args.work)
        else:
            index = args.work / config["index"] if config.get("index") else (args.index or args.work / "index")
            overrides = {"llm": args.llm} if args.llm else {}  # paid answer model; retrieval and judge unchanged
            rag = _rag(config["rag_type"], **{**config["kwargs"], **overrides}).load(str(index))
        rows = [row for row in read_jsonl(path) if row["id"] in done]  # drop earlier failures, retry them
        for question in todo:
            row = _answer(rag, question)
            rows.append(row)
            with lock:
                path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                status = "ERROR " + row["error"][:80] if "error" in row else f"{row['latency_s']:.1f}s"
                print(f"[{name}] {question['id']} {status}", flush=True)
        return f"{name}: {len(rows)} answers"

    # Configs run in parallel; they share one rate limiter for the (provider, key) pair.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(run_config, names):
            print(message, flush=True)


def cmd_export(args: argparse.Namespace) -> None:
    questions = {q["id"]: q for q in read_json(args.work / "questions.json")}
    ore_dir = args.work / "ore"
    ore_dir.mkdir(parents=True, exist_ok=True)
    for name in _answer_names(args.work):
        rows = [row for row in read_jsonl(args.work / "answers" / f"{name}.jsonl") if "error" not in row]
        if not rows:
            continue
        results_dir = ore_dir / "results" / name
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "answers.csv"  # Open RAG Eval reads <results_folder>/<generated_answers>
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["query_id", "query", "query_run", "passage_id", "passage", "generated_answer"])
            for row in rows:
                query = questions[row["id"]]["query"]
                sources = row["sources"] or [{"n": 1, "text": ""}]
                for position, source in enumerate(sources):
                    writer.writerow([
                        row["id"], query, 1, f"[{source['n']}]", source["text"],
                        row["answer"] if position == 0 else "",
                    ])
        (ore_dir / f"{name}.yaml").write_text(
            "\n".join([
                f"results_folder: {results_dir.as_posix()}",
                f"generated_answers: {csv_path.name}",
                "eval_results_file: eval_results.csv",
                "metrics_file: metrics.png",
                "evaluator:",
                "  type: TRECEvaluator",
                "  model:",
                "    type: OpenAIModel",
                f"    name: {JUDGE}",
                f"    base_url: {NVIDIA_BASE_URL}",
                "    api_key: ${oc.env:NVIDIA_API_KEY}",
                "  options:",
                "    k_values: [1, 3, 5]",
                "    run_consistency: false",
                "",
            ]),
            encoding="utf-8",
        )
        print(f"{name}: {len(rows)} answers -> {csv_path.name}")


CORRECTNESS_PROMPT = """You grade a question-answering system against a reference answer.

Question: {question}
Reference answer: {gold}
System response: {response}

Rules:
- If the reference answer is "Insufficient information", the response is correct only if it says the
  documents do not contain the answer (declining), and wrong if it gives a substantive answer.
- Otherwise the response is correct if its final answer means the same as the reference (aliases,
  short forms and paraphrases count; "does not contradict" answers "no" to "does it contradict").
- A response that declines, hedges between options, or gives a different answer is wrong.

Return JSON only: {{"correct": true or false, "reason": "<one sentence>"}}"""


def cmd_judge(args: argparse.Namespace) -> None:
    """LLM-judged correctness against the gold answer (one call per answer, judge model, temperature 0)."""

    from raglib.llm.clients import build_chat_client

    questions = {q["id"]: q for q in read_json(args.work / "questions.json")}
    from raglib.llm.resilience import RateLimiter

    judge = build_chat_client("nvidia", model_name=JUDGE, max_retries=6, timeout=120)
    judge.rate_limiter = RateLimiter(args.rpm)  # the judge shares the account limit with everything else
    path = args.work / "correctness.jsonl"
    done = {(row["config"], row["id"]) for row in read_jsonl(path) if row.get("correct") is not None}
    todo = []
    for name in _answer_names(args.work):
        for row in read_jsonl(args.work / "answers" / f"{name}.jsonl"):
            if "error" not in row and (name, row["id"]) not in done:
                todo.append((name, row))
    lock = threading.Lock()

    def grade(item):
        name, row = item
        question = questions[row["id"]]
        prompt = CORRECTNESS_PROMPT.format(question=question["query"], gold=question["answer"], response=row["answer"])
        try:
            verdict = judge.complete_json(prompt, default={})
        except Exception as exc:  # noqa: BLE001 - recorded, retried on the next run
            verdict = {"reason": f"judge error: {exc}"[:300]}
        verdict = verdict if isinstance(verdict, dict) else {}
        if not verdict:
            verdict = {"reason": "judge call failed (rate limit or unparsable reply); retried on the next run"}
        correct = verdict.get("correct")
        record = {"config": name, "id": row["id"], "correct": correct if isinstance(correct, bool) else None,
                  "reason": verdict.get("reason")}
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[judge] {name} {row['id']} -> {record['correct']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(grade, todo))
    print(f"judged {len(todo)} answers -> {path}")


def _ore_scores(path: Path) -> Dict[str, Dict[str, float]]:
    """Per-question Open RAG Eval scores from its eval_results.csv (column names vary by version)."""

    if not path.exists():
        return {}
    # 0.3.0 writes "run_1_<metric>" columns; nested scores are JSON strings.
    plain = {
        "umbrela": "retrieval_score_mean_umbrela_score",
        "nugget_vital": "generation_score_vital_nuggetizer_score",
        "hallucination": "generation_score_hallucination_score",
        "citation_f1": "generation_score_citation_f1_score",
    }
    nested = {
        "nugget_all": ("generation_score_autonugget_scores", ("nuggetizer_scores", "All")),
        "citation_precision": ("generation_score_citation_scores", ("weighted_precision",)),
    }
    scores: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {re.sub(r"^run_\d+_", "", key): value for key, value in raw.items()}
            entry: Dict[str, float] = {}
            for key, column in plain.items():
                try:
                    entry[key] = float(row[column])
                except (KeyError, TypeError, ValueError):
                    pass
            for key, (column, path_keys) in nested.items():
                try:
                    value: Any = json.loads(row[column])
                    for part in path_keys:
                        value = value[part]
                    entry[key] = float(value)
                except (KeyError, TypeError, ValueError):
                    pass
            if entry:  # a failed judge call leaves empty columns: not counted as zero
                scores[raw.get("query_id", "")] = entry
    return scores


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def _answer_names(work: Path) -> List[str]:
    """Configs with answers, in CONFIGS order; "<config>__<label>" variants follow their base config."""

    present = {path.stem for path in (work / "answers").glob("*.jsonl")}
    ordered = []
    for base in CONFIGS:
        ordered += sorted(name for name in present if name == base or name.startswith(base + "__"))
    return ordered


def cmd_summarize(args: argparse.Namespace) -> None:
    questions = {q["id"]: q for q in read_json(args.work / "questions.json")}
    verdicts: Dict[tuple, Optional[bool]] = {}
    reasons: Dict[tuple, str] = {}
    for verdict in read_jsonl(args.work / "correctness.jsonl"):
        if verdict.get("correct") is not None:
            verdicts[(verdict["config"], verdict["id"])] = verdict["correct"]
            reasons[(verdict["config"], verdict["id"])] = verdict.get("reason") or ""
    configs = []
    per_question = []
    for name in _answer_names(args.work):
        config = CONFIGS[name.split("__")[0]]
        rows = read_jsonl(args.work / "answers" / f"{name}.jsonl")
        if not rows:
            continue
        judged = _ore_scores(args.work / "ore" / "results" / name / "TRECEvaluator-eval_results.csv") or _ore_scores(
            args.work / "ore" / "results" / name / "eval_results.csv"
        )
        by_type: Dict[str, List[Optional[bool]]] = {kind: [] for kind in QUESTION_TYPES}
        strict: List[bool] = []
        recalls, latencies, calls, tokens, errors = [], [], [], [], 0
        for row in rows:
            question = questions[row["id"]]
            if "error" in row:
                errors += 1
                by_type[question["question_type"]].append(False)
                strict.append(False)
                per_question.append({"config": name, "id": row["id"], "correct": False, "error": row["error"]})
                continue
            correct = verdicts.get((name, row["id"]))
            strict_correct = is_correct(question, row["answer"])
            recall = evidence_recall(question, row["sources"])
            by_type[question["question_type"]].append(correct)
            strict.append(strict_correct)
            recalls.append(recall)
            latencies.append(row["latency_s"])
            calls.append(row["usage"].get("llm_calls"))
            tokens.append(row["usage"].get("total_tokens"))
            per_question.append({
                "config": name, "id": row["id"], "type": question["question_type"], "correct": correct,
                "strict_correct": strict_correct, "evidence_recall": recall, "answer": row["answer"],
                "gold": question["answer"], "judge": judged.get(row["id"], {}),
                "query": question["query"], "reason": reasons.get((name, row["id"]), ""),
                "sources": [source.get("title") for source in row["sources"]],
                "evidence": [item["title"] for item in question["evidence_list"]],
                "llm_calls": row["usage"].get("llm_calls"), "latency_s": row["latency_s"],
            })
        all_correct = [c for values in by_type.values() for c in values]
        judge_keys = sorted({key for scores in judged.values() for key in scores})
        configs.append({
            "name": name,
            "rag_type": config["rag_type"],
            "settings": config["kwargs"],
            "about": config["about"],
            "questions": len(rows),
            "errors": errors,
            "accuracy": _mean(None if c is None else float(c) for c in all_correct),
            "accuracy_judged_n": sum(c is not None for c in all_correct),
            "accuracy_by_type": {
                kind: _mean(None if c is None else float(c) for c in values) for kind, values in by_type.items()
            },
            "counts_by_type": {
                kind: [sum(c is True for c in values), sum(c is not None for c in values)]
                for kind, values in by_type.items()
            },
            "strict_accuracy": _mean(float(c) for c in strict),
            "evidence_recall": _mean(recalls),
            "latency_s_median": round(statistics.median(latencies), 2) if latencies else None,
            "llm_calls_mean": _mean(calls),
            "tokens_mean": _mean(tokens),
            "judge": {key: _mean(scores.get(key) for scores in judged.values()) for key in judge_keys},
        })
    summary = {
        "benchmark": "MultiHop-RAG",
        "dataset": {
            "name": "MultiHop-RAG", "authors": "Yixuan Tang and Yi Yang", "year": 2024,
            "paper": "https://arxiv.org/abs/2401.15391", "source": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
            "license": "ODC-BY", "corpus_articles": 609,
        },
        "sample": {"questions": len(questions), "seed": args.seed,
                   "by_type": {k: sum(q["question_type"] == k for q in questions.values()) for k in QUESTION_TYPES}},
        "models": {"generator": GENERATOR, "embedding": EMBEDDING, "judge": JUDGE, "endpoint": NVIDIA_BASE_URL},
        "correctness_prompt": CORRECTNESS_PROMPT,
        "excluded": EXCLUDED,
        "configs": configs,
        "generated_at": time.strftime("%Y-%m-%d"),
    }
    write_json(args.work / "summary.json", summary)
    write_json(args.work / "per_question.json", per_question)
    for config in configs:
        judge = ", ".join(f"{k}={v}" for k, v in config["judge"].items())
        print(f"{config['name']:<16} acc={config['accuracy']} strict={config['strict_accuracy']} recall={config['evidence_recall']} "
              f"lat={config['latency_s_median']}s calls={config['llm_calls_mean']} {judge}")


# Measured on 2026-11-21 (see the README); things the run cannot read back from its own outputs.
MEASURED = {
    "graph_build": {
        "model": GRAPH_MODEL,
        "via": "an OpenAI-compatible gateway (paid, INR billing)",
        "cost_inr": 95,
        "note": "includes about ₹5 lost to double-billed timeouts and model tests, before the extraction fixes in the changelog",
        "nvidia_free_tier": "The same build on NVIDIA's free tier ran at about 2.5 successful calls/min (rate limited): ~17 h for the corpus",
    },
    "answer_model": {
        "config": "advanced",
        "free": {"model": GENERATOR, "cost_inr_30": 0},
        "paid": {"model": "anthropic/claude-haiku-5.5", "cost_inr_30": 3.03, "cost_inr_per_answer": 0.10},
    },
    "load": {"json_s": 81, "binary_s": 2.2, "json_mb": 1700, "binary_mb": 402},
    "storage": {
        "articles": 609, "chunks": 3281, "embedding_dim": 2048,
        "raw_mb": 6.5,
        "hybrid": {"disk_mb": 45, "ram_mb": 340},
        "graph": {"disk_mb": 402, "ram_mb": 680, "entity_vectors_mb": 268},
    },
}


def cmd_site(args: argparse.Namespace) -> None:
    """Write site/public/benchmarks/*.json: current configs ranked, plus fixes, costs and storage."""

    site = HERE.parents[1] / "site" / "public" / "benchmarks"
    main_summary = read_json(args.work / "summary.json")
    held = HERE / "work_heldout" / "summary.json"
    held_summary = read_json(held) if held.exists() else {"configs": []}
    by_name = {c["name"]: c for c in main_summary["configs"]}
    held_by_name = {c["name"]: c for c in held_summary["configs"]}

    def pair(table, name):
        before, after = table.get(f"{name}__before_fix"), table.get(name)
        if not before or not after:
            return None
        keep = ("accuracy", "accuracy_judged_n", "evidence_recall", "strict_accuracy", "counts_by_type")
        return {"before": {k: before.get(k) for k in keep}, "after": {k: after.get(k) for k in keep}}

    fixes = []
    for name in ("corrective", "multi_query", "graph", "graph_context"):
        entry = {"config": name, "main": pair(by_name, name), "heldout": pair(held_by_name, name)}
        if entry["main"] or entry["heldout"]:
            fixes.append(entry)
    stats_path = args.work / f"{GRAPH_INDEX}_stats.json"
    graph_stats = read_json(stats_path) if stats_path.exists() else {}
    current = [c for c in main_summary["configs"] if "__" not in c["name"]]
    paid = by_name.get("advanced__haiku")
    payload = {
        **{k: v for k, v in main_summary.items() if k != "configs"},
        "configs": current,
        "fixes": fixes,
        "heldout_reference": held_by_name.get("advanced", {}).get("accuracy"),
        "answer_model": {**MEASURED["answer_model"], "free_result": by_name.get("advanced"), "paid_result": paid},
        "graph_build": {**MEASURED["graph_build"], **{k: graph_stats.get(k) for k in ("llm_calls", "tokens", "seconds")},
                        **(graph_stats.get("graph") or {})},
        "load": MEASURED["load"],
        "storage": MEASURED["storage"],
    }
    payload["models"] = {**main_summary["models"], "graph_builder": GRAPH_MODEL}
    per_question = [row for row in read_json(args.work / "per_question.json") if "__" not in row.get("config", "")]
    write_json(site / "multihop-rag.json", payload)
    write_json(site / "multihop-rag-per-question.json", per_question)
    print(f"wrote {site}: {len(current)} configs, {len(fixes)} fixes, {len(per_question)} answers")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=HERE / "work", help="data, index, answers and results")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sample = sub.add_parser("sample")
    sample.add_argument("--n", type=int, default=30)
    sample.add_argument("--exclude", type=Path, help="questions.json whose questions must not be reused (held-out set)")
    sub.add_parser("index")
    graph_index = sub.add_parser("graph-index")
    graph_index.add_argument("--batch", type=int, default=50, help="articles per saved batch")
    graph_index.add_argument("--workers", type=int, default=6)
    graph_index.add_argument("--rpm", type=float, default=30)
    graph_index.add_argument("--min-balance", type=float, default=2.0, help="stop when the gateway balance (Rs) drops below this")
    generate = sub.add_parser("generate")
    generate.add_argument("--configs", nargs="*", choices=list(CONFIGS))
    generate.add_argument("--workers", type=int, default=4)
    generate.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    generate.add_argument("--index", type=Path, help="index directory (default: <work>/index)")
    generate.add_argument("--save-as", help="answers file name, e.g. corrective__before_fix (one config only)")
    generate.add_argument("--llm", help="answer model instead of the NVIDIA generator, e.g. velona/anthropic/claude-haiku-5.5")
    judge = sub.add_parser("judge")
    judge.add_argument("--workers", type=int, default=2)
    judge.add_argument("--rpm", type=float, default=20, help="judge requests per minute")
    sub.add_parser("export")
    sub.add_parser("summarize")
    sub.add_parser("site", help="write the results JSON used by the website")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.command in {"index", "graph-index", "generate", "judge"} and not os.getenv("NVIDIA_API_KEY"):
        sys.exit("Set NVIDIA_API_KEY first")
    {"download": cmd_download, "sample": cmd_sample, "index": cmd_index, "generate": cmd_generate,
     "judge": cmd_judge, "graph-index": cmd_graph_index, "export": cmd_export, "summarize": cmd_summarize,
     "site": cmd_site}[args.command](args)


if __name__ == "__main__":
    main()
