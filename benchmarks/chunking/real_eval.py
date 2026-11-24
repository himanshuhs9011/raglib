"""Real-data chunking benchmark: your documents, a real embedding model and your golden questions.

Compares the embedding chunker's threshold types (and the fixed-size baseline) on retrieval quality,
chunk shape and embedding cost, so you can pick settings - and check the default
``semantic_min_tokens`` - on your own data instead of the synthetic comparison corpus.

    # local model, free (pip install raglib-py[hf])
    python benchmarks/chunking/real_eval.py --docs my_docs/ --golden eval/golden.jsonl \\
        --embedding huggingface/BAAI/bge-base-en-v1.5

    # paid API: prints the estimated tokens / cost and stops; add --yes to run it
    python benchmarks/chunking/real_eval.py --docs my_docs/ --embedding openai/text-embedding-3-small \\
        --price-per-1m 0.02

Golden set: one JSON object per line (see ``golden.example.jsonl``)::

    {"question": "How long do refunds take?", "relevant_text": ["refunds are paid within fourteen days"]}
    {"question": "Who approves travel?", "relevant_ids": ["policies/travel.md"]}

``relevant_text`` (preferred here) does not depend on how documents are chunked: a retrieved chunk is
relevant when it contains one of the snippets (case- and whitespace-insensitive). ``relevant_ids``
(document ids, or file paths absolute or relative to ``--docs``) works too.

Reported per configuration: recall@k, precision@k, number of chunks, average chunk size, and the
embedding tokens/cost that configuration needs on its own. Configs share one embedding cache, so the
real spend is lower than the sum of the rows.

Chunk sizes (``--chunk-size``, the size column) are measured with the embedding model's own tokenizer
when it has one (local Hugging Face models), so "fixed 500" means 500 model tokens: chunks fit a
512-token model and are never split into fragments to fit it. Without a tokenizer, tiktoken.

Scoring by ``relevant_text``: a snippet counts as found when the retrieved chunks *together* cover it
in its document (chunk character offsets), so a snippet that straddles a chunk boundary is a hit when
both chunks are retrieved; a retrieved chunk is relevant when it overlaps a snippet.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from raglib.core.text import count_tokens  # noqa: E402
from raglib.embedding.base_embedding import BaseEmbedding  # noqa: E402

FREE_PROVIDERS = {"huggingface", "hf", "ollama", "lmstudio", "vllm", "local", "mock", "free"}


class CountingEmbedding(BaseEmbedding):
    """Counts the tokens of every text one configuration embeds, then delegates to the shared model."""

    def __init__(self, inner: BaseEmbedding) -> None:
        self.inner = inner
        self.tokens = 0

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.inner.max_input_tokens  # the real limit (512 for bge-base), not a guess

    def count_input_tokens(self, text: str) -> Optional[int]:
        return self.inner.count_input_tokens(text)  # the model's own tokenizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.tokens += sum(count_tokens(text) for text in texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        self.tokens += count_tokens(text)
        return self.inner.embed_query(text)

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def model_name(self) -> Optional[str]:
        return self.inner.model_name


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_golden(path: Path) -> List[Dict[str, Any]]:
    """Read and validate the golden questions."""

    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question"):
            raise ValueError(f"{path}:{number}: missing 'question'")
        texts = row.get("relevant_text") or []
        row["relevant_text"] = [texts] if isinstance(texts, str) else list(texts)
        row["relevant_ids"] = list(row.get("relevant_ids") or [])
        if not row["relevant_text"] and not row["relevant_ids"]:
            raise ValueError(f"{path}:{number}: give 'relevant_text' or 'relevant_ids'")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no questions")
    return rows


def _snippet_spans(text: str, snippet: str) -> List[Tuple[int, int]]:
    """Where ``snippet`` occurs in ``text`` (case- and whitespace-insensitive)."""

    words = snippet.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(word) for word in words)
    return [match.span() for match in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _covered(span: Tuple[int, int], intervals: List[Tuple[int, int]]) -> bool:
    """True when the union of ``intervals`` covers ``span`` (whitespace gaps between chunks allowed)."""

    start, end = span
    reach = start
    for a, b in sorted(intervals):
        if a > reach + 1:  # a gap (one separator character is fine)
            if a >= end:
                break
            return False
        reach = max(reach, b)
        if reach >= end:
            return True
    return reach >= end


def score(sources: Sequence[Any], row: Dict[str, Any], k: int, documents: Optional[Dict[str, str]] = None) -> Tuple[float, float]:
    """(recall@k, precision@k) for one question.

    With ``documents`` (document id -> text) and chunk offsets, ``relevant_text`` snippets are
    matched by coverage: found when the retrieved chunks of a document together cover an occurrence;
    a chunk is relevant when it overlaps one. Otherwise a chunk must contain the whole snippet.
    """

    top = list(sources)[:k]
    if row["relevant_text"]:
        snippets = [text for text in row["relevant_text"] if text.strip()]
        positioned = documents is not None and all(
            "start_char" in doc.metadata and "end_char" in doc.metadata and doc.metadata.get("parent_id") in documents
            for doc in top
        )
        if positioned:
            found, relevant = 0, set()
            spans_by_doc: Dict[str, List[Tuple[int, int]]] = {}
            for index, doc in enumerate(top):
                spans_by_doc.setdefault(doc.metadata["parent_id"], []).append((doc.metadata["start_char"], doc.metadata["end_char"]))
            for snippet in snippets:
                hit = False
                for parent, text in documents.items():
                    occurrences = _snippet_spans(text, snippet)
                    if not occurrences:
                        continue
                    if any(_covered(span, spans_by_doc.get(parent, [])) for span in occurrences):
                        hit = True
                    for index, doc in enumerate(top):
                        if doc.metadata["parent_id"] == parent and any(
                            doc.metadata["start_char"] < b and doc.metadata["end_char"] > a for a, b in occurrences
                        ):
                            relevant.add(index)
                found += hit
            return found / len(snippets), len(relevant) / k
        normalized = [_normalize(text) for text in snippets]
        contents = [_normalize(doc.content) for doc in top]
        found = sum(1 for snippet in normalized if any(snippet in content for content in contents))
        relevant_chunks = sum(1 for content in contents if any(snippet in content for snippet in normalized))
        return found / len(normalized), relevant_chunks / k
    wanted = [_path(item) for item in row["relevant_ids"]]
    hits = [any(_matches_id(doc, item) for item in wanted) for doc in top]
    covered = sum(1 for item in wanted if any(_matches_id(doc, item) for doc in top))
    return covered / len(wanted), sum(hits) / k


def _path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip()


def _matches_id(doc: Any, wanted: str) -> bool:
    """A chunk matches a document id or a file path - absolute, or relative to the docs folder."""

    for candidate in (doc.id, doc.metadata.get("parent_id"), doc.metadata.get("file"), doc.source):
        candidate = _path(candidate)
        if candidate and (candidate == wanted or candidate.endswith("/" + wanted.lstrip("/"))):
            return True
    return doc.id.startswith(f"{wanted}_chunk_")


def configurations(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    """Baseline + every semantic threshold type at every semantic_min_tokens value."""

    configs: List[Tuple[str, Dict[str, Any]]] = [
        (f"fixed {args.chunk_size}/{args.overlap} (baseline)", {"chunking_strategy": "fixed"}),
    ]
    semantic = [
        ("percentile 95", {"semantic_breakpoint": "percentile"}),
        ("interquartile 1.5", {"semantic_breakpoint": "interquartile"}),
        ("gradient 95", {"semantic_breakpoint": "gradient"}),
        (f"number_of_chunks={args.number_of_chunks}", {"semantic_number_of_chunks": args.number_of_chunks}),
        ("standard_deviation 1.5", {"semantic_breakpoint": "standard_deviation"}),
    ]
    for minimum in args.min_tokens:
        for label, options in semantic:
            configs.append((f"{label}, min={minimum}", {"chunking_strategy": "embedding", "semantic_min_tokens": minimum, **options}))
    return configs


def load_embedding(provider: str, model: str) -> BaseEmbedding:
    """The shared embedding model (a seam for tests: replace it with a fake)."""

    from raglib import RAG

    return RAG(chat_llm="mock", embedding_llm=provider, embedding_model=model or None, token_counter="auto")._embedding


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="folder (or file) with your documents")
    parser.add_argument("--golden", default="eval/golden.jsonl", help="golden questions (JSONL)")
    parser.add_argument("--embedding", default="huggingface/BAAI/bge-base-en-v1.5",
                        help="provider/model, e.g. openai/text-embedding-3-small or huggingface/BAAI/bge-base-en-v1.5")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100, help="overlap of the fixed baseline")
    parser.add_argument("--number-of-chunks", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, nargs="+", default=[32, 64, 125])
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid", help="retrieval mode")
    parser.add_argument("--price-per-1m", type=float, help="USD per 1M embedding tokens (for the cost column)")
    parser.add_argument("--yes", action="store_true", help="confirm a run that calls a paid embedding API")
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args(argv)

    from raglib import RAG
    from raglib.loaders.document_loader import DocumentLoader

    documents = DocumentLoader().load(args.docs)
    golden = load_golden(Path(args.golden))
    provider, _, model = args.embedding.partition("/")
    configs = configurations(args)
    corpus_tokens = sum(count_tokens(doc.content) for doc in documents)
    semantic_configs = sum(1 for _, options in configs if options["chunking_strategy"] == "embedding")
    # Sentence windows (~2x the corpus) once thanks to the shared cache, plus every config's own chunks.
    estimate = int(corpus_tokens * 2 + corpus_tokens * len(configs) + len(golden) * 20 * len(configs))
    print(f"{len(documents)} documents ({corpus_tokens:,} tokens), {len(golden)} questions, {len(configs)} configurations "
          f"({semantic_configs} embedding-based), embedding={args.embedding}")
    cost_note = f" = ${estimate * args.price_per_1m / 1e6:,.2f}" if args.price_per_1m is not None else ""
    print(f"Estimated embedding tokens actually sent: ~{estimate:,}{cost_note} (upper bound; the cache removes repeats)\n")
    if provider.lower() not in FREE_PROVIDERS and not args.yes:
        print(f"'{provider}' is a paid API: nothing was sent. Check the estimate above and re-run with --yes to run it.")
        return 2

    shared = load_embedding(provider, model)
    from raglib.embedding.limits import TOKENIZER_SAMPLE

    model_tokens = shared.count_input_tokens(TOKENIZER_SAMPLE) is not None
    token_counter: Any = (lambda text: shared.count_input_tokens(text) or 0) if model_tokens else "auto"
    measure = token_counter if model_tokens else count_tokens
    unit = "the embedding model's tokens" if model_tokens else "tiktoken tokens (no model tokenizer)"
    print(f"Chunk sizes in {unit}.")
    texts = {doc.id: doc.content for doc in documents}
    rows: List[Dict[str, Any]] = []
    for label, options in configs:
        counting = CountingEmbedding(shared)
        rag = RAG(
            chat_llm="mock", embedding_llm=counting, vector_db="memory", retrieval_mode=args.mode, token_counter=token_counter,
            chunk_size=args.chunk_size, chunk_overlap=args.overlap if options["chunking_strategy"] == "fixed" else 0,
            top_k=args.k, **options,
        )
        rag.add(documents)
        chunks = rag.retriever.documents
        scores = [score(rag.retriever.retrieve(row["question"], top_k=args.k), row, args.k, texts) for row in golden]
        rows.append({
            "config": label,
            "recall": statistics.fmean(recall for recall, _ in scores),
            "precision": statistics.fmean(precision for _, precision in scores),
            "chunks": len(chunks),
            "avg_tokens": statistics.fmean(measure(chunk.content) for chunk in chunks) if chunks else 0.0,
            "split_to_fit": rag.ingest_usage()["oversized_chunks_split"],
            "embedding_tokens": counting.tokens,
            "usd": counting.tokens * args.price_per_1m / 1e6 if args.price_per_1m is not None else None,
            "fallbacks": rag.ingest_usage()["chunking_fallbacks"],
        })
        print(f"  done: {label}")

    k = args.k
    print(f"\n| config | recall@{k} | precision@{k} | chunks | avg tokens | split to fit model | embedding tokens | cost |")
    print("| :--- |" + " ---: |" * 7)
    for row in rows:
        usd = f"${row['usd']:.4f}" if row["usd"] is not None else "-"
        print(f"| {row['config']} | {row['recall']:.3f} | {row['precision']:.3f} | {row['chunks']} | "
              f"{row['avg_tokens']:.0f} | {row['split_to_fit']} | {row['embedding_tokens']:,} | {usd} |")
    if any(row["fallbacks"] for row in rows):
        print("\nWarning: some documents fell back to recursive chunking (embedding errors); see ingest_usage().")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
