"""Compare raglib's embedding chunker with LangChain's SemanticChunker on documents with known topic changes.

Runs offline: a deterministic hashing bag-of-words embedding (no model, no network) is shared by
both splitters, so differences come from the chunking algorithms only. The synthetic corpus has
known topic boundaries and includes Markdown tables, lists, abbreviations ("Dr. Sharma") and Hindi.

    pip install langchain-experimental      # optional: without it only raglib is measured
    python benchmarks/chunking/compare_langchain.py [--docs 40] [--chunk-size 120] [--json out.json]

Metrics (per configuration, over the whole corpus):
    boundary P/R/F1   predicted chunk starts vs true topic changes (match within one sentence)
    chunks/doc        mean number of chunks per document
    max / mean size   chunk size in tiktoken tokens when installed, else words
    over limit        % of chunks larger than --chunk-size (LangChain has no maximum)
    blocks intact     % of tables / lists found unbroken, with their line breaks, in one chunk
    abbr. splits      chunk boundaries placed right after an abbreviation such as "Dr."
    texts embedded    texts sent to the embedding model (cost proxy)
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOPICS: Dict[str, List[str]] = {
    "billing": ["invoice", "payment", "refund", "subscription", "billing", "charge", "credit", "plan", "price", "receipt"],
    "rockets": ["rocket", "launch", "thrust", "fuel", "orbit", "engine", "stage", "booster", "payload", "liftoff"],
    "medicine": ["patient", "dose", "clinic", "symptom", "therapy", "nurse", "diagnosis", "vaccine", "treatment", "ward"],
    "cooking": ["recipe", "oven", "flour", "butter", "simmer", "spice", "dough", "garlic", "sauce", "bake"],
    "security": ["password", "firewall", "breach", "encryption", "token", "audit", "malware", "access", "phishing", "patch"],
    "gardening": ["soil", "seed", "compost", "prune", "bloom", "root", "watering", "mulch", "harvest", "greenhouse"],
}
FILLER = ["the", "team", "said", "that", "every", "week", "we", "review", "our", "notes", "and", "update", "it"]
ABBREVIATIONS = ["Dr.", "Mr.", "e.g.", "Inc."]
HINDI = ["यह योजना हर महीने बिल की जाती है।", "भुगतान कार्ड से किया जाता है।", "रसीद ईमेल पर भेजी जाती है।"]


class HashingEmbedding:
    """Deterministic bag-of-words embedding: each word adds to one of 128 hashed dimensions."""

    dims = 128

    def __init__(self) -> None:
        self.texts_embedded = 0

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dims
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dims
            vector[bucket] += 0.2 if word in FILLER else 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.texts_embedded += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def _sentence(rng: random.Random, topic: str) -> str:
    words = rng.sample(TOPICS[topic], 4) + rng.sample(FILLER, 4)
    rng.shuffle(words)
    sentence = " ".join(words)
    if rng.random() < 0.25:
        sentence = f"{rng.choice(ABBREVIATIONS)} {sentence}"  # an abbreviation inside the sentence
    return sentence[0].upper() + sentence[1:] + "."


def _block(rng: random.Random, topic: str) -> str:
    words = TOPICS[topic]
    if rng.random() < 0.5:
        rows = [f"| {rng.choice(words)} | {rng.choice(words)} {rng.randint(1, 99)} |" for _ in range(3)]
        return "\n".join(["| Item | Detail |", "|------|--------|", *rows])
    return "\n".join(f"- {rng.choice(words)} {rng.choice(words)} {rng.choice(FILLER)}" for _ in range(3))


def make_corpus(docs: int, seed: int = 7) -> List[Dict[str, Any]]:
    """Documents of 3-5 topic segments; returns text, true boundary offsets and the blocks they contain."""

    rng = random.Random(seed)
    corpus = []
    for index in range(docs):
        topics = rng.sample(list(TOPICS), rng.randint(3, 5))
        parts: List[str] = []
        boundaries: List[int] = []
        blocks: List[str] = []
        position = 0
        for number, topic in enumerate(topics):
            sentences = [_sentence(rng, topic) for _ in range(rng.randint(4, 7))]
            segment = " ".join(sentences)
            if rng.random() < 0.5:
                block = _block(rng, topic)
                blocks.append(block)
                segment += "\n\n" + block + "\n\n" + _sentence(rng, topic)
            if topic == "billing" and rng.random() < 0.5:
                segment += " " + " ".join(HINDI)
            if number:
                parts.append("\n\n")
                position += 2
                boundaries.append(position)
            parts.append(segment)
            position += len(segment)
        corpus.append({"id": f"doc{index}", "text": "".join(parts), "boundaries": boundaries, "blocks": blocks})
    return corpus


def _size_counter() -> Tuple[Callable[[str], int], str]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(encoding.encode(text, disallowed_special=()))), "tiktoken tokens"
    except Exception:  # noqa: BLE001
        return (lambda text: len(text.split())), "words"


def _starts(text: str, chunks: Sequence[str], offsets: Optional[Sequence[int]]) -> List[int]:
    """Character offsets where chunks start (located by text search when no offsets are given)."""

    if offsets is not None:
        return list(offsets)
    starts, cursor = [], 0
    for chunk in chunks:
        probe = chunk.strip().split("\n")[0][:25]
        found = text.find(probe, cursor) if probe else -1
        if found < 0:
            found = text.find(probe) if probe else -1
        starts.append(max(found, 0))
        cursor = max(found, cursor)
    return starts


def _evaluate(corpus, results, chunk_size: int, size: Callable[[str], int], embedder: HashingEmbedding) -> Dict[str, Any]:
    tp = fp = fn = 0
    sizes: List[int] = []
    blocks_total = blocks_intact = abbreviation_splits = 0
    counts = []
    for doc, (chunks, offsets) in zip(corpus, results):
        text = doc["text"]
        counts.append(len(chunks))
        sizes.extend(size(chunk) for chunk in chunks)
        predicted = _starts(text, chunks, offsets)[1:]
        # Sentence (or line) starts: a predicted boundary matches a true one within one sentence.
        sentence_starts = [0] + [m.end() for m in re.finditer(r"[.!?।]\s+|\n+", text)]

        def sentence_index(offset: int) -> int:
            return bisect.bisect_right(sentence_starts, offset) - 1

        matched = set()
        for start in predicted:
            hit = next(
                (b for b in doc["boundaries"] if abs(sentence_index(b) - sentence_index(start)) <= 1 and b not in matched),
                None,
            )
            if hit is None:
                fp += 1
            else:
                matched.add(hit)
                tp += 1
        fn += len(doc["boundaries"]) - len(matched)
        for block in doc["blocks"]:
            blocks_total += 1
            blocks_intact += any(block in chunk for chunk in chunks)
        for start in predicted:
            before = text[:start].rstrip().split()
            abbreviation_splits += bool(before) and before[-1] in ABBREVIATIONS
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "chunks_per_doc": statistics.fmean(counts),
        "max_size": max(sizes),
        "mean_size": statistics.fmean(sizes),
        "over_limit_pct": 100.0 * sum(s > chunk_size for s in sizes) / len(sizes),
        "blocks_intact_pct": 100.0 * blocks_intact / blocks_total if blocks_total else 100.0,
        "abbreviation_splits": abbreviation_splits,
        "texts_embedded": embedder.texts_embedded,
    }


def run_raglib(corpus, chunk_size: int, options: Dict[str, Any]):
    from raglib.embedding.base_embedding import BaseEmbedding
    from raglib.loaders.chunk_splitter import ChunkSplitter
    from raglib.schemas import Document

    embedder = HashingEmbedding()

    class Adapter(BaseEmbedding):
        def embed_documents(self, texts):
            return embedder.embed_documents(texts)

        def embed_query(self, text):
            return embedder.embed_query(text)

        @property
        def dimension(self):
            return HashingEmbedding.dims

    splitter = ChunkSplitter(
        chunk_size=chunk_size, overlap=0, min_tokens=1, strategy="embedding", embedding=Adapter(),
        token_counter="auto", on_embedding_failure="raise", **options,
    )
    results = []
    for doc in corpus:
        chunks = splitter.split([Document(id=doc["id"], content=doc["text"])])
        offsets = [chunk.metadata.get("start_char") for chunk in chunks]
        results.append(([chunk.content for chunk in chunks], offsets if None not in offsets else None))
    return results, embedder


def run_langchain(corpus, options: Dict[str, Any]):
    from langchain_core.embeddings import Embeddings
    from langchain_experimental.text_splitter import SemanticChunker

    embedder = HashingEmbedding()

    class Adapter(Embeddings):
        def embed_documents(self, texts):
            return embedder.embed_documents(texts)

        def embed_query(self, text):
            return embedder.embed_query(text)

    splitter = SemanticChunker(Adapter(), **options)
    return [(splitter.split_text(doc["text"]), None) for doc in corpus], embedder


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args(argv)

    corpus = make_corpus(args.docs)
    size, unit = _size_counter()
    true_chunks = round(statistics.fmean(len(doc["boundaries"]) + 1 for doc in corpus))
    configs: List[Tuple[str, str, Dict[str, Any]]] = [
        ("raglib", "percentile 95", {"semantic_breakpoint": "percentile"}),
        ("raglib", "standard_deviation 3", {"semantic_breakpoint": "standard_deviation"}),
        ("raglib", "interquartile 1.5", {"semantic_breakpoint": "interquartile"}),
        ("raglib", "gradient 95", {"semantic_breakpoint": "gradient"}),
        ("raglib", f"number_of_chunks={true_chunks}", {"number_of_chunks": true_chunks}),
        ("langchain", "percentile 95", {"breakpoint_threshold_type": "percentile"}),
        ("langchain", "standard_deviation 3", {"breakpoint_threshold_type": "standard_deviation"}),
        ("langchain", "interquartile 1.5", {"breakpoint_threshold_type": "interquartile"}),
        ("langchain", "gradient 95", {"breakpoint_threshold_type": "gradient"}),
        ("langchain", f"number_of_chunks={true_chunks}", {"number_of_chunks": true_chunks}),
    ]
    rows = []
    for library, label, options in configs:
        try:
            if library == "raglib":
                results, embedder = run_raglib(corpus, args.chunk_size, options)
            else:
                results, embedder = run_langchain(corpus, options)
        except ImportError as exc:
            rows.append({"library": library, "config": label, "error": f"not installed ({exc.name})"})
            continue
        except (TypeError, ValueError) as exc:  # an option this version does not support
            rows.append({"library": library, "config": label, "error": f"not supported ({str(exc)[:60]})"})
            continue
        rows.append({"library": library, "config": label, **_evaluate(corpus, results, args.chunk_size, size, embedder)})

    print(f"{args.docs} documents, {sum(len(d['boundaries']) for d in corpus)} true topic changes, "
          f"chunk_size={args.chunk_size} ({unit}); sizes in {unit}\n")
    header = "| library | config | P | R | F1 | chunks/doc | max size | mean size | over limit | blocks intact | abbr. splits | texts embedded |"
    print(header)
    print("| :--- | :--- |" + " ---: |" * 10)
    for row in rows:
        if "error" in row:
            print(f"| {row['library']} | {row['config']} | {row['error']} |" + " |" * 10)
            continue
        print(
            f"| {row['library']} | {row['config']} | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1']:.2f} | "
            f"{row['chunks_per_doc']:.1f} | {row['max_size']} | {row['mean_size']:.0f} | {row['over_limit_pct']:.0f}% | "
            f"{row['blocks_intact_pct']:.0f}% | {row['abbreviation_splits']} | {row['texts_embedded']} |"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
