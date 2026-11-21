# raglib on MultiHop-RAG

This benchmark compares raglib's RAG types on a public, third-party benchmark. Every configuration gets the same questions, corpus, chunking, embedding model, generator and judge, so score differences come from the pipeline.

## Data

[MultiHop-RAG](https://arxiv.org/abs/2401.15391) (Yixuan Tang and Yi Yang, 2024) is a corpus of 609 news articles from late 2023, with 2,556 questions whose evidence is spread over 2 to 4 articles. It is published under ODC-BY at [huggingface.co/datasets/yixuantt/MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG). Each question has a type:

| Type | What it asks | Gold answer |
| :--- | :--- | :--- |
| `inference_query` | Combine facts from several articles to identify something | Entity, e.g. "Sam Bankman-Fried" |
| `comparison_query` | Compare what two sources report | Usually Yes / No |
| `temporal_query` | Order or compare events in time | Usually Yes / No or before / after |
| `null_query` | Asks for something no article contains | "Insufficient information" (the system must decline) |

The pilot uses 30 questions, stratified by type (8 inference, 8 comparison, 7 temporal, 7 null), sampled with seed 42. The sampled dataset indexes are in `work/questions.json`.

## Setup (identical for every configuration)

- **Indexing:** each article becomes title, source, date and author lines followed by the body, because the questions refer to sources and dates. raglib's default chunking is used (400 words, 50 overlap). The index is built once and loaded by every configuration.
- **Embeddings:** `nvidia/nemotron-3-embed-1b`.
- **Generator:** `nvidia/nemotron-3.5-lightning-30b-a3b` at temperature 0, with raglib's default prompt and `top_k=5`.
- **Judge:** `nvidia/nemotron-3-ultra-550b-a55b`, a different and larger model than the generator, though from the same vendor (see Limitations).
- All models run on build.nvidia.com, except GraphRAG's graph builder (below) and the paid-answer-model comparison.

## Configurations

| Name | rag_type | Settings |
| :--- | :--- | :--- |
| naive | `naive` | `retrieval_mode="dense"` (vector search only) |
| advanced | `advanced` | library defaults (BM25 + dense hybrid) |
| advanced_rerank | `advanced` | `reranker="cross_encoder"` (BAAI/bge-reranker-base), `fetch_k=20` |
| multi_query | `multi_query` | defaults |
| multi_hop | `multi_hop` | defaults |
| corrective | `corrective` | defaults; no web search configured, so CRAG cannot fall back to the web |
| self | `self` | defaults |
| agentic | `agentic` | defaults; document search only |
| graph | `graph` | `graph_llm="velona/inclusionai/ling-3.0-flash"` (graph built once, see below); answers by the same generator |
| graph_context | `graph_context` | GraphContextWindow over the same corpus; no LLM calls to build |

Not run, and why: `hybrid` (same retrieval as `advanced` here), `web` and `hybrid_web` (live web search is not reproducible and can leak answers), `routing` (with web and graph off it reduces to `advanced`), `memory` (multi-turn), `tool` (needs app-specific tools).

**GraphRAG's graph** needs one extraction call per chunk plus one call per community summary: 5,120 calls and 17M tokens for 609 articles, giving 32,754 entities, 60,522 relations and 1,617 communities. On NVIDIA's free tier this ran at about 2.5 successful calls per minute (rate limited), which is roughly 17 hours. It was therefore built with a cheap model through a paid OpenAI-compatible gateway: about 2.8 hours and about ₹95. About 9% of chunks returned an empty reply from that model even after a retry. They stay searchable but add nothing to the graph. `python bench.py graph-index` builds it, resumes after interruptions, and stops at `--min-balance`.

**Fixes found by the benchmark** were checked on a second, held-out sample of 30 questions (`--work work_heldout`, seed 7, no overlap with the first sample). Each was run against a snapshot of the pre-fix code (`RAGLIB_SRC`). Pre-fix answers are kept as `<config>__before_fix`.

**Paid answer model:** `generate --llm velona/anthropic/claude-haiku-5.5 --save-as advanced__haiku` changes only the answer model. Retrieval and the judge stay the same.

## Metrics

Two scores need no LLM judge:

- **Accuracy** is MultiHop-RAG's own measure. A null query counts as correct when the answer declines. Otherwise the gold answer must appear in the response, and for Yes / No the first yes or no word of the response must match. Matching is strict, so aliases such as "SBF" for "Sam Bankman-Fried" count as wrong.
- **Evidence recall** is the share of the gold evidence articles that were among the sources given to the generator.

[Open RAG Eval](https://github.com/vectara/open-rag-eval) 0.3.0 (Vectara, TREC-RAG methodology) adds metrics that need no golden answers:

- **UMBRELA** grades the relevance of each retrieved passage from 0 to 3.
- **AutoNuggetizer** checks whether the answer covers the key facts (nuggets).
- **Citation precision** checks whether each cited passage supports its sentence.
- **HHEM** is Vectara's hallucination model, run locally: factual consistency of the answer with the passages.

## Reproduce

```bash
export NVIDIA_API_KEY=nvapi-...
pip install -e . "open-rag-eval==0.3.0" "sentence-transformers>=3"
python benchmarks/multihop_rag/bench.py download
python benchmarks/multihop_rag/bench.py sample --n 30
python benchmarks/multihop_rag/bench.py index
python benchmarks/multihop_rag/bench.py generate
python benchmarks/multihop_rag/bench.py export
for f in benchmarks/multihop_rag/work/ore/*.yaml; do open-rag-eval eval --config "$f"; done
python benchmarks/multihop_rag/bench.py summarize
python benchmarks/multihop_rag/bench.py site      # JSON for the website's Benchmarks page
```

## Limitations

- **30 questions is a pilot.** One question is about 3 percentage points of accuracy, and differences of a few points between configurations are within noise.
- **Judge and generator share a vendor.** The judge is a different, much larger model, but a fully independent judge (another vendor) would be stronger.
- **One generator model.** Rankings can change with a stronger or weaker generator. Swapping the answer model of `advanced` for Claude Haiku 5.5 left accuracy unchanged (76.7%), but answers came about 4× faster.
- **No corpus-wide questions.** MultiHop-RAG asks about specific facts in specific articles. It has no "main themes" questions, which is where GraphRAG's global search is meant to help.
- **Strict string matching** undercounts correct answers that use aliases. The LLM-judged nugget metric compensates.
- **raglib's own benchmark.** The method, code and raw outputs are published so anyone can check or rerun them.
