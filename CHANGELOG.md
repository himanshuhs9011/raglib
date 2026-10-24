# Changelog

## 2.0.4 - 2026-10-24

- **`result.metadata` now carries pipeline diagnostics:** `crag_verdict`, `agent_steps`, `hop_queries`, `graph_mode`, `self_rag_llm_calls` were computed but dropped before the result was returned. Internal Document lists stay out, so metadata is JSON-serializable.
- **Examples:** `examples/rag_types_tour.py` (every rag_type, its options, and how to read each result field) and `examples/fastapi_app.py` (ask, stream, upload, delete, sessions, health; tested).

## 2.0.3 - 2026-10-24

- **Graph viewer layout:** nodes that started almost on top of each other were flung thousands of pixels away (repulsion had no minimum distance), so "Fit" zoomed out until the graph was a speck. Repulsion now has a minimum distance and node speed is capped.

## 2.0.2 - 2026-10-24

- **Runs in the browser (Pyodide) and other thread-less runtimes:** all parallel work goes through `raglib.core.concurrency.parallel_map`, which runs sequentially when threads are unavailable or cannot be started. Every `rag_type` now works offline in the browser, which powers the website playground.

## 2.0.1 - 2026-10-24

Fixes found by live testing every `rag_type` on OpenRouter free models.

- **hybrid_web:** web results never reached the answer. Weighted RRF (0.6/0.4, k=60) ranked the best web hit below corpus hit #31. Top-k slots are now split by weight (at least 1 web slot when the web returns results) and interleaved.
- **agentic / multi_hop:** evidence from separate steps or hops was merged with RRF, so repeated searches for one sub-question pushed out the evidence for another (the answer said the Bluefin code was "not mentioned" although step 1 found it). Steps and hops are now merged round-robin (`raglib.core.fusion.interleave`).
- **tool:** `rag_type="tool"` with `tools=[...]` raised an error. It now runs the agentic tool loop, where the LLM chooses which tools to call.
- **Generation prompt:** answer directly when the sources contain the answer (weaker models hedged with "not stated... however source 3 says...").

## 2.0.0 - 2026-10-24

Production-hardening release with breaking changes. See README section 16 for the migration table.

- **New: GraphRAG** (`rag_type="graph"`, `raglib.graphrag`): parallel, cached LLM entity/relation extraction; NetworkX (JSON-persisted) or Neo4j graph stores; hierarchical Leiden/Louvain communities with incremental reports; local search (no extra LLM calls) and global map-reduce search over embedding-prefiltered communities.
- **Retrieval:** BM25 + dense hybrid search by default with relative-score fusion (or RRF), metadata filters, MMR, neighbour (small-to-big) expansion, Unicode-aware tokenization (Indic, Arabic, Thai, Cyrillic, ...).
- **Reranking:** cross-encoder, Cohere, listwise-LLM and custom rerankers; the lossy lexical reranker was removed.
- **Generation:** delimited sources with prompt-injection guard, `[n]` citations parsed into `GenerationResult.citations`, "don't know" instruction, token-budgeted packing that trims instead of dropping, lost-in-the-middle ordering, streaming and async.
- **Conversations:** per-`session_id` memory with token budget and standalone-question condensation for every rag_type.
- **Ingestion:** idempotent upserts with content-hashed chunk ids, `update()`, `delete()`, `save()` / `load()` without re-embedding.
- **Strategies rewritten on one shared `BaseRAG` pipeline:** paper-faithful CRAG verdicts with batched grading and no implicit web search; Self-RAG with an LLM-call budget; real ReAct `AgenticRAG` with tools; iterative `MultiHopRAG`; parallel RRF `MultiQueryRAG`; `hybrid_web`; routing with a graph route.
- **Reliability:** LLM timeouts, jittered exponential-backoff retries, LRU response cache, cached + batched embeddings, per-stage `TraceEvent`s with an `on_event` hook, `raglib.eval` (hit rate, MRR, recall, nDCG, faithfulness, answer relevance, context precision, correctness).
- **Safety fixes:** substring keyword routing ("this" matched "hi" and skipped retrieval), silent mock LLM / embedding / vector-store fallbacks, shared Chroma collection across instances, deprecated default models.
- **LLM providers:** 17 providers from one registry: OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity, Cerebras, Azure OpenAI, LM Studio, vLLM and any OpenAI-compatible endpoint, alongside OpenAI, Anthropic, Gemini, Groq and Ollama. `llm="provider/model"` strings, provider auto-detection from environment keys, and longest-prefix key detection (OpenRouter `sk-or-` keys were previously treated as OpenAI). Chat keys are no longer sent to a different embedding provider.
- **Simpler usage:** `raglib.ask(source, question)` one-liner and a `raglib` CLI (`raglib ask`, `raglib chat`, `raglib providers`, also `python -m raglib`).
- **Graph visualization:** `rag.visualize_graph()`, `GraphContextWindow.to_html()` and `raglib graph` write a self-contained interactive HTML view (no CDN; content escaped against script injection).
- **Concurrency:** read-write locks in the retriever and in-memory store and locked graph-store reads. v1/early-v2 crashed in 25-42% of queries when documents were added during chats; now 0 errors under 16 concurrent sessions.
- **Scale:** float32 in-memory vectors (2.5 GB -> 1 GB at 30k chunks), argpartition top-k, streaming batched `add()` with `on_progress` and resume-after-failure, concurrent embedding batches.
- **Security:** SSRF guard for URL sources (private/metadata IPs, redirects, 20 MB cap), `allowed_dirs` path sandbox, user queries removed from INFO logs, `.env.example`, committed run reports with local paths untracked.
- **Rate limits and resilience:** a shared client-side limiter (`llm_rpm`; OpenRouter `:free` models default to 20/min, with chat and embeddings sharing one budget), retries that honour `Retry-After` / `X-RateLimit-Reset`, `QuotaExceededError` for daily or credit caps (fail fast), `llm_fallbacks=[...]` model fallback, and timeouts and retries for embedding calls.
- **GraphRAG cost:** communities with fewer than `min_llm_report_size` (3) entities get a template report instead of an LLM call.
- **GraphContextWindow:** optional `embedding=` for semantic seeding (finds paraphrased questions).
- **Ingestion and citations:** `rag.add()` accepts `Document` objects. Citations and prompt source labels use the file name and page, never the full local path (which used to be sent to the LLM provider).
- **Live-verified** on OpenRouter free chat and embedding models: needle-in-a-haystack, GraphRAG local and global, GraphContextWindow.
- **Packaging:** the core has no required dependencies; integrations are extras. Classifier set to Beta.
- **Removed:** `raglib.config`, `raglib.main`, duplicate `powerpoint_loader` / `plain_text_loader` / `webpage_loader`; `raglib.preprocessing` is a deprecated shim.

## 1.0.0 - 2026-10-14

- **Major Release (v1.0.0):**
  - Added selectable chunking strategies (`fixed`, `recursive`, and `llm` semantic chunking) in `ChunkSplitter` and `RAG(..., chunking_strategy=...)`.
  - Added standalone Graph-Based Context Window engine (`GraphContextWindow`, `GraphNode`, `ConnectedNodeRef`, `GraphContextResult`) supporting hierarchical document graphs (`Document -> Section -> Chunk`), neighbor summary signposts (`NEXT_CHUNK`, `PREV_CHUNK`, `SHARED_ENTITY`), and multi-turn conversational chat graphs (`PREV_TURN`, `SAME_TOPIC_TURN`, `CITED_CHUNK`).
  - Upgraded `Retriever` to perform incremental document indexing and reconstruct documents directly from pre-populated persistent vector stores (`ChromaVectorStore`, `InMemoryVectorStore`).
  - Replaced non-deterministic `hash()` IDs with deterministic SHA-256 document IDs across all loaders, providers, and retrievers.
  - Hardened `SelfRAG`, `RoutingRAG`, `MultiQueryRAG`, `MultiHopRAG`, `Generator`, `WebLoader`, and `DuckDuckGoProvider` for production reliability.
  - Exported all 12 RAG strategy classes, core components, and graph context classes from top-level `raglib`.
  - Updated repository URLs to `https://github.com/himanshuhs9011/raglib`.

## 0.1.10 - 2026-04-06

- Added additional interactive chat kill-switch commands: `bye` and `stop`
- Updated interactive session banner/help text to show all exit commands
- Updated README and usage docs so users can clearly see how to stop continuous chat sessions

## 0.1.9 - 2026-04-06

- Clarified chat continuity behavior in README for PyPI users (default stateless turn handling vs memory-enabled multi-turn context)
- Added two explicit usage snippets: one-time/stateless query mode and continuous/contextual memory mode
- Updated usage guide with matching chat continuity section and corrected section numbering

## 0.1.8 - 2026-04-06

- Added first-class web search integration in `RAG(...)` with new inputs:
	- `web_search_provider`
	- `web_search_api_key`
	- `web_search_base_url`
	- `web_search_cse_id`
	- `web_search_provider_kwargs`
	- `validate_web_search_api_key`
- Added authenticated web provider key validation with environment variable fallback support
- Added cloud chat/embedding key validation for explicit provider selection (`openai`, `anthropic`, `groq`, `google`)
- Updated default web provider to DuckDuckGo for out-of-the-box internet search
- Integrated `langchain_community` DuckDuckGo tools (`DuckDuckGoSearchRun` with validated output) for default web search
- Upgraded `WebRetriever` into a plug-and-play provider orchestrator with provider-name factory and automatic DuckDuckGo fallback
- Updated `WebRetriever` to fail safely by default and return empty results when web provider calls fail
- Updated `raglib/main.py` demo wiring to match latest orchestrator constructor signatures
- Added explicit web-provider count summary to README for PyPI package page clarity
- Updated README and usage docs with production "3-key" input patterns and provider/key requirements

## 0.1.7 - 2026-04-05

- Reworked CorrectiveRAG into a faithful three-path CRAG flow: CORRECT, AMBIGUOUS, and INCORRECT routing
- Added LLM-based retrieval evaluator and knowledge-refinement strip filtering pipeline
- Added CRAG web-search branch with query rewriting and context merge policy (k_in / k_ex)
- Added robust fallback behavior when optional LLM/web components are not configured

## 0.1.6 - 2026-04-05

- Updated default package dependencies so `pip install raglib-py` includes the full runtime stack used by core code paths
- Fixed common first-run missing module issue for Ollama chat/embedding by bundling `langchain-ollama` in base install
- Kept optional extras for users who prefer explicit feature grouping

## 0.1.5 - 2026-04-05

- Added explicit per-provider key options in the RAG facade: chat_api_key, embedding_api_key, vision_api_key
- Added endpoint options in the RAG facade: chat_base_url and vision_base_url
- Updated README and usage docs with clear API key ownership guidance and constructor examples

## 0.1.4 - 2026-04-05

- Expanded README into a full PyPI-first user guide with complete setup and usage coverage
- Documented all 12 built-in RAG strategies and usage patterns
- Standardized install instructions and runtime dependency hints to use raglib-py

## 0.1.3 - 2026-04-05

- Release version bump for PyPI publication
- Packaging and distribution preflight improvements

## 0.1.2 - 2026-04-04

- Initial release of raglib
- Added modular RAG components, providers, and orchestrators
- Added offline-capable MockLLMClient
- Added full production-grade test suite
