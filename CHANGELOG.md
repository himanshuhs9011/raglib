# Changelog

## Unreleased

**Guardrails, tier 2 (LLM judge, used sparingly).** See README "Tier 2: the LLM judge".
- Tier 1 guards escalate to the judge only when their confidence is below `escalate_below` (per guard, default 0.6), when their verdict was unsure, or for high-risk requests: a question for an agent with tools, or an answer from non-public documents.
- There are two hardened judgments. `input_safety` checks injection and toxicity in one shared call. `grounding` runs on `grounding_sample_rate` (default 5%) of normal answers plus every answer that used non-public documents. A local NLI model checks every answer and escalates only when unsure or non-public.
- `RAG(judge_model=...)` sets the judge (default: the cheapest configured chat model). `RAG(judge_budget=JudgeBudget(...))` caps judge calls or cost per minute or per day; once a cap is reached, the tier-1 verdict stands and the trace records `budget_exhausted`.
- Unparseable judge output is unsure: input safety fails closed and grounding fails open.
- Verdicts are cached by input hash plus the judge's configuration (with single-flight, so concurrent escalations of one question make one call). They are never cached for grounding over non-public documents.
- Each judge call is its own `judge.<kind>` trace stage, marked `escalated` with the reason, so it gets its own `LatencyStats` stage and `CostReport` line. Guard decisions carry `escalated` and `judge`. `GuardContext.high_risk` is new.

**Guardrails, tier 1 (cheap model checks, opt-in).** See README "Guardrails" for the model table.
- `TopicGuard(topics)`: compares the question with example questions per topic, using the query embedding retrieval makes anyway (served from the embedding cache). No extra model call. It runs before retrieval and blocks off-topic questions.
- `InjectionClassifier` and `ToxicityClassifier`: a local classifier if you configured one, else your cheapest configured chat model (toxicity: OpenAI's free moderation endpoint when the chat provider is OpenAI), else skipped.
- `GroundingGuard`: a local NLI model on every answer, else the cheapest chat model on `grounding_sample_rate` of answers (default 5%), else skipped. It is a quality guard and is not streaming-safe.
- Model choice is local, then api, then skipped. The mode is recorded on every decision and trace stage. A skipped guard gives a one-time `GuardSkippedWarning` naming the local model to install and the setting to add. Local models come from `local_model=` or the new `RAG(guard_models={...})`. A model id is loaded with `local_files_only=True`, so nothing is downloaded unless you pass `allow_download=True`. New extra: `pip install raglib-py[guards]`.
- Tier 1 input classifiers run in parallel with retrieval. Every LLM call (generation, agent planning, query rewriting, routing) waits for them, and a block cancels the next retrieval stage. A blocked question never reaches a model. Cancellation raises `GuardCancelled` (a `BaseException`), so components that degrade on errors cannot swallow it. Retrieval work that had already started is still paid for.
- API-based guard calls are hardened like the main prompts. The checked text sits inside the per-prompt delimiter, the prompt states a data-not-instructions rule (including notes to "the safety checker"), and the reply must be one JSON object of a fixed shape. Unparseable or malformed verdicts are *unsure*: security guards fail closed and grounding fails open.
- `CitationAccessGuard(summary_rule="all_sources")`: a GraphRAG summary is visible when the user may read every one of its source chunks, which is more precise than the default strictest-tags rule for users in several departments. Summaries now also carry `source_chunk_count`.
- Guard API calls are traced in the guard's own stage, so they get their own `CostReport` line. The offline mock LLM answers the guard prompts with simple heuristics, for demos and tests.

**Security: secure by default.** Every `RAG(...)` now gets four protections. None of them changes the answer to a normal question: with them on, plain `RAG(...)` gives the same answers, sources and trace stages as before, and still streams token by token.
- `CitationAccessGuard` whenever `access_control=True`. It withholds an answer that cites a document the user may not read, which only happens when a custom retriever, tool or cache slips past the retrieval filters. Turn off with `secure_defaults={"citation_access": False}`.
- An agent tool limit, `ToolPolicy(max_tool_calls=10)`. It stops a looping agent; the default `max_steps=4` never reaches it. Turn off with `secure_defaults={"tool_limit": False}`, or pass your own `tool_policy=`.
- Injection flags: `InjectionHeuristic(flag_only=True)` on the question. Turn off with `secure_defaults={"injection_flags": False}`.
- Secret flags: `OutputLeakGuard(check_pii=False, flag_only=True)` on the answer, for API keys, tokens and private keys (never PII). Turn off with `secure_defaults={"secret_flags": False}`.
- `secure_defaults=False` turns all four off. A guard of the same class that you configure yourself replaces the default one.

Flag-only guards never block or modify. Their findings go to the new `result.guard_flags` and to a `guard_flag` event on `on_event`. They add no trace stage, and nothing at all when they find nothing. Flag-only output guards run once on the finished answer, so they never hold back a stream. Any guard accepts `flag_only=True`.

With `access_control=True`, streaming trails the model by about one word (8 characters), because the citation check needs whole `[n]` markers.

**Security: GraphRAG summaries inherit access tags.** Community reports record the chunks they were built from (`Community.source_chunk_ids`, set when the report is built; older saved graphs fall back to the members' chunks). Every graph summary document (entities, relations, community reports, global-search points) carries the strictest access tags of its source chunks, from the new `AccessControl.strictest`: the highest `access_level`, and the intersection of `department` / `allowed_roles`. An empty intersection means nobody can see it. `CitationAccessGuard` now checks summaries by these tags instead of skipping them, and a summary without tags is denied. Query-time filtering is unchanged: a report is shown only when every member entity is fully visible to the user.

Two protections were already unconditional and stay that way: source delimiters and escaping in every prompt, and the agent's web-search argument guard. A `RAG(...)` created without `preset="production"` and without guards of your own logs one INFO line per process pointing to the guardrails docs (hide it with `RAGLIB_GUARD_TIP=0`). `CitationAccessGuard` now judges a source by its access tags, so chunk copies without the internal `_acl_*` keys are handled. GraphRAG's derived summaries (`metadata["supplementary"]`) are not re-checked, because the graph filters them per user. PII masking, the "nothing relevant" exit, the question length limit and JSON schema checks stay in `preset="production"` or opt-in, and rate limits and budgets stay opt-in.

**Guardrails, tier 0 (free rules).** New `raglib.guards` and `RAG(input_guards=..., retrieval_guards=..., output_guards=..., tool_policy=..., guard_cache_ttl=3600)`. Guards run cheapest tier first and stop at the first block. Each returns allow / block (with a user-facing message) / modify, plus a confidence. See README "Guardrails".
- Input: `PIIMasker` (e-mail, phone, Aadhaar, PAN and card numbers, with checksums; it runs before the question reaches traces, `LatencyStats` tags, caches, memory or the LLM), `MaxQuestionTokens`, `UserRateLimit`, `UserDailyBudget` (priced like `CostReport`; skipped with a one-time `GuardSkippedWarning` without `prices=`), `InjectionHeuristic`.
- Retrieval: `NothingRelevant` answers "I couldn't find this in the documents." without an LLM call when the best chunk is below a threshold. The default threshold depends on the score type: cosine 0.15, Cohere 0.02, cross-encoder logit -7.
- Output: `OutputLeakGuard` (secrets, and PII that is not in the user's sources), `CitationAccessGuard`, `JSONSchemaGuard`, `MaxOutputTokens`.
- Agent: `ToolPolicy`, with a tool allowlist per role, a cap on tool calls per question and argument limits.
- Security guards fail closed and quality guards fail open; `fail_closed=` overrides this. Deterministic decisions are cached by input hash.
- Every decision is in `result.metadata["guards"]` (plus `blocked_by` / `block_reason` on a block) and is traced as its own stage `guard.<name>`. Guard stages appear in `on_event`, `LatencyStats` (listed first; the `total` event has a new `blocked_by` run tag) and `CostReport` (each guard on its own line, $0 for tier 0).
- `preset="production"` turns on the defaults for every stage. This preset is unreleased, so the change is not breaking. A plain `RAG(...)` gets only the secure defaults (see "Security: secure by default").
- With output guards, `stream()` yields the checked answer in one piece instead of token by token.
- `BaseRAG.run` / `stream` accept optional `tracer=` / `meter=` to continue a caller's trace and usage, and `usage_scope(meter)` reuses an active meter without double counting.
- An answer refused by a guard is not stored in the answer cache.
- PII masking is reversible: typed placeholders (`<EMAIL_1>`), a per-request map that is never traced, logged or cached, and the real values restored in the answer returned to the caller (`PIIMasker(restore=True)`). Stored copies (caches, memory, `output_dir`, traces) keep the placeholders. `PIIMasker` always runs before every other guard.
- The default PII set is e-mail, card (Luhn), Aadhaar (Verhoeff, consistent separators) and PAN (strict format). Phone numbers are opt-in (`phone=True`). Every number pattern is bounded, so order ids, invoice numbers, dates, prices and version numbers are not masked.
- Streaming with output guards: guards declare `streaming_safe`, and the safe ones (leak masking, citation access, truncation) check the answer piece by piece, so `stream()` keeps streaming, the production preset included. Pieces are released per sentence, with a 64-character hold-back that never cuts through a secret, card number, citation or placeholder. Only non-streaming-safe guards (`JSONSchemaGuard`) buffer the whole answer.
- Rate-limit and budget identity: `RAG(rate_limit_key=fn)`, `query(..., user_id=..., request={...})` (no access control needed), a per-guard `key=`, and `anonymous_policy="skip" | "shared" | "block"` (default `"skip"`). A `session_id` alone no longer counts as an identity. Counters go through a pluggable `GuardStore` (`RAG(guard_store=...)`). The default `InMemoryGuardStore` is per process.
- The answer-cache key includes a hash of the active guard configuration, so changing guards never serves old cached answers.
- `RAG(stream_holdback_chars=64)`: how far streamed text may trail the model while streaming-safe output guards check it. The minimum is 40 (the longest secret prefix that can contain a space, a PEM header), and smaller values raise `ValueError`. Release still happens at the end of a sentence or once text is past the window.
- `preset="production"` sets `llm_max_tokens=1024`, which caps what each answer costs. `MaxOutputTokens` only trims what is shown.
- Guard messages, reasons, errors and log lines are sanitized: the checked text is removed, and PII and secrets are masked. The internal retrieval-block exception is now `GuardBlockedError` (`GuardBlocked` is kept as an alias).

**Security: filters were not applied on several retrieval paths.** `query(..., filters=...)` restricted dense and BM25 search, but these paths ignored it and could return documents outside the filter:
- GraphRAG local and global search (`rag_type="graph"`), the router's `graph` and `hybrid` routes (`rag_type="routing"`), and the agentic `graph_search` tool.
- The decision engine's web/hybrid fallback (`BaseRAG._retrieve_by_mode`), and `HybridRAG` when no corpus retriever is set.
- `web_search_provider="local"`, which searches the corpus.
- `neighbor_window` expansion, which could add neighbouring chunks that fail the filter.

All of them now apply the filter. GraphRAG uses an entity, relation or community report only when the source chunks behind it pass the filter. A backend that cannot enforce a filter now raises `FilterNotSupportedError` instead of returning everything. As defence in depth, dense hits are re-checked against the filter, and hits a vector store returns outside it are dropped and logged.

**Access control.** `RAG(access_control=True)` adds per-document permissions: tag documents at ingest (`add(..., access=access_tags("confidential", department="finance", allowed_roles=["cfo"]))`) and pass `query(..., user=UserContext(...))`. raglib builds the permission filter, ANDs it with your filters and applies it as a pre-filter in every backend. It denies by default: untagged documents are never returned, and queries without a user see only the lowest level. Re-adding unchanged text with new tags updates its permissions. Conversation memory is kept per user. See README "Access Control".

**Production setup in one command.** `raglib-py --production [PATH] --llm PROVIDER [--docs DIR]` (also `raglib init`) writes a complete `rag_pipeline.py`. Every `RAG(...)` option is on its own line with its production value and a comment: 500-token chunks with 100 overlap, hybrid retrieval of 10 candidates (new `rerank_candidates` option), and a local cross-encoder that keeps the best 3. The command also writes a `.env.example` for your credentials, and the file links to the docs for further customization. The same values are available in code as `RAG(..., preset="production")` / `RAG.production(...)`, and arguments you pass yourself win. New `raglib-py` console command. `raglib presets` lists presets, and `raglib ask/chat/graph` accept `--preset`. The library defaults are unchanged.

**Per-stage latency percentiles.** The pipeline now times `query_rewrite`, `query_embedding`, `dense_search`, `filtering`, `bm25_search`, `fusion`, `rerank`, `prompt_build`, `generate`, `post_process` and a final `total` event. `TraceEvent` has new `parent`, `run_id` and `started_at` fields. `LatencyStats` (pass it as `on_event`) keeps a rolling window per stage and reports count, P50, P95, P99, min and max (linear interpolation). Each sample carries tags: `cache_hit`, `query_tokens`, `num_candidates`, `reranker_used` and `is_cold_start`. `slowest(stage, n)` shows why queries were slow. `benchmark(rag, queries, repeats)` and `benchmarks/latency/bench_latency.py` print the table. Optional `PrometheusExporter` and `OpenTelemetryExporter` (`[prometheus]`, `[otel]` extras), and `on_event` accepts a list of callbacks.
- Existing stage names are kept, and `usage["latency_ms"]` is still the sum of top-level stages. The new `usage["wall_latency_ms"]` is the wall time of the run.
- With the built-in `Generator`, the `generate` event now times only the LLM call: prompt building and citation parsing are the separate `prompt_build` and `post_process` events. This is a few milliseconds less than before. `trace_events` also ends with the new `total` event.

**Caching and invalidation.**
- `llm_cache_ttl` / `embedding_cache_ttl`: entry lifetimes for both caches. Both are now built on `raglib.cache.TTLCache`, an LRU cache that tags each entry with the documents it was built from.
- Every document has a version (a hash of its content and permissions). Re-ingesting a changed document drops the cached LLM responses, answers and vectors built from it.
- `rag.invalidate(doc)`, `rag.invalidate_all()` and `rag.on_document_update(source=..., doc_ids=...)` give event-driven invalidation for external update jobs.
- `answer_cache=True` adds a question-level answer cache (the FAQ tier). An entry is served only while its source documents keep their versions, and a permission change drops it.
- `prompt_caching=True` marks the system prompt with Anthropic `cache_control` (no-op for other providers). Cache read and write tokens are reported in usage.

**Conditional reranking.** `rerank_skip_threshold` / `rerank_skip_margin` skip the reranker when the top candidate's dense cosine similarity is high enough. The fusion score is not used, because RRF and relative-fusion scores are not comparable across queries.

**Retrieval quality.**
- `chunking_strategy="embedding"`: semantic chunking with embeddings, breaking where similarity drops (percentile or fixed cut-off, min/max chunk size). It reuses the embedding cache. The LLM-based `"semantic"` strategy is unchanged.
- `parent_retrieval="heading" | "fixed"`: search small child chunks and send their de-duplicated parent sections to the LLM.
- `cheap_llm` + `tier_router`: route each query to the FAQ tier (answer cache), a cheap model, or the full pipeline. The router is pluggable (`HeuristicTierRouter` by default), and the chosen tier is in `result.metadata["tier"]` and on the trace.

**Cost tracking.**
- `prices=PriceTable(...)` (USD per 1M tokens for LLM input/output and embeddings, per 1K searches for Cohere rerank and web search APIs) adds `cost_usd`, `cost_by_stage`, `cost_by_model` and `cost_by_category` to `result.usage`.
- `CostReport` aggregates cost over many queries and projects a monthly bill, with LLM spend kept separate from vector-DB hosting. `rag.ingest_usage()` reports ingestion cost.
- raglib ships no built-in prices.
- `result.usage` also has `embedding_tokens` and a per-model `usage_by_model`, and `record_usage()` accepts `model=` and cache token counts.

**Evaluation.** `precision@k` sits next to `recall@k` in `raglib.eval`.

**Embedding chunker correctness.**
- No silent fallback: `on_embedding_failure="fallback" | "raise"` (production: `"raise"`). A fallback labels the chunks `recursive_fallback`, emits a `chunking_fallback` warning event and is counted in `ingest_usage()`.
- Pluggable `token_counter` (`"units"` default, `"tiktoken"`, `"auto"`, callable), used by chunk sizes, overlap, hard splits, `semantic_min_tokens` and the embedding cost estimate. Production uses `"auto"`.
- Chunks are exact slices of the original text with `start_char` / `end_char` metadata, so formatting is kept.
- Markdown tables and list runs are single units.
- Sentences don't end after configurable `abbreviations`.
- The double danda `॥` ends a sentence.
- `document_version` is stamped on every chunk and parent section.
- `semantic_min_tokens` is exposed on `RAG(...)`.

**Embedding chunker options.**
- `semantic_buffer_size`: the window centred on each gap. The default of 2 keeps the previous behaviour.
- New threshold types: `standard_deviation` (1.5), `interquartile` (1.5) and `gradient` (95), next to `percentile` and `fixed`. The `standard_deviation` default is 1.5, not LangChain's 3: with the default 2-sentence window, mean + 3 sd found no topic change in the comparison (F1 0.00). With 1.5 it scores F1 0.81-0.88, the best of all types. The README says which threshold type to use when, with benchmark numbers.
- `semantic_number_of_chunks`: an approximate target count.
- `sentence_splitter`: a regex or a callable.
- `semantic_overlap_sentences`: carries the last N sentences of each chunk into the next.
- `benchmarks/chunking/compare_langchain.py` compares the chunker with LangChain's `SemanticChunker` offline.
- The default `semantic_min_tokens` is now `min(32, chunk_size // 8)` instead of `chunk_size // 4`. The larger minimum swallowed short topic sections: boundary F1 was 0.23 at `chunk_size=500` in the comparison, and 0.69 with the new default.
- `gradient` breaks at the distance peak, not one gap before it as the literal rule does.

**Index settings and the token counter.**
- The chunking settings (strategy, sizes, `token_counter`, semantic options, parent sections) are recorded in each saved index (`chunks.json` -> `index.chunking`), and every chunk carries a `chunking_signature`. `rag.chunking_settings()` / `rag.chunking_signature()` return them.
- Opening an index (`load`) or adding to one with different settings raises an `IndexSettingsWarning`. It lists the changed settings and how to re-index: add all sources again with the new settings and `save()` to a new directory, or pass the settings the index was built with.
- **Deprecation:** `token_counter` still defaults to `"units"` (whitespace words). It will become `"auto"` (tiktoken when installed) in the next major version, which changes chunk boundaries and so re-chunks existing indexes. Leaving it unset emits a `DeprecationWarning` once per process when tiktoken is installed. Pass `token_counter="units"` to keep the current behaviour, or `"auto"` to switch now. `preset="production"` and the `raglib-py --production` file already use `"auto"`.

**Benchmarks.**
- `benchmarks/chunking/real_eval.py` runs the chunker on your own documents, embedding model and golden set. It reports recall@k, precision@k, chunk count and average size, and embedding tokens/cost per threshold type and `semantic_min_tokens` (32/64/125). It refuses to call a paid API without `--yes`.
- New `[bench]` extra pins `langchain-experimental==0.4.2` (being sunset) for the comparison script.

**Behaviour change: personalized queries are no longer cached.** With the default `cache_personalized=False`, queries that pass `user=` or `filters=` skip the LLM cache and the answer cache. Answers are identical, but repeated personalized questions are no longer served from cache. Pass `cache_personalized=True` for the old behaviour. The README's "Trade-offs" table lists what every new option costs.

**Security: retrieved text is delimited in every prompt.** Documents, web results and conversation history are wrapped in an unguessable per-prompt tag (`<src-XXXXXXXX>`) in every prompt that includes them, not just the answer prompt. That covers the agent, grader, knowledge strips, LLM reranker, Self-RAG evaluators and revise step, multi-hop planner, refiner, condenser, graph extraction and community reports, global search, LLM chunking and the eval judge. Before, a document containing `</source>` could close its block. Tag-like text inside documents is now escaped, and attribute values such as `name="..."` are HTML-escaped. Every such prompt carries a "data, never instructions" rule; the answer system prompt keeps a stable wording, so provider prompt caching still works. Custom `prompt_template`s without `{data_rule}` get the rule after `{context}` automatically. The tag is an HMAC of the prompt's content under a per-process secret: unpredictable for a document author, and identical for identical requests, so the LLM cache keeps hitting.

**Security: the agent's web search cannot carry retrieved text out.**
- `search_web` queries are rebuilt from the user's question plus model-written keywords that do not occur in any retrieved document, including split-up copies, and are capped at 200 characters (`AgenticRAG(max_web_query_chars=...)`).
- With `access_control`, web search is disabled for a query that retrieved a non-public document. `allow_web_after_private=True` turns it back on, still sanitized.
- Every tool call and its arguments is logged in the `tool` trace stage and in `result.metadata["tool_calls"]`.
- A heuristic flags, but never blocks, sources with instruction-like text, in `result.metadata["injection_flags"]` and an `injection_flag` warning event.

**Embedding model limits: chunks are never silently truncated.**
- Known per-model limits (bge 512, all-MiniLM 256, OpenAI 8191, Gemini 2048, Ollama models, ...), overridable with `embedding_max_tokens`.
- Chunks are counted with the model's tokenizer when available (HuggingFace `tokenizer` / `max_seq_length`, OpenAI via tiktoken), else estimated with a 10% margin.
- At ingest, chunks over the limit are split at sentence borders, with an `EmbeddingLimitWarning`, and counted in `ingest_usage()`. With `oversized_chunks="raise"` (the production preset) they raise `EmbeddingLimitError`. The semantic chunker's comparison windows are kept within the limit too.
- A startup warning appears when `chunk_size` cannot fit the model, with a safe size suggested.
- OpenAI inputs are checked before the API call.
- New `BaseEmbedding.max_input_tokens` / `count_input_tokens()`.

**Reliability fixes.**
- The Cohere reranker has a request timeout (`reranker_timeout`, default 10 s) and the same retries as LLM calls. After the retries it keeps the first-stage order and records the failure in `result.metadata["degraded"]`.
- Embedding models you pass yourself now get `llm_max_retries` instead of the default 2. New `embedding_timeout` for the embedding APIs raglib builds.
- **Behaviour change:** the LLM response cache now expires after 24 hours by default (`llm_cache_ttl=86400`). Before, entries never expired. `llm_cache_ttl=None` keeps the old behaviour.
- The README states that caches, rate limits and invalidation are per process.

**Other changes (not breaking):**
- New optional parameters: `RAG(access_control=)`, `query/aquery/stream(user=)`, `add(access=, metadata=)`, `reset_memory(user=)`. New `rag.permission_filter(user)`.
- `GraphRetriever.retrieve`, `LocalSearch.search`, `GlobalSearch.search` and `WebRetriever.retrieve` accept `filters=`. `GlobalSearch` takes an optional `chunk_lookup`, which it needs to serve filtered queries.
- `BaseVectorStore.supports_metadata_filters` and `BaseSearchProvider.searches_corpus` / `supports_filters` capability flags.
- `matches_filter(..., array_contains=True)` gives list-valued fields "contains" semantics, like Qdrant and pgvector. The default is unchanged.
- Chroma `where` translation is recursive, so nested `$and`/`$or` filters with several keys per clause work.

## 2.2.0 - 2026-11-21

**Benchmark.** raglib's RAG types are now measured on MultiHop-RAG, a public third-party dataset: method, code and every answer are in `benchmarks/multihop_rag/`, and results are on the website's Benchmarks page.

**Fixes found by benchmarking on MultiHop-RAG** (see `benchmarks/multihop_rag/`).
- **Corrective RAG answered from an empty context.** Its grader asked whether each chunk *answers* the question, so for questions whose evidence spans several documents every chunk was graded irrelevant. Without web search the INCORRECT path then had nothing to use: on 30 unseen questions, 9 answers were generated with zero sources. The grader now counts a chunk about any one of the entities, sources or events the question names as partially relevant. Without web search, Corrective RAG falls back to the retrieved chunks instead of an empty context (`result.metadata["crag_fallback"]`). Empty-context answers went from 9/30 to 0/30.
- **Multi-query retrieval lost parts of compound questions.** The expander only paraphrased the whole question, and RRF fusion rewards chunks found by many queries, so for "does source A say X while source B says Y" one article took every slot. It now writes one focused query per entity, source or claim, and the fusion keeps each query's best hit before filling by RRF.
- **GraphContextWindow:** entity detection no longer joins words across line breaks (junk `SHARED_ENTITY` links), and `retrieve()` returns chunks before section summaries, with a `title` on every result.
- **GraphRAG kept holes in the graph after provider errors.** When an extraction call failed (rate limit, timeout, unparsable reply), the chunk was cached and marked indexed with no entities, so a temporary 429 permanently removed its facts from the graph. Failed chunks are now neither cached nor marked indexed. `rag.stats()["graph"]["failed_extractions"]` counts them, `rag.retry_graph_extraction()` extracts them again, and loading a saved index retries them automatically.
- **`graph_llm`:** build the GraphRAG graph with a cheaper or faster model than the answers.
- **GraphRAG answers saw half the evidence.** The entity, relation and community summaries were packed first into the 3,000-token context budget and used about 60% of it, so answers got 2.6 article chunks instead of 5. Questions that name two articles then failed with "the second source is not provided". Documents marked `metadata["supplementary"]` (GraphRAG's summaries) now have their own budget (`Generator(max_supplementary_tokens=1200)`) and can no longer push evidence out of the context.
- **Bounded extraction retries.** A chunk whose extraction keeps failing (some small models loop until `max_tokens` and return an empty reply) was retried, and paid for, on every add and every load. It is now retried at most `graph_kwargs={"max_extraction_attempts": 3}` times in total, then skipped. Its text stays searchable and `stats()["graph"]["skipped_extractions"]` counts it. `retry_graph_extraction(include_skipped=True)` tries again, for example with another model.
- **Truncated JSON is salvaged.** `parse_json` keeps every complete item of a reply cut off at `max_tokens` (`{"entities": [{...}, {...}, {"na` keeps the first two entities) instead of discarding the whole reply. It also no longer returns an inner list when the outer object is truncated.
- **Extraction replies are capped** at `graph_kwargs={"extraction_max_tokens": 4096}`, and the prompt states the entity and relation limits, so a looping model cannot run for minutes on one chunk.
- **Indexes load in seconds.** In-memory vectors (chunks and GraphRAG entities) are saved as binary float32 (`vectors.f32` + `vectors.meta.json`) instead of JSON numbers. A 609-article GraphRAG index went from 1.7 GB and an 81 s load to 402 MB and 2–5 s, with identical search results. Indexes saved by older versions still load, and the next `save()` converts them.
- **Background graph builds.** `RAG(rag_type="graph", graph_build="background")` returns from `add()` and `load()` as soon as chunks are searchable. Queries are answered immediately from hybrid search plus the part of the graph built so far, while one worker extracts in batches and builds community reports when the queue is empty. `rag.graph_status()` reports progress and `rag.wait_for_graph(timeout)` blocks until it is ready. Graph reads now use a read-write lock: queries run concurrently with each other and with community report generation, and are only paused for the short merge of each batch. Entity embedding calls no longer run while the graph is locked.
- **Per-model `json_mode`.** `models=[{"id": "...", "json_mode": False}]` on a custom provider stops raglib from sending `response_format` to models that accept JSON mode but then loop in it. JSON is parsed from the plain reply instead.

- **`SQLTool` broke after its first query on Python 3.9 and 3.10.** Clearing the read-only SQLite authorizer with `None` only works from Python 3.11. Before that, every later statement, `schema()` included, failed with "not authorized".
- **Packaging:** the source distribution no longer picks up README and LICENSE files from `site/node_modules`.
- **Citation labels no longer leak full Windows paths on Linux** (`C:\Users\...\report.pdf` is shown as `report.pdf`).

**LangChain is no longer used to call providers.**
- New native clients in `raglib.llm.clients` speak the three wire protocols directly with the standard library (`urllib`): `OpenAICompatibleClient` (OpenAI, Azure OpenAI, Groq, Ollama, OpenRouter, DeepSeek, Mistral, Together, Fireworks, xAI, Perplexity, Cerebras, LM Studio, vLLM), `AnthropicClient` and `GeminiClient`. They support SSE streaming, JSON mode (OpenAI-compatible and Gemini), provider token usage, retries and rate limiting. `build_chat_client(provider, ...)` builds one.
- Embeddings (OpenAI / Azure / compatible, Gemini, Ollama) call the REST APIs directly; `huggingface` uses `sentence-transformers` without `langchain-huggingface`. Class names are unchanged, so saved indexes still load.
- Vision OCR (OpenAI, Anthropic, Gemini) reuses the native chat clients. DuckDuckGo search uses `ddgs` directly.
- Retries use the HTTP status when there is one (408/409/425/429/5xx), not only the error text. An error event at the start of a stream (e.g. Anthropic `overloaded_error`) is retried.
- Importing raglib and building a client no longer loads LangChain: creating a `RAG` with a hosted provider went from about 4.3 s to 0.45 s.
- The `[openai]`, `[anthropic]`, `[groq]`, `[google]`, `[ollama]`, `[openrouter]` and `[all-llm]` extras are now empty (kept so existing install commands work); `[hf]` is just `sentence-transformers`. New `[langchain]` extra for wrapping your own LangChain model.
**Custom providers.**
- `register_provider("myprovider", api="openai" | "anthropic" | "gemini", base_url=..., api_key="{env:NAME}", headers=..., models=[{"id", "name", "reasoning", "image"}], embedding_model=...)` makes `RAG(llm="myprovider/model")` work for any third-party or self-hosted API. The key is optional (auth can go in headers), and `{env:NAME}` is expanded in keys and header values.
- `load_providers("providers.json")`, the `RAGLIB_PROVIDERS_FILE` environment variable, and `raglib ... --providers FILE` register providers from JSON. `raglib providers` lists them with their models.
- Model flags: `reasoning` stops sending `temperature`; `image` enables OCR of scanned pages with that model. A leading `<think>...</think>` block from reasoning models is removed from answers and streams.
- Custom providers get embeddings through their own API (OpenAI and Gemini protocols) and keep the key-isolation rule: a key is never sent to a different provider.

**NVIDIA NIM.** New built-in provider `nvidia` (alias `nim`, key prefix `nvapi-`, `NVIDIA_API_KEY`), for build.nvidia.com or a self-hosted NIM via `NVIDIA_BASE_URL`. Chat defaults to `nvidia/nemotron-3.5-lightning-30b-a3b`, embeddings to `nvidia/nemotron-3-embed-1b` (with the `input_type` passage/query fields these models require), and OCR of scanned pages to `nvidia/nemotron-3.5-super-vl-120b-a12b`. Tested live: cited answers, streaming with real usage, Corrective RAG grading in JSON mode, hybrid search, and OCR of a scanned PDF.
- Custom providers also accept `extra_body`, `embedding_input_types`, `embedding_params` and `stream_usage`. `vision_llm` works for any built-in provider whose models accept images (e.g. `vision_llm="openrouter/<model>"`).

- `LangChainAdapter` still wraps a LangChain chat model you pass in (`RAG(chat_llm=ChatOpenAI(...))`). `LangChainAdapter.build()` is deprecated and now returns the native client.

## 2.1.0 - 2026-11-06

Fixes and improvements from a full library review.

**Bug fixes**
- **Chunking ignored `chunk_size` for text without sentence punctuation.** The default `fixed` strategy put a 5,000-line file (logs, CSV-like records, one fact per line) into a single 50,000-word chunk, so the model saw only the first part after trimming. Lines are now boundaries, and anything longer than `chunk_size` is split further, for `fixed` and `recursive` alike. Chunks keep their line breaks.
- **Chinese, Japanese and Thai text was not chunked or searchable.** Sizes were counted in space-separated words, so 7,200 characters of Japanese became one chunk, and BM25 indexed whole clauses as single tokens. Each CJK/Thai/Lao/Khmer/Myanmar character now counts as one unit, `。！？` end sentences, keyword search uses character bigrams, the token estimate counts these characters (when tiktoken is missing), and truncation no longer empties such text.
- **A mistyped folder path was indexed as text.** `RAG("./my_docs")` with a missing folder created a document containing `"./my_docs"`. Path-like strings now raise `FileNotFoundError`.
- **A changed embedding model gave meaningless results silently.** Stores now refuse vectors of a different size, and `save()` records the embedding model and dimension; `load()` raises on a mismatch (`allow_model_change=True` re-embeds).
- **DNS rebinding could bypass the URL safety check.** The web loader now connects to the exact address it validated.
- **Conversation memory could raise "deque mutated during iteration"** when one session got concurrent requests; it also kept every turn's retrieved documents (unused), which grows with thousands of sessions. History is read under the lock and documents are not stored unless `store_documents=True`.
- **Streaming had no retries**: a 429 before the first token failed the request. Opening the stream is now retried like `complete`.
- **`rag.chat()` left memory switched on** for later `query()` calls.
- Folder ingestion no longer walks `.git`, `node_modules`, virtualenvs or caches, and skips binary files.
- PyMuPDF 1.24+/2.x: imports `pymupdf` (the `fitz` name is deprecated).

**New**
- `result.usage` reports totals over every LLM call of a query (`llm_calls`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`), using the provider's real counts when LangChain returns them.
- `result.metadata["degraded"]` lists optional steps that failed and fell back (grader, router, reranker, rewrite, web search, tools, ...); `RAG(strict=True)` raises instead.
- `SQLTool`: exact counting/aggregation over CSV/TSV/XLSX/JSON tables for `rag_type="agentic"` (read-only SQLite, timeout, row cap).
- `SQLiteConversationStore` and `RAG(conversation_store=...)`: chat history that survives restarts and is shared by worker processes.
- Vector stores: `vector_db="qdrant"` (`[qdrant]` extra) and `vector_db="pgvector"` (`[pgvector]` extra).
- Loaders: HTML files, CSV/TSV, JSON/JSONL, XLSX (`[xlsx]` extra), EPUB, and source-code/config files. Web pages keep headings, paragraphs, lists and tables, prefer `<main>`/`<article>`, and drop navigation and footers; PDF responses are parsed as PDFs.
- `RAG(llm_temperature=..., llm_max_tokens=...)`.
- JSON mode for OpenAI and OpenAI-compatible providers in `complete_json` (falls back automatically when a model rejects it); the LLM cache and fallback clients keep it.
- Consecutive PDF pages and slides are linked, so `neighbor_window` can follow text across page breaks.

**Performance**
- MMR uses numpy when available.
- `delete` no longer blocks queries while the vector store deletes over the network.
- Chroma queries skip the extra `count()` round trip.

**Tooling**
- GitHub Actions CI: ruff, and pytest on Python 3.9-3.13 (plus Windows, and a pgvector service job).
- Ruff configuration in `pyproject.toml` (Python 3.9 target).

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
