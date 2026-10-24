import type { ReactNode } from "react";
import { Code, Install } from "../components/Code";
import { LINKS } from "../data";

const TOC: [string, string][] = [
  ["install", "Install"],
  ["quickstart", "Quickstart"],
  ["models", "Models & providers"],
  ["free", "Free models & rate limits"],
  ["documents", "Adding documents"],
  ["retrieval", "Retrieval options"],
  ["rag-types", "Choosing a RAG type"],
  ["graphrag", "GraphRAG"],
  ["context-window", "Graph context window"],
  ["chat", "Chat, streaming, async"],
  ["result", "Reading the result"],
  ["fastapi", "FastAPI"],
  ["evaluation", "Evaluation"],
  ["security", "Security & privacy"],
  ["scaling", "Scaling"],
  ["cli", "Command line"],
  ["reference", "Full reference"],
];

function Doc({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function Docs() {
  return (
    <div className="wrap docs">
      <nav className="toc" aria-label="On this page">
        {TOC.map(([id, label]) => (
          <a key={id} href={`#/docs/${id}`}>
            {label}
          </a>
        ))}
      </nav>

      <div className="doc-body">
        <Doc id="install" title="Install">
          <p>The core has no required dependencies. Add extras for the integrations you use.</p>
          <Install />
          <Code
            lang="shell"
            code={`
pip install "raglib-py[openai]"     # OpenAI, OpenRouter and every OpenAI-compatible API
pip install "raglib-py[pdf,docx]"   # document loaders
pip install "raglib-py[graph]"      # GraphRAG (networkx)
pip install "raglib-py[chroma]"     # persistent vector store
pip install "raglib-py[all]"        # everything
`}
          />
          <p className="muted">
            Other extras: <code>anthropic</code>, <code>groq</code>, <code>google</code>, <code>ollama</code>, <code>hf</code>,{" "}
            <code>rerank</code>, <code>cohere</code>, <code>leiden</code>, <code>pptx</code>, <code>ocr</code>, <code>web</code>,{" "}
            <code>tavily</code>, <code>serpapi</code>, <code>fast</code>.
          </p>
        </Doc>

        <Doc id="quickstart" title="Quickstart">
          <Code
            code={`
from raglib import RAG

rag = RAG("docs/", llm="openrouter/openai/gpt-4.1-mini")
result = rag.query("What is the refund window?")

print(result.answer)                     # "Refunds are accepted within 30 days [1]."
for c in result.citations:
    print(c.index, c.source, c.snippet)  # which chunk backs each [n]
`}
          />
          <p>No API key yet? Everything runs offline with a deterministic mock model:</p>
          <Code code={`rag = RAG("docs/", offline=True)   # or set RAGLIB_OFFLINE=1`} />
        </Doc>

        <Doc id="models" title="Models & providers">
          <p>
            Pass <code>llm="provider/model"</code>. With no <code>llm</code>, raglib uses the first provider whose API key is set in
            the environment.
          </p>
          <Code
            code={`
RAG("docs/", llm="openai")                                   # OPENAI_API_KEY
RAG("docs/", llm="anthropic/claude-haiku-5-5")              # ANTHROPIC_API_KEY
RAG("docs/", llm="openrouter/google/gemini-2.5-flash")      # OPENROUTER_API_KEY
RAG("docs/", llm="ollama/llama3.1", embedding_llm="ollama") # fully local
RAG("docs/", llm="deepseek", embedding_llm="huggingface")   # local embeddings
`}
          />
          <p className="muted">
            17 providers: openai, anthropic, google, groq, ollama, openrouter, deepseek, mistral, together, fireworks, xai,
            perplexity, cerebras, lmstudio, vllm, openai_compatible, azure. Run <code>raglib providers</code> to list them with
            their environment variables.
          </p>
          <div className="note">
            <strong>Embeddings.</strong> If <code>embedding_llm</code> is omitted, raglib uses the chat provider's embeddings
            when it has them, else local HuggingFace if installed, else BM25 keyword search with a warning. A chat key is only
            reused for embeddings from the same provider.
          </div>
        </Doc>

        <Doc id="free" title="Free models & rate limits">
          <Code
            code={`
rag = RAG(
    "docs/",
    llm="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    llm_fallbacks=["openrouter/cohere/north-mini-code:free"],
    embedding_llm="openrouter/nvidia/nemotron-3-embed-1b:free",
)
`}
          />
          <ul>
            <li>
              <code>llm_rpm</code> caps requests per minute. Chat and embedding calls to the same provider and key share one
              budget. OpenRouter <code>:free</code> models default to 20/min.
            </li>
            <li>Rate limits and 5xx errors are retried, waiting as long as the server asks (<code>Retry-After</code>).</li>
            <li>
              Daily caps and out-of-credit errors raise <code>QuotaExceededError</code> at once instead of retrying.
            </li>
            <li>
              <code>llm_fallbacks</code> tries the next model when one is down.
            </li>
          </ul>
          <div className="note">
            Free model lists change often. An OpenRouter free account allows about 50 free-model requests per day.
          </div>
        </Doc>

        <Doc id="documents" title="Adding documents">
          <p>
            Files (<code>.pdf .docx .pptx .txt .md</code>), folders, web URLs, raw text and <code>Document</code> objects.
          </p>
          <Code
            code={`
rag = RAG(llm="openai")
rag.add("handbook/", batch_size=256, on_progress=print)   # streams large folders, resumes after failures
rag.add("https://example.com/policy")                    # private / metadata IPs are blocked
rag.update("handbook/pricing.md")                         # re-ingest a changed file
rag.delete("handbook/old.pdf")                            # removes chunks, vectors, graph entities, caches
rag.save("./index"); rag.load("./index")                  # persist without re-embedding
`}
          />
          <p className="muted">
            <code>add()</code> is idempotent: unchanged documents are skipped. Chunking: <code>chunking_strategy</code> ={" "}
            <code>"fixed"</code> (default), <code>"recursive"</code> or <code>"llm"</code>, with <code>chunk_size</code> and{" "}
            <code>chunk_overlap</code>.
          </p>
        </Doc>

        <Doc id="retrieval" title="Retrieval options">
          <Code
            code={`
rag = RAG(
    "docs/", llm="openai",
    retrieval_mode="hybrid",      # "hybrid" (default) | "dense" | "sparse"
    top_k=5,
    reranker="cross_encoder",     # "none" | "cross_encoder" | "cohere" | "llm"
    mmr_lambda=0.5,               # diversity
    neighbor_window=1,            # add neighbouring chunks (small-to-big)
)
rag.query("refund policy", filters={"file": {"$in": ["docs/policy_2026.md"]}})
`}
          />
          <p className="muted">
            Filters support <code>$eq $ne $in $nin $gt $gte $lt $lte $and $or</code>. Hybrid search scores every candidate on
            both BM25 and vector similarity, so exact codes and paraphrases both match.
          </p>
        </Doc>

        <Doc id="rag-types" title="Choosing a RAG type">
          <p>
            Set <code>rag_type</code>. All 14 share one pipeline, so citations, memory, filters and tracing work the same in each.{" "}
            <a href="#/">See every type with its cost in LLM calls</a>.
          </p>
          <ul>
            <li><code>advanced</code> (default): most document Q&amp;A.</li>
            <li><code>corrective</code> / <code>self</code>: when answers must be checked against the sources.</li>
            <li><code>multi_hop</code> / <code>agentic</code>: questions that chain several facts.</li>
            <li><code>memory</code>: chatbots (follow-ups are rewritten into standalone questions).</li>
            <li><code>web</code> / <code>hybrid_web</code>: fresh facts; pass <code>web_search_provider="duckduckgo"</code>.</li>
            <li><code>tool</code>: your own tools. Pass <code>tools=[Tool(name, description, func)]</code>.</li>
            <li><code>graph</code>: relationships and corpus-wide themes.</li>
          </ul>
        </Doc>

        <Doc id="graphrag" title="GraphRAG">
          <p>
            One LLM call per chunk extracts entities and relations (in parallel, cached). Communities are built when first
            needed. Local search walks the graph from matching entities with no extra LLM calls; global search summarizes the
            most relevant communities.
          </p>
          <Code
            code={`
rag = RAG("papers/", rag_type="graph", llm="openai", search_mode="auto")   # "local" | "global"
rag.query("Which datasets did the YOLO authors also use?")    # local: entity -> hops -> chunks
rag.query("What are the main themes across these papers?")   # global: community reports
rag.visualize_graph("graph.html")                             # interactive, self-contained HTML
`}
          />
          <p className="muted">
            Neo4j: <code>graph_store="neo4j"</code> with <code>graph_kwargs=&#123;"store_options": &#123;...&#125;&#125;</code>.
            Merge near-duplicate entities with <code>graph_kwargs=&#123;"resolve_with_embeddings": True&#125;</code>.
          </p>
        </Doc>

        <Doc id="context-window" title="Graph context window">
          <p>
            A lightweight, LLM-optional graph of Document → Section → Chunk plus chat turns, for assembling conversational
            context. Add <code>embedding=</code> so paraphrased questions still find the right chunk.
          </p>
          <Code
            code={`
from raglib.graph_context.engine import GraphContextWindow

graph = GraphContextWindow(chunking_strategy="recursive", embedding=my_embedding)
graph.add_documents(docs)
window = graph.build_context_window("How do I reset my password?")
print(window.formatted_context)
graph.to_html("context_graph.html")
`}
          />
        </Doc>

        <Doc id="chat" title="Chat, streaming, async">
          <Code
            code={`
rag = RAG("docs/", llm="openai", rag_type="memory")
rag.query("What is our auth service called?", session_id="user-42")
rag.query("Which services depend on it?", session_id="user-42")   # history is per session

for token in rag.stream("Summarize the onboarding guide"):
    print(token, end="")

result = await rag.aquery("...", session_id=request.user_id)       # FastAPI etc.
`}
          />
          <p className="muted">
            One <code>RAG</code> instance is thread-safe: give each chat window its own <code>session_id</code>. Trace every stage
            with <code>on_event=lambda e: print(e.stage, e.latency_ms)</code>.
          </p>
        </Doc>

        <Doc id="result" title="Reading the result">
          <p>Every rag_type returns a <code>GenerationResult</code>:</p>
          <ul>
            <li><code>answer</code>: the text, with <code>[n]</code> citation markers.</li>
            <li><code>citations</code>: <code>index</code>, <code>source</code> (file name and page), <code>snippet</code> for each <code>[n]</code>.</li>
            <li><code>sources</code>: every chunk the LLM saw (<code>id</code>, <code>content</code>, <code>score</code>, <code>metadata</code>).</li>
            <li><code>reasoning_trace</code>: readable pipeline steps, e.g. <code>evaluation:correct</code>.</li>
            <li><code>trace_events</code>: timed stages (<code>stage</code>, <code>latency_ms</code>, <code>meta</code>).</li>
            <li><code>usage</code>: prompt and completion tokens, total latency.</li>
            <li>
              <code>metadata</code>: pipeline diagnostics such as <code>crag_verdict</code>, <code>agent_steps</code>,{" "}
              <code>hop_queries</code>, <code>graph_mode</code>, <code>self_rag_llm_calls</code>, and the standalone{" "}
              <code>search_query</code>.
            </li>
          </ul>
          <p className="muted">
            <a href="#/">The home page</a> has a snippet for each rag_type, and{" "}
            <a href={`${LINKS.github}/blob/main/examples/rag_types_tour.py`} target="_blank" rel="noreferrer">examples/rag_types_tour.py</a>{" "}
            runs all of them.
          </p>
        </Doc>

        <Doc id="fastapi" title="FastAPI">
          <p>Create one <code>RAG</code> at startup and share it. It is thread-safe; use <code>aquery</code> in async endpoints.</p>
          <Code
            code={`
@app.post("/ask")
async def ask(req: Ask):
    r = await rag.aquery(req.question, session_id=req.session_id, filters=req.filters)
    return {"answer": r.answer, "citations": [c.__dict__ for c in r.citations], "metadata": r.metadata}

@app.post("/documents")
async def upload(file: UploadFile):
    ...                                                    # save the file, then index it off the event loop
    return {"chunks_added": await run_in_threadpool(rag.add, path)}
`}
          />
          <ul>
            <li>Map <code>QuotaExceededError</code> to HTTP 429 and <code>ValueError</code> to 400.</li>
            <li>Stream with <code>StreamingResponse(rag.stream(question, session_id=...))</code>.</li>
            <li>Set <code>allowed_dirs</code> and validate upload names so users cannot read or write outside your folder.</li>
            <li>Several uvicorn workers each hold their own index: use a shared vector database (e.g. Chroma server) in that case.</li>
          </ul>
          <p className="muted">
            Complete, tested app with uploads, deletion, streaming and a health check:{" "}
            <a href={`${LINKS.github}/blob/main/examples/fastapi_app.py`} target="_blank" rel="noreferrer">examples/fastapi_app.py</a>.
          </p>
        </Doc>

        <Doc id="evaluation" title="Evaluation">
          <Code
            code={`
from raglib.eval import evaluate

dataset = [{"question": "What is the refund window?", "relevant_ids": ["docs/refunds.md"], "reference": "30 days"}]
print(evaluate(rag, dataset, judge_llm=judge, k=5))
# hit_rate@5, mrr@5, recall@5, ndcg@5, faithfulness, answer_relevance, context_precision, correctness
`}
          />
        </Doc>

        <Doc id="security" title="Security & privacy">
          <ul>
            <li>Questions, retrieved chunks and chat history go to the LLM you choose. Use Ollama, LM Studio or vLLM for confidential data.</li>
            <li>Questions, answers and document text are never logged at INFO or above.</li>
            <li>URL sources block private, loopback and cloud-metadata addresses, re-check redirects and cap responses at 20 MB.</li>
            <li>Sandbox user-supplied paths with <code>allowed_dirs=[...]</code>.</li>
            <li><code>delete()</code> removes chunks, vectors, graph entities and cached copies of the text.</li>
            <li>Citations show file names, never full local paths.</li>
            <li>For multiple tenants, use one <code>RAG</code> per tenant or a tenant filter on every query.</li>
          </ul>
        </Doc>

        <Doc id="scaling" title="Scaling">
          <ul>
            <li>The in-memory store uses about 1 GB per 30,000 chunks (1536-dim). Above ~100k chunks use <code>vector_db="chroma"</code>.</li>
            <li>Queries take a read lock and run in parallel; <code>add</code> / <code>delete</code> take a short write lock.</li>
            <li>Several server processes need a shared vector database and chat store.</li>
          </ul>
          <Code code={`rag = RAG("docs/", llm="openai", vector_db="chroma", vector_db_kwargs={"persist_directory": "./chroma"})`} />
        </Doc>

        <Doc id="cli" title="Command line">
          <Code
            lang="shell"
            code={`
raglib ask docs/ "What changed in v2?"
raglib chat docs/ --llm openrouter/openai/gpt-4.1-mini
raglib graph docs/ --out graph.html
raglib providers
`}
          />
        </Doc>

        <Doc id="reference" title="Full reference">
          <p>Every constructor option, schema and component is documented on GitHub.</p>
          <div className="row">
            <a className="btn" href={LINKS.readme} target="_blank" rel="noreferrer">README</a>
            <a className="btn ghost" href={LINKS.usage} target="_blank" rel="noreferrer">USAGE.md</a>
            <a className="btn ghost" href={LINKS.changelog} target="_blank" rel="noreferrer">Changelog</a>
          </div>
        </Doc>
      </div>
    </div>
  );
}
