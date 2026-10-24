import { Code } from "../components/Code";
import { METHODS, PARAM_GROUPS, TOTAL_PARAMS, TYPE_OPTIONS, type Param } from "../reference";

const MINIMAL = `
from raglib import RAG

rag = RAG("docs/", llm="openai")     # everything else has a production-ready default
`;

const PRODUCTION = `
import logging
from raglib import RAG

rag = RAG(
    "docs/",
    # models: keys come from the environment / your secret manager, never from code
    llm="openai/gpt-4.1-mini",
    embedding_llm="openai",                       # text-embedding-3-small
    llm_fallbacks=["anthropic/claude-haiku-5-5"], # used if OpenAI is down or rate limited
    llm_timeout=30, llm_max_retries=3, llm_rpm=500,

    # retrieval quality
    retrieval_mode="hybrid", reranker="cross_encoder", top_k=5, neighbor_window=1,
    chunking_strategy="recursive", chunk_size=300, chunk_overlap=40,

    # durable storage shared by all workers
    vector_db="chroma",
    vector_db_kwargs={"host": "chroma.internal", "port": 8000, "collection_name": "handbook"},

    # safety + monitoring
    allowed_dirs=["/srv/docs"],
    on_event=lambda e: logging.info("rag.%s %.0fms", e.stage, e.latency_ms),
)
`;

const PRIVATE = `
# Nothing leaves your network: self-hosted LLM + local embeddings
rag = RAG(
    "docs/",
    llm="vllm/Qwen/Qwen2.5-7B-Instruct",
    chat_base_url="http://gpu-01.internal:8000/v1",   # vLLM, LM Studio or any OpenAI-compatible server
    embedding_llm="huggingface",                      # runs on your CPU/GPU
    vector_db="chroma", vector_db_kwargs={"persist_directory": "/data/chroma"},
)
# or: llm="ollama/llama3.1", embedding_llm="ollama"
`;

const TESTS = `
# tests/conftest.py: the only place offline mode belongs
import pytest

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RAGLIB_OFFLINE", "1")   # unit tests: no keys, no network, deterministic
`;

function ParamTable({ rows }: { rows: Param[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Argument</th>
            <th>Default</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <td><code>{row.name}</code></td>
              <td>{row.default && <code>{row.default}</code>}</td>
              <td className="muted">{row.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Reference() {
  return (
    <div className="wrap docs">
      <nav className="toc" aria-label="On this page">
        <a href="#/reference/overview">Overview</a>
        {PARAM_GROUPS.map((group) => (
          <a key={group.id} href={`#/reference/${group.id}`}>{group.title}</a>
        ))}
        <a href="#/reference/type-options">Per rag_type options</a>
        <a href="#/reference/methods">Methods</a>
        <a href="#/reference/production">Production vs offline</a>
      </nav>

      <div className="doc-body">
        <section id="overview">
          <h2>RAG(...) reference</h2>
          <p>
            <code>RAG(...)</code> takes {TOTAL_PARAMS} arguments. All of them are optional except a model, and every one has a
            sensible default, so a working system is one line:
          </p>
          <Code code={MINIMAL} />
          <p className="muted">
            Arguments are grouped below by what they control. The snippets on the home page show only the few that matter for
            each rag_type; everything here can be combined with any rag_type.
          </p>
        </section>

        {PARAM_GROUPS.map((group) => (
          <section id={group.id} key={group.id}>
            <h2>{group.title}</h2>
            <p className="muted">{group.intro}</p>
            <ParamTable rows={group.params} />
          </section>
        ))}

        <section id="type-options">
          <h2>Per rag_type options</h2>
          <p className="muted">
            Passed directly to <code>RAG(...)</code>, e.g. <code>RAG("docs/", rag_type="multi_hop", max_hops=2)</code>.
          </p>
          <ParamTable rows={TYPE_OPTIONS} />
        </section>

        <section id="methods">
          <h2>Methods</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Returns</th>
                  <th>What it does</th>
                </tr>
              </thead>
              <tbody>
                {METHODS.map((row) => (
                  <tr key={row.name}>
                    <td><code>rag.{row.name}</code></td>
                    <td><code>{row.default}</code></td>
                    <td className="muted">{row.text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="production">
          <h2>Production vs offline mode</h2>
          <p>
            Companies do not run offline mode, and raglib is built so they never can by accident.
          </p>

          <h3>What offline mode is</h3>
          <ul>
            <li>
              <code>offline=True</code> (or <code>RAGLIB_OFFLINE=1</code>) swaps the LLM for a mock that returns templated text
              built from the retrieved chunks, and the embedding model for hashed word features with no real meaning.
            </li>
            <li>No network, no API key, fully deterministic: that makes it right for unit tests, CI pipelines, trying the API, and the
              browser playground on this site.</li>
            <li>Retrieval in offline mode is real (BM25 works fully), but the answers are not generated by a model.</li>
          </ul>

          <h3>Why production can't fall into it</h3>
          <ul>
            <li>
              Without a key and without <code>offline=True</code>, <code>RAG()</code> raises an error at startup. It never silently
              serves mock answers or fake vectors (raglib v1 did; v2 removed that).
            </li>
            <li>If an embedding model can't be resolved, raglib falls back to real BM25 keyword search with a warning, not to fake vectors.</li>
            <li>Keep <code>RAGLIB_OFFLINE</code> out of production environment variables; set it only in test configuration.</li>
          </ul>

          <h3>The four production setups</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Setup</th>
                  <th>Example</th>
                  <th>Where your data goes</th>
                  <th>Choose it when</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Hosted API</td>
                  <td><code>llm="openai"</code>, <code>"anthropic"</code>, <code>"google"</code>, <code>"mistral"</code></td>
                  <td className="muted">Questions and retrieved chunks go to that provider (check its data-retention terms)</td>
                  <td className="muted">Best quality with the least setup</td>
                </tr>
                <tr>
                  <td>Gateway</td>
                  <td><code>llm="openrouter/&lt;model&gt;"</code></td>
                  <td className="muted">The gateway and the model provider it routes to</td>
                  <td className="muted">One key for many models, easy switching and fallbacks</td>
                </tr>
                <tr>
                  <td>Enterprise cloud</td>
                  <td><code>llm="azure/&lt;deployment&gt;"</code></td>
                  <td className="muted">Stays in your cloud tenant and region</td>
                  <td className="muted">Compliance requirements, existing Azure contracts</td>
                </tr>
                <tr>
                  <td>Self-hosted</td>
                  <td><code>llm="vllm/..."</code>, <code>"ollama/..."</code>, <code>embedding_llm="huggingface"</code></td>
                  <td className="muted">Nowhere: everything runs on your servers</td>
                  <td className="muted">Confidential data, air-gapped networks; you run the GPUs</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>A production configuration</h3>
          <Code code={PRODUCTION} />

          <h3>Fully private</h3>
          <Code code={PRIVATE} />

          <h3>Tests: the one place offline mode belongs</h3>
          <Code code={TESTS} />

          <h3>Production checklist</h3>
          <ul>
            <li>API keys in environment variables or a secret manager, never in code; rotate any key that was ever committed.</li>
            <li>A persistent vector store (<code>vector_db="chroma"</code> with a server or persist directory) or <code>rag.save()</code> / <code>rag.load()</code>, so restarts do not re-embed everything.</li>
            <li>One shared <code>RAG</code> per process with a <code>session_id</code> per user; with several workers, a shared vector database.</li>
            <li><code>llm_fallbacks</code>, <code>llm_timeout</code>, <code>llm_max_retries</code> and <code>llm_rpm</code> set for your provider's limits; map <code>QuotaExceededError</code> to HTTP 429.</li>
            <li><code>allowed_dirs</code> set, uploads validated, <code>allow_private_urls</code> left False.</li>
            <li><code>on_event</code> wired to your logging / tracing; watch latency per stage and token usage in <code>result.usage</code>.</li>
            <li>An evaluation set run in CI with <code>raglib.eval.evaluate</code>, so changes to chunking or models can't silently lower answer quality.</li>
            <li>Per-tenant isolation: one <code>RAG</code> per tenant, or a tenant metadata filter on every query.</li>
            <li><code>rag.delete(source)</code> for data-deletion requests (removes chunks, vectors, graph entities and cached text).</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
