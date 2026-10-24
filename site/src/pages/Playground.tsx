import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { loadRuntime, RAGLIB_VERSION, type AskResult, type BuildStats, type RagRuntime } from "../pyodide";
import { LINKS } from "../data";

// rag_types that run fully offline in the browser (web and tool types need network / your own tools).
const TYPES = [
  "advanced", "naive", "hybrid", "corrective", "self", "multi_query", "multi_hop", "agentic", "memory", "routing", "graph",
];
const MODES = ["hybrid", "sparse", "dense"];
const SAMPLE_QUESTIONS = [
  "What is the launch authorization code for Project Bluefin?",
  "Who services the backup generator in Building C?",
  "What does Dr. Okafor snack on during audits?",
  "How much money can I get back for travelling to a conference in Iceland's capital?",
];
const MAX_UPLOAD_BYTES = 2_000_000;

type Status = { kind: "idle" | "loading" | "ready" | "error"; message: string };
type Tab = "answer" | "sources" | "graph";

const STOPWORDS = new Set("the a an of to in for on and or is are was what who how which does do i can my me with from at by be it".split(" "));

/** Highlight the question's words inside a retrieved chunk (React elements, no innerHTML). */
function highlight(text: string, question: string): ReactNode[] {
  const terms = question
    .toLowerCase()
    .split(/[^\p{L}\p{N}-]+/u)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
  if (!terms.length) return [text];
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${escaped.join("|")})`, "giu"));
  return parts.map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : part));
}

export function Playground() {
  const [status, setStatus] = useState<Status>({ kind: "idle", message: "Python runtime not loaded yet." });
  const [text, setText] = useState("");
  const [ragType, setRagType] = useState("advanced");
  const [mode, setMode] = useState("hybrid");
  const [chunkSize, setChunkSize] = useState(60);
  const [topK, setTopK] = useState(4);
  const [stats, setStats] = useState<BuildStats | null>(null);
  const [builtFor, setBuiltFor] = useState("");
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [result, setResult] = useState<AskResult | null>(null);
  const [asked, setAsked] = useState("");
  const [tab, setTab] = useState<Tab>("answer");
  const [graph, setGraph] = useState("");
  const runtimeRef = useRef<RagRuntime | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const settingsKey = `${ragType}|${mode}|${chunkSize}|${topK}|${text.length}|${text.slice(0, 200)}`;
  const stale = stats !== null && builtFor !== settingsKey;
  const busy = status.kind === "loading";

  useEffect(() => {
    fetch(new URL("sample.md", document.baseURI))
      .then((r) => r.text())
      .then((sample) => setText((current) => current || sample))
      .catch(() => undefined);
  }, []);

  async function ensureRuntime(): Promise<RagRuntime | null> {
    if (runtimeRef.current) return runtimeRef.current;
    try {
      runtimeRef.current = await loadRuntime((message) => setStatus({ kind: "loading", message }));
      return runtimeRef.current;
    } catch (error) {
      const reason = String((error as Error).message || error).split("\n").filter(Boolean).pop();
      setStatus({ kind: "error", message: `Could not start Python (${reason}). Check your connection and try again.` });
      return null;
    }
  }

  async function buildIndex(): Promise<boolean> {
    if (!text.trim()) {
      setStatus({ kind: "error", message: "Paste or upload some text first." });
      return false;
    }
    const runtime = await ensureRuntime();
    if (!runtime) return false;
    setStatus({ kind: "loading", message: `Indexing with rag_type="${ragType}"…` });
    await new Promise((r) => setTimeout(r, 30)); // let the status paint before Python blocks the thread
    try {
      const built = runtime.build(text, ragType, mode, chunkSize, topK);
      setStats(built);
      setBuiltFor(settingsKey);
      setResult(null);
      setGraph(ragType === "graph" ? runtime.graphHtml().replace('<html lang="en">', '<html lang="en" data-theme="dark">') : "");
      setStatus({ kind: "ready", message: `Indexed ${built.chunks} chunks in ${built.index_ms} ms. Ask a question.` });
      return true;
    } catch (error) {
      setStatus({ kind: "error", message: `Indexing failed: ${String(error).split("\n").filter(Boolean).pop()}` });
      return false;
    }
  }

  async function askQuestion(q: string = question) {
    if (!q.trim()) return;
    if (!stats || stale) {
      if (!(await buildIndex())) return;
    }
    const runtime = runtimeRef.current;
    if (!runtime) return;
    try {
      const answer = runtime.ask(q);
      if (answer.error) throw new Error(answer.error);
      setResult(answer);
      setAsked(q);
      setTab("answer");
      setStatus({ kind: "ready", message: `Answered in ${answer.ms} ms, entirely in your browser.` });
    } catch (error) {
      setStatus({ kind: "error", message: `Query failed: ${String(error).split("\n").filter(Boolean).pop()}` });
    }
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatus({ kind: "error", message: "Files up to 2 MB of text work best in the browser." });
      return;
    }
    setText(await file.text());
    setStats(null);
    setResult(null);
    setStatus({ kind: status.kind === "error" ? "idle" : status.kind, message: `Loaded ${file.name}. Build the index or just ask.` });
  }

  const maxScore = useMemo(() => Math.max(1e-9, ...(result?.sources.map((s) => s.score) ?? [0])), [result]);

  return (
    <div className="wrap">
      <div style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: "2rem" }}>Playground</h1>
        <p className="lede">
          The real raglib {RAGLIB_VERSION} running in your browser through Pyodide. Your text never leaves this page.
          There is no API key and no server. Instead of an LLM, the answer is the best-matching sentence from what retrieval found, so you see exactly how good the retrieval is.
        </p>
      </div>

      <div className="pg">
        {/* ------------------------------------------------------------ input */}
        <div className="panel">
          <div className="panel-head">
            <span className="label">1 · Document</span>
            <div className="row">
              <button className="btn ghost small" type="button" onClick={() => fileRef.current?.click()}>
                Upload .txt / .md
              </button>
              <button
                className="btn ghost small"
                type="button"
                onClick={async () => {
                  setText(await fetch(new URL("sample.md", document.baseURI)).then((r) => r.text()));
                  setStats(null);
                }}
              >
                Sample
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".txt,.md,.markdown,text/plain,text/markdown"
                hidden
                onChange={(e) => onUpload(e.target.files?.[0])}
              />
            </div>
          </div>
          <div className="panel-body">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              spellCheck={false}
              aria-label="Document text"
              placeholder="Paste any text here…"
            />
            <p className="dim mono" style={{ fontSize: ".75rem", margin: "6px 0 0" }}>
              {text.length.toLocaleString()} characters. The sample is a 105-section company handbook with 4 hidden facts and 16 look-alike decoys.
            </p>
            <div className="fields">
              <div className="field">
                <label className="label" htmlFor="rag-type">rag_type</label>
                <select id="rag-type" value={ragType} onChange={(e) => setRagType(e.target.value)}>
                  {TYPES.map((t) => (
                    <option key={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label className="label" htmlFor="mode">retrieval_mode</label>
                <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
                  {MODES.map((m) => (
                    <option key={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label className="label" htmlFor="chunk">chunk_size (words)</label>
                <input id="chunk" type="number" min={20} max={400} value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value) || 60)} />
              </div>
              <div className="field">
                <label className="label" htmlFor="topk">top_k</label>
                <input id="topk" type="number" min={1} max={10} value={topK} onChange={(e) => setTopK(Math.min(10, Math.max(1, Number(e.target.value) || 4)))} />
              </div>
            </div>
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn" type="button" disabled={busy} onClick={buildIndex}>
                {stats && !stale ? "Rebuild index" : "Build index"}
              </button>
              {stale && <span className="dim mono" style={{ fontSize: ".78rem" }}>settings changed, will rebuild on ask</span>}
            </div>
          </div>
          <div className={`status ${status.kind === "ready" ? "ready" : ""} ${status.kind === "error" ? "error" : ""}`} role="status">
            <span className="dot" />
            {status.message}
          </div>
        </div>

        {/* ------------------------------------------------------------ output */}
        <div className="panel">
          <div className="panel-head">
            <span className="label">2 · Ask</span>
            {stats && (
              <span className="dim mono" style={{ fontSize: ".75rem" }}>
                {stats.chunks} chunks · {stats.rag_type} · {stats.retrieval_mode}
              </span>
            )}
          </div>
          <div className="panel-body">
            <form
              className="ask"
              onSubmit={(e) => {
                e.preventDefault();
                askQuestion();
              }}
            >
              <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} aria-label="Question" placeholder="Ask about the document…" />
              <button className="btn" type="submit" disabled={busy}>
                Ask
              </button>
            </form>
            <div className="samples">
              {SAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setQuestion(q);
                    askQuestion(q);
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          <div className="tabs" role="tablist" style={{ borderLeft: 0, borderRight: 0 }}>
            {(["answer", "sources", "graph"] as Tab[]).map((t) => (
              <button key={t} role="tab" type="button" aria-selected={tab === t} onClick={() => setTab(t)}>
                {t === "sources" && result ? `sources (${result.sources.length})` : t}
              </button>
            ))}
          </div>

          {tab === "answer" &&
            (result ? (
              <>
                <div className="panel-body">
                  <p className="label">Extractive answer: best-matching sentence from the sources, no LLM</p>
                  <p className="answer">{result.answer}</p>
                  <p className="label" style={{ marginTop: 18 }}>Pipeline trace</p>
                  <div className="trace">
                    {result.trace.map((step, i) => (
                      <span key={i}>{step}</span>
                    ))}
                  </div>
                </div>
                <div className="kv">
                  <div><span className="label">query</span><b>{result.ms} ms</b></div>
                  <div><span className="label">index</span><b>{stats?.index_ms} ms</b></div>
                  <div><span className="label">chunks</span><b>{stats?.chunks}</b></div>
                  <div><span className="label">sources</span><b>{result.sources.length}</b></div>
                </div>
              </>
            ) : (
              <p className="empty">Ask a question to see the answer, the sources it used and the pipeline trace.</p>
            ))}

          {tab === "sources" &&
            (result && result.sources.length ? (
              <div>
                {result.sources.map((source, i) => (
                  <div className="source" key={source.id}>
                    <div className="source-head">
                      <span className="rank">[{i + 1}]{result.citations.includes(i + 1) ? " cited" : ""}</span>
                      <span>score {source.score.toFixed(3)}</span>
                    </div>
                    <p>{highlight(source.content, asked)}</p>
                    <div className="bar"><i style={{ width: `${Math.max(4, (source.score / maxScore) * 100)}%` }} /></div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty">No sources yet.</p>
            ))}

          {tab === "graph" &&
            (graph ? (
              <iframe className="graph-frame" title="Knowledge graph" sandbox="allow-scripts" srcDoc={graph} />
            ) : (
              <p className="empty">Choose rag_type "graph" and build the index to explore the knowledge graph.</p>
            ))}
        </div>
      </div>

      <section style={{ paddingTop: 0 }}>
        <div className="grid c3">
          <div className="card">
            <span className="label">What is real here</span>
            <p>Chunking, BM25 and hybrid retrieval, fusion and every pipeline stage are the actual raglib code from PyPI.</p>
          </div>
          <div className="card">
            <span className="label">What is simulated</span>
            <p>
              There is no LLM, and "dense" uses hashing vectors instead of a real embedding model, so the paraphrased question about
              Iceland's capital fails here. With a real embedding model raglib finds it at rank 1.
            </p>
          </div>
          <div className="card">
            <span className="label">Use it for real</span>
            <p>
              <code>pip install raglib-py</code> and pass <code>llm="provider/model"</code>. See the <a href="#/docs">docs</a> or{" "}
              <a href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
