/** Result rows and extra measurements written by `bench.py site`. */
export type ConfigResult = {
  name: string;
  rag_type: string;
  settings: Record<string, unknown>;
  about: string;
  questions: number;
  errors: number;
  accuracy: number | null;
  accuracy_judged_n: number;
  accuracy_by_type: Record<string, number | null>;
  counts_by_type: Record<string, [number, number]>;
  strict_accuracy: number | null;
  evidence_recall: number | null;
  latency_s_median: number | null;
  llm_calls_mean: number | null;
  tokens_mean: number | null;
  judge: Record<string, number | null>;
};

type Side = { accuracy: number | null; accuracy_judged_n: number; evidence_recall: number | null; strict_accuracy: number | null };
export type Fix = { config: string; main: { before: Side; after: Side } | null; heldout: { before: Side; after: Side } | null };

export type Extras = {
  fixes?: Fix[];
  heldout_reference?: number | null;
  answer_model?: {
    config: string;
    free: { model: string; cost_inr_30: number };
    paid: { model: string; cost_inr_30: number; cost_inr_per_answer: number };
    free_result?: ConfigResult | null;
    paid_result?: ConfigResult | null;
  };
  graph_build?: {
    model: string; via: string; cost_inr: number; note: string; nvidia_free_tier: string;
    llm_calls?: number; tokens?: number; seconds?: number;
    entities?: number; relations?: number; communities?: number; skipped_extractions?: number;
  };
  load?: { json_s: number; binary_s: number; json_mb: number; binary_mb: number };
  storage?: {
    articles: number; chunks: number; embedding_dim: number; raw_mb: number;
    hybrid: { disk_mb: number; ram_mb: number };
    graph: { disk_mb: number; ram_mb: number; entity_vectors_mb: number };
  };
};

export const pct = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "–" : `${(value * 100).toFixed(digits)}%`;
const num = (value: number | undefined) => (value === undefined ? "–" : value.toLocaleString("en-IN"));

function Head({ n, title }: { n: string; title: string }) {
  return <div className="section-head"><span className="num">{n}</span><h2>{title}</h2></div>;
}

/* ------------------------------------------------------------------ 01 which RAG */

const CHOICES: { need: string; use: string; code: string; why: string }[] = [
  {
    need: "Most apps: start here",
    use: "advanced_rerank",
    code: 'RAG(docs, rag_type="advanced", reranker="cross_encoder")',
    why: "Top accuracy with one LLM call per question. The reranker runs locally and costs nothing per query.",
  },
  {
    need: "Questions that compare or combine several documents",
    use: "multi_query",
    code: 'RAG(docs, rag_type="multi_query")',
    why: "Writes one search per entity or source in the question, so it finds the most evidence of any type.",
  },
  {
    need: "Step-by-step questions (find A, then use A to find B)",
    use: "multi_hop",
    code: 'RAG(docs, rag_type="multi_hop")',
    why: "Retrieves again after each reasoning step. Top accuracy, about 3 LLM calls.",
  },
  {
    need: "Your documents may not contain the answer",
    use: "corrective",
    code: 'RAG(docs, rag_type="corrective", web_search_provider="duckduckgo")',
    why: "Grades what it retrieved and falls back to the web. Highest score on unseen questions too, but slow (about 5 calls).",
  },
  {
    need: "Lowest cost and latency",
    use: "advanced",
    code: 'RAG(docs)  # the default: BM25 + vector hybrid',
    why: "One call, no extra models. A few points below the reranked version.",
  },
  {
    need: "Relationships and big-picture themes across a corpus",
    use: "graph",
    code: 'RAG(docs, rag_type="graph", graph_llm="<cheap model>", graph_build="background")',
    why: "Fastest at query time and good on inference questions, but not better than hybrid search on these questions, and it needs a one-off graph build.",
  },
];

const AVOID: [string, string][] = [
  ["self", "lowest accuracy of the full pipelines here, with the most LLM calls (6–7 per question)"],
  ["graph_context", "finds less than half the evidence the hybrid types find; use it only when you cannot afford any LLM calls at indexing"],
  ["naive", "dense-only search misses exact names and dates that BM25 catches"],
];

export function WhichRag({ byName }: { byName: Map<string, ConfigResult> }) {
  return (
    <section>
      <div className="wrap">
        <Head n="01" title="Which RAG should you use?" />
        <p className="muted bench-explain">
          Guidance from the results below. Accuracy is on the same 30 questions; with this sample, differences of one or
          two questions (3–7 points) are not reliable, so cost and latency matter as much as the ranking.
        </p>
        <div className="table-wrap">
          <table className="rec-table">
            <thead>
              <tr>
                <th scope="col">If you need</th>
                <th scope="col">Use</th>
                <th scope="col">Accuracy</th>
                <th scope="col">LLM calls</th>
                <th scope="col">Why</th>
              </tr>
            </thead>
            <tbody>
              {CHOICES.map((choice) => {
                const result = byName.get(choice.use);
                return (
                  <tr key={choice.use}>
                    <td>{choice.need}</td>
                    <td><code>{choice.use}</code><div className="dim rec-code">{choice.code}</div></td>
                    <td className="mono">{pct(result?.accuracy)}</td>
                    <td className="mono">{result?.llm_calls_mean?.toFixed(1) ?? "–"}</td>
                    <td className="muted bench-about">{choice.why}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <h3 className="bench-sub">Not recommended for this kind of data</h3>
        <ul className="list">
          {AVOID.map(([name, reason]) => (
            <li key={name}>
              <span className="mark">–</span>
              <span><strong>{name}</strong> <span className="mono dim">{pct(byName.get(name)?.accuracy)}</span><span className="desc">{reason}</span></span>
            </li>
          ))}
        </ul>
        <p className="note rec-note">
          <strong>Improve retrieval before paying for a bigger answer model.</strong> Swapping the answer model for a paid
          one left accuracy unchanged (section 06), while adding a free local reranker gained a question. Every
          wrong GraphRAG answer we read was a retrieval miss, not a reasoning error.
        </p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ 05 fixes */

const FIX_TEXT: Record<string, string> = {
  corrective: "Answered from an empty context: for multi-document questions every chunk was graded irrelevant (9 of 30 answers had no sources).",
  multi_query: "Only paraphrased the whole question, so for “does A say X while B says Y” one article took every slot.",
  graph: "Graph summaries filled ~60% of the context budget, so answers saw 2.6 evidence chunks instead of 5.",
  graph_context: "Entity detection joined words across line breaks, and results were padded with section summaries.",
};

function Change({ side }: { side?: { before: Side; after: Side } | null }) {
  if (!side) return <span className="dim">not run</span>;
  const delta = (side.after.accuracy ?? 0) - (side.before.accuracy ?? 0);
  return (
    <span className="mono">
      {pct(side.before.accuracy)} → <b>{pct(side.after.accuracy)}</b>{" "}
      <span className={delta > 0 ? "delta up" : delta < 0 ? "delta down" : "delta"}>
        {delta > 0 ? "+" : ""}{Math.round(delta * 100)}
      </span>
    </span>
  );
}

export function Fixes({ fixes, reference }: { fixes: Fix[]; reference?: number | null }) {
  if (!fixes.length) return null;
  return (
    <section>
      <div className="wrap">
        <Head n="05" title="What the benchmark found, and what we fixed" />
        <p className="muted bench-explain">
          Each bug was found on the first 30 questions, fixed in raglib, then checked on 30 different questions the fix
          was never tuned on (held-out, seed 7, no overlap). The old code ran from a snapshot, so “before” is the real
          previous behaviour. On the held-out set, the unchanged <code>advanced</code> scored {pct(reference)}.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Config</th>
                <th scope="col">What was wrong</th>
                <th scope="col">First 30 questions</th>
                <th scope="col">Held-out 30</th>
                <th scope="col">Evidence found</th>
              </tr>
            </thead>
            <tbody>
              {fixes.map((fix) => (
                <tr key={fix.config}>
                  <td><code>{fix.config}</code></td>
                  <td className="muted bench-about">{FIX_TEXT[fix.config] ?? ""}</td>
                  <td><Change side={fix.main} /></td>
                  <td><Change side={fix.heldout} /></td>
                  <td className="mono">
                    {fix.main ? `${pct(fix.main.before.evidence_recall)} → ${pct(fix.main.after.evidence_recall)}` : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="dim bench-foot">
          Numbers are the share of questions judged correct; +N is the change in points. The full list of fixes is in the
          changelog.
        </p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ 06 answer model */

export function AnswerModel({ info }: { info?: Extras["answer_model"] }) {
  if (!info?.free_result || !info.paid_result) return null;
  const rows: [string, string, string][] = [
    ["Judged accuracy", pct(info.free_result.accuracy), pct(info.paid_result.accuracy)],
    ["Strict match", pct(info.free_result.strict_accuracy), pct(info.paid_result.strict_accuracy)],
    ["Comparison questions", (info.free_result.counts_by_type.comparison_query ?? []).join("/"), (info.paid_result.counts_by_type.comparison_query ?? []).join("/")],
    ["Temporal questions", (info.free_result.counts_by_type.temporal_query ?? []).join("/"), (info.paid_result.counts_by_type.temporal_query ?? []).join("/")],
    ["Median time per answer", `${info.free_result.latency_s_median?.toFixed(0)} s`, `${info.paid_result.latency_s_median?.toFixed(0)} s`],
    ["Cost for 30 questions", `₹${info.free.cost_inr_30}`, `₹${info.paid.cost_inr_30.toFixed(2)} (₹${info.paid.cost_inr_per_answer.toFixed(2)} each)`],
  ];
  return (
    <section>
      <div className="wrap">
        <Head n="06" title="Does a paid answer model help?" />
        <p className="muted bench-explain">
          Same retrieval, same index, same judge; only the model that writes the answer changes, on{" "}
          <code>{info.config}</code>. The paid model fixed two answers and hedged on two others, so accuracy did not move.
          It answers about four times faster.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col"></th>
                <th scope="col">Free: <code>{info.free.model}</code></th>
                <th scope="col">Paid: <code>{info.paid.model}</code></th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, free, paid]) => (
                <tr key={label}><th scope="row">{label}</th><td className="mono">{free}</td><td className="mono">{paid}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ 07 graph cost and storage */

export function GraphCost({ build, load, storage }: { build?: Extras["graph_build"]; load?: Extras["load"]; storage?: Extras["storage"] }) {
  if (!build) return null;
  const hours = build.seconds ? (build.seconds / 3600).toFixed(1) : "–";
  const tiles: [string, string][] = [
    ["Entities", num(build.entities)],
    ["Relations", num(build.relations)],
    ["Communities", num(build.communities)],
    ["LLM calls", num(build.llm_calls)],
    ["Tokens", build.tokens ? `${(build.tokens / 1e6).toFixed(1)}M` : "–"],
    ["Build time", `${hours} h`],
    ["Cost", `≈ ₹${build.cost_inr}`],
    ["Chunks skipped", num(build.skipped_extractions)],
  ];
  return (
    <section>
      <div className="wrap">
        <Head n="07" title="GraphRAG: build cost, loading and storage" />
        <p className="muted bench-explain">
          GraphRAG needs a knowledge graph before it can answer: one LLM call per chunk to extract entities and relations,
          plus one per community summary. It is built once and saved; loading it makes no LLM calls. Built with{" "}
          <code>{build.model}</code> through {build.via}; answers still use the same generator as every other config.
        </p>
        <div className="grid c4 bench-tiles">
          {tiles.map(([label, value]) => (
            <div key={label} className="card"><span className="label">{label}</span><p className="tile-value">{value}</p></div>
          ))}
        </div>
        <ul className="list bench-sub">
          <li><span className="mark">·</span><span>{build.nvidia_free_tier}.</span></li>
          <li><span className="mark">·</span><span>About 9% of chunks returned an empty reply from the small extraction model even after a retry; they stay searchable but add nothing to the graph.</span></li>
          <li><span className="mark">·</span><span>Cost {build.note}.</span></li>
          <li><span className="mark">·</span><span>With <code>graph_build="background"</code> the app answers from hybrid search immediately while the graph is built.</span></li>
        </ul>
        {load && storage && (
          <>
            <h3 className="bench-sub">Loading and storage ({storage.articles} articles, {num(storage.chunks)} chunks)</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th scope="col"></th><th scope="col">On disk</th><th scope="col">In memory</th><th scope="col">Load time</th></tr>
                </thead>
                <tbody>
                  <tr><th scope="row">Raw article text</th><td className="mono">{storage.raw_mb} MB</td><td className="mono">–</td><td className="mono">–</td></tr>
                  <tr><th scope="row">Hybrid index (<code>advanced</code> and most types)</th><td className="mono">{storage.hybrid.disk_mb} MB</td><td className="mono">~{storage.hybrid.ram_mb} MB</td><td className="mono">seconds</td></tr>
                  <tr><th scope="row">GraphRAG index, old JSON format</th><td className="mono">{(load.json_mb / 1000).toFixed(1)} GB</td><td className="mono">–</td><td className="mono">{load.json_s} s</td></tr>
                  <tr><th scope="row">GraphRAG index, binary vectors</th><td className="mono">{load.binary_mb} MB</td><td className="mono">~{storage.graph.ram_mb} MB</td><td className="mono">{load.binary_s} s</td></tr>
                </tbody>
              </table>
            </div>
            <p className="dim bench-foot">
              Each vector has {num(storage.embedding_dim)} dimensions (8 KB). The {num(build.entities)} entity vectors are{" "}
              {storage.graph.entity_vectors_mb} MB, two-thirds of the graph index. Smaller embeddings, float16 or an external
              vector database (pgvector, Qdrant) shrink this. Everything grows linearly with the number of articles.
            </p>
          </>
        )}
      </div>
    </section>
  );
}
