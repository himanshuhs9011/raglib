import { useEffect, useMemo, useState } from "react";
import { Code } from "../components/Code";
import { LINKS } from "../data";
import { AnswerModel, type ConfigResult, type Extras, Fixes, GraphCost, pct, WhichRag } from "./BenchmarkSections";

type Summary = Extras & {
  dataset: { name: string; authors: string; year: number; paper: string; source: string; license: string; corpus_articles: number };
  sample: { questions: number; seed: number; by_type: Record<string, number> };
  models: { generator: string; embedding: string; judge: string; endpoint: string; graph_builder?: string };
  excluded: Record<string, string>;
  configs: ConfigResult[];
  generated_at: string;
};

const DATA_URL = "benchmarks/multihop-rag.json";
const RAW_URL = "benchmarks/multihop-rag-per-question.json";
const TYPES: { key: string; label: string; asks: string }[] = [
  { key: "inference_query", label: "Inference", asks: "combine facts from several articles to name something" },
  { key: "comparison_query", label: "Comparison", asks: "compare what two sources report" },
  { key: "temporal_query", label: "Temporal", asks: "order or relate events in time" },
  { key: "null_query", label: "Unanswerable", asks: "the answer is in no article: the system must decline" },
];

const REPRODUCE = `
export NVIDIA_API_KEY=nvapi-...
pip install -e . "open-rag-eval==0.3.0" "sentence-transformers>=3"
python benchmarks/multihop_rag/bench.py download     # MultiHop-RAG from Hugging Face
python benchmarks/multihop_rag/bench.py sample --n 30 # stratified by type, seed 42
python benchmarks/multihop_rag/bench.py index        # 609 articles, embedded once
python benchmarks/multihop_rag/bench.py graph-index  # GraphRAG only: the knowledge graph (paid model, see 07)
python benchmarks/multihop_rag/bench.py generate     # every config answers every question
python benchmarks/multihop_rag/bench.py judge        # correctness vs the gold answer
python benchmarks/multihop_rag/bench.py export       # Open RAG Eval inputs
for f in benchmarks/multihop_rag/work/ore/*.yaml; do open-rag-eval eval --config "$f"; done
python benchmarks/multihop_rag/bench.py summarize
python benchmarks/multihop_rag/bench.py site         # the JSON this page reads
`;

/** 95% Wilson score interval for a proportion, as [low, high]. */
function wilson(p: number, n: number): [number, number] {
  if (!n) return [0, 1];
  const z = 1.96;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Sequential grayscale on the black surface: more = brighter. Text flips to dark on bright cells. */
function cellStyle(value: number | null) {
  if (value === null) return { background: "var(--panel)", color: "var(--dim)" };
  const light = Math.round(18 + value * 74); // 18% .. 92% lightness, one hue (neutral)
  return { background: `hsl(0 0% ${light}%)`, color: light > 55 ? "#000" : "var(--ink)" };
}

function judgeValue(config: ConfigResult, key: string): number | null {
  const value = config.judge?.[key];
  return typeof value === "number" ? value : null;
}

type Answer = {
  config: string;
  id: string;
  type?: string;
  query?: string;
  gold?: string;
  answer?: string;
  correct: boolean | null;
  reason?: string;
  sources?: (string | null)[];
  evidence?: string[];
  evidence_recall?: number | null;
  llm_calls?: number;
  latency_s?: number;
  error?: string;
};

const typeLabel = (key?: string) => TYPES.find((type) => type.key === key)?.label ?? key ?? "";

function Verdict({ correct }: { correct: boolean | null }) {
  const text = correct === true ? "correct" : correct === false ? "wrong" : "not judged";
  return <span className={`verdict ${correct === true ? "ok" : correct === false ? "bad" : ""}`}>{correct === true ? "✓" : correct === false ? "✗" : "?"} {text}</span>;
}

/** Every question with every configuration's answer and verdict, filterable. */
function AnswerBrowser({ configs, num }: { configs: string[]; num: string }) {
  const [rows, setRows] = useState<Answer[] | null>(null);
  const [config, setConfig] = useState("all");
  const [type, setType] = useState("all");
  const [outcome, setOutcome] = useState("all");

  useEffect(() => {
    fetch(RAW_URL).then((response) => (response.ok ? response.json() : [])).then(setRows).catch(() => setRows([]));
  }, []);

  const questions = useMemo(() => {
    const byId = new Map<string, Answer[]>();
    for (const row of rows ?? []) {
      if (config !== "all" && row.config !== config) continue;
      if (type !== "all" && row.type !== type) continue;
      if (outcome === "correct" && row.correct !== true) continue;
      if (outcome === "wrong" && row.correct !== false) continue;
      byId.set(row.id, [...(byId.get(row.id) ?? []), row]);
    }
    return [...byId.entries()];
  }, [rows, config, type, outcome]);

  return (
    <section>
      <div className="wrap">
        <div className="section-head"><span className="num">{num}</span><h2>Every answer</h2></div>
        <p className="muted bench-explain">
          Each question, the dataset's gold answer, and what every configuration answered, with the judge's verdict and
          reason. Sources are the articles the model was given; gold evidence is what the dataset says is needed.
        </p>
        <div className="filters" role="group" aria-label="Filter answers">
          <label><span className="label">Config</span>
            <select value={config} onChange={(event) => setConfig(event.target.value)}>
              <option value="all">All configs</option>
              {configs.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label><span className="label">Question type</span>
            <select value={type} onChange={(event) => setType(event.target.value)}>
              <option value="all">All types</option>
              {TYPES.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
            </select>
          </label>
          <label><span className="label">Outcome</span>
            <select value={outcome} onChange={(event) => setOutcome(event.target.value)}>
              <option value="all">Correct and wrong</option>
              <option value="correct">Correct only</option>
              <option value="wrong">Wrong only</option>
            </select>
          </label>
        </div>
        {rows === null ? (
          <p className="empty">Loading answers…</p>
        ) : questions.length === 0 ? (
          <p className="empty">No answers match these filters.</p>
        ) : (
          <div className="qa-list">
            <p className="dim qa-count">{questions.length} questions</p>
            {questions.map(([id, answers]) => (
              <details key={id} className="qa">
                <summary>
                  <span className="qa-meta"><code>{id}</code> {typeLabel(answers[0].type)}</span>
                  <span className="qa-q">{answers[0].query}</span>
                  <span className="qa-score">
                    {answers.filter((answer) => answer.correct === true).length}/{answers.length} correct
                  </span>
                </summary>
                <p className="qa-gold"><span className="label">Gold answer</span> {answers[0].gold}</p>
                {answers[0].evidence && answers[0].evidence.length > 0 && (
                  <p className="dim qa-evidence">Gold evidence: {answers[0].evidence.join(" · ")}</p>
                )}
                {answers.map((answer) => (
                  <div key={answer.config} className="qa-answer">
                    <div className="qa-answer-head">
                      <code>{answer.config}</code>
                      <Verdict correct={answer.correct} />
                      <span className="dim">
                        {answer.llm_calls ?? "–"} calls · {answer.latency_s ? `${answer.latency_s.toFixed(0)} s` : "–"} ·
                        evidence found {pct(answer.evidence_recall)}
                      </span>
                    </div>
                    <p className="qa-text">{answer.error ?? answer.answer}</p>
                    {answer.reason && <p className="dim qa-reason">Judge: {answer.reason}</p>}
                  </div>
                ))}
              </details>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export function Benchmarks() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(DATA_URL)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then(setData)
      .catch((reason) => setError(String(reason)));
  }, []);

  const ranked = useMemo(
    () => [...(data?.configs ?? [])].sort((a, b) => (b.accuracy ?? -1) - (a.accuracy ?? -1)),
    [data],
  );
  const byName = useMemo(() => new Map((data?.configs ?? []).map((config) => [config.name, config])), [data]);

  if (error) return <div className="wrap"><p className="empty">Could not load benchmark results ({error}).</p></div>;
  if (!data) return <div className="wrap"><p className="empty">Loading benchmark results…</p></div>;

  const n = data.sample.questions;
  const top = ranked.filter((config) => config.accuracy === ranked[0]?.accuracy);
  const hasJudge = data.configs.some((config) => Object.keys(config.judge ?? {}).length > 0);

  return (
    <>
      <section className="hero bench-hero">
        <div className="wrap">
          <span className="label">Benchmark · {data.generated_at}</span>
          <h1>Which RAG type works?</h1>
          <p className="lede">
            {data.configs.length} raglib configurations on <a href={data.dataset.paper}>{data.dataset.name}</a>, a public
            third-party benchmark. Same {n} questions, same {data.dataset.corpus_articles}-article corpus, same models,
            same judge. Only the pipeline changes.
          </p>
          <div className="meta-row">
            <span className="chip">{n} questions · seed {data.sample.seed}</span>
            <span className="chip">generator {data.models.generator}</span>
            <span className="chip">judge {data.models.judge}</span>
            <span className="chip">embeddings {data.models.embedding}</span>
            {data.models.graph_builder && <span className="chip">graph built by {data.models.graph_builder}</span>}
          </div>
          <p className="note">
            <strong>Pilot run.</strong> With {n} questions one answer moves accuracy by about {Math.round(100 / n)} points,
            so bars whose intervals overlap are not reliably different. Best here, tied at {pct(top[0]?.accuracy)}:{" "}
            {top.map((config, index) => (
              <span key={config.name}>{index > 0 && ", "}<code>{config.name}</code></span>
            ))}.
          </p>
        </div>
      </section>

      <WhichRag byName={byName} />

      <section>
        <div className="wrap">
          <div className="section-head"><span className="num">02</span><h2>Answer accuracy</h2></div>
          <p className="muted bench-explain">
            Share of questions answered correctly, judged against the dataset's gold answer by{" "}
            <code>{data.models.judge}</code>. Unanswerable questions count as correct only when the system declines.
            The thin line is the 95% confidence interval.
          </p>
          <ol className="meter-list" aria-label="Accuracy by configuration, best first">
            {ranked.map((config) => {
              const value = config.accuracy ?? 0;
              const [low, high] = wilson(value, config.accuracy_judged_n || n);
              const tip = `${config.name}: ${pct(config.accuracy)} correct (95% CI ${pct(low)}–${pct(high)}), ${config.about}`;
              return (
                <li key={config.name} title={tip}>
                  <span className="meter-name"><code>{config.name}</code></span>
                  <span className="meter-track" role="img" aria-label={tip}>
                    <i className="meter-fill" style={{ width: `${value * 100}%` }} />
                    <i className="meter-ci" style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%` }} />
                  </span>
                  <span className="meter-value">{pct(config.accuracy)}</span>
                </li>
              );
            })}
          </ol>
        </div>
      </section>

      <section>
        <div className="wrap">
          <div className="section-head"><span className="num">03</span><h2>By question type</h2></div>
          <p className="muted bench-explain">Which kind of question each pipeline handles. Brighter is better.</p>
          <div className="table-wrap">
            <table className="heat">
              <thead>
                <tr>
                  <th scope="col">Config</th>
                  {TYPES.map((type) => (
                    <th key={type.key} scope="col" title={type.asks}>
                      {type.label} <span className="dim">n={data.sample.by_type[type.key] ?? 0}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ranked.map((config) => (
                  <tr key={config.name}>
                    <th scope="row"><code>{config.name}</code></th>
                    {TYPES.map((type) => {
                      const value = config.accuracy_by_type[type.key] ?? null;
                      const [correct, total] = config.counts_by_type?.[type.key] ?? [0, 0];
                      return (
                        <td
                          key={type.key}
                          style={cellStyle(value)}
                          title={`${config.name} · ${type.label}: ${correct} of ${total} correct`}
                        >
                          <b>{pct(value)}</b> <small>{correct}/{total}</small>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="type-key">
            {TYPES.map((type) => <li key={type.key}><strong>{type.label}:</strong> {type.asks}</li>)}
          </ul>
        </div>
      </section>

      <section>
        <div className="wrap">
          <div className="section-head"><span className="num">04</span><h2>All metrics</h2></div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Config</th>
                  <th scope="col" title="LLM-judged against the gold answer">Accuracy</th>
                  <th scope="col" title="Gold answer string appears in the response (MultiHop-RAG style)">Strict match</th>
                  <th scope="col" title="Share of gold evidence articles among the sources given to the model">Evidence recall</th>
                  {hasJudge && <th scope="col" title="Open RAG Eval UMBRELA passage relevance, 0-3">UMBRELA</th>}
                  {hasJudge && <th scope="col" title="Open RAG Eval citation precision">Citations</th>}
                  {hasJudge && <th scope="col" title="Open RAG Eval HHEM factual consistency (higher = fewer hallucinations)">HHEM</th>}
                  <th scope="col">LLM calls</th>
                  <th scope="col">Median time</th>
                  <th scope="col">What it does</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((config) => (
                  <tr key={config.name}>
                    <td><code>{config.name}</code></td>
                    <td className="mono">{pct(config.accuracy)}</td>
                    <td className="mono">{pct(config.strict_accuracy)}</td>
                    <td className="mono">{pct(config.evidence_recall)}</td>
                    {hasJudge && <td className="mono">{judgeValue(config, "umbrela")?.toFixed(2) ?? "–"}</td>}
                    {hasJudge && <td className="mono">{pct(judgeValue(config, "citation_precision"))}</td>}
                    {hasJudge && <td className="mono">{judgeValue(config, "hallucination")?.toFixed(2) ?? "–"}</td>}
                    <td className="mono">{config.llm_calls_mean?.toFixed(1) ?? "–"}</td>
                    <td className="mono">{config.latency_s_median ? `${config.latency_s_median.toFixed(0)} s` : "–"}</td>
                    <td className="muted bench-about">{config.about}{config.errors ? ` · ${config.errors} failed` : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="dim bench-foot">
            Download every answer, source list and verdict: <a href={RAW_URL}>per-question results (JSON)</a> ·{" "}
            <a href={DATA_URL}>summary (JSON)</a>
          </p>
        </div>
      </section>

      <Fixes fixes={data.fixes ?? []} reference={data.heldout_reference} />
      <AnswerModel info={data.answer_model} />
      <GraphCost build={data.graph_build} load={data.load} storage={data.storage} />
      <AnswerBrowser configs={ranked.map((config) => config.name)} num="08" />

      <section>
        <div className="wrap">
          <div className="section-head"><span className="num">09</span><h2>Method</h2></div>
          <div className="grid c2">
            <div className="card">
              <span className="label">Data</span>
              <p>
                <a href={data.dataset.source}>{data.dataset.name}</a> ({data.dataset.authors}, {data.dataset.year},{" "}
                {data.dataset.license}): {data.dataset.corpus_articles} news articles, questions whose evidence spans 2 to 4
                articles. {n} questions sampled evenly across the four types. Articles are indexed with their title,
                source and date, because questions refer to them.
              </p>
            </div>
            <div className="card">
              <span className="label">Held constant</span>
              <p>
                raglib defaults for chunking (400 words), <code>top_k=5</code> and the answer prompt; one shared index;
                generator <code>{data.models.generator}</code> at temperature 0 on NVIDIA's API for every config. Only
                GraphRAG's knowledge graph was built by another model (section 07), and section 06 swaps the answer model
                on purpose.
              </p>
            </div>
            <div className="card">
              <span className="label">Scoring</span>
              <p>
                Accuracy is judged by a different, larger model than the generator. Strict match and evidence recall need
                no judge. <a href="https://github.com/vectara/open-rag-eval">Open RAG Eval</a> (Vectara, TREC-RAG
                methodology) adds passage relevance, citation support and HHEM hallucination scores.
              </p>
            </div>
            <div className="card">
              <span className="label">Limitations</span>
              <p>
                A {n}-question pilot; judge and generator share a vendor; one generator model; raglib benchmarking itself.
                MultiHop-RAG asks about specific facts in specific articles and has no corpus-wide "themes" questions,
                where GraphRAG's global search is meant to help. Everything needed to check or rerun it is published below.
              </p>
            </div>
          </div>
          <h3 className="bench-sub">Not run</h3>
          <ul className="list">
            {Object.entries(data.excluded).map(([name, reason]) => (
              <li key={name}><span className="mark">–</span><span><strong>{name}</strong><span className="desc">{reason}</span></span></li>
            ))}
          </ul>
          <h3 className="bench-sub">Reproduce</h3>
          <Code code={REPRODUCE} lang="bash" />
          <p className="dim">
            Code and full method: <a href={`${LINKS.github}/tree/main/benchmarks/multihop_rag`}>benchmarks/multihop_rag</a>
          </p>
        </div>
      </section>
    </>
  );
}
