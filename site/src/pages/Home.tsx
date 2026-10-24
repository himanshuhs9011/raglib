import { useState } from "react";
import { Code, Install } from "../components/Code";
import { Logo } from "../components/Logo";
import { FEATURES, LIMITATIONS, LINKS, LIVE_RESULTS, RAG_TYPES, VERSION, WHY } from "../data";

const QUICKSTART = `
from raglib import RAG

rag = RAG("docs/", llm="openrouter/openai/gpt-4.1-mini")   # any of 17 providers
result = rag.query("What is our refund policy?")

print(result.answer)       # grounded answer with [1] [2] citations
print(result.citations)    # which chunks support it
`;

const ONE_LINER = `
import raglib
print(raglib.ask("handbook.pdf", "How many vacation days do I get?").answer)
`;

function Section({ num, title, children, id }: { num: string; title: string; children: React.ReactNode; id?: string }) {
  return (
    <section id={id}>
      <div className="wrap">
        <div className="section-head">
          <span className="num">{num}</span>
          <h2>{title}</h2>
        </div>
        {children}
      </div>
    </section>
  );
}

export function Home() {
  const [selected, setSelected] = useState(RAG_TYPES[1].id);
  const type = RAG_TYPES.find((item) => item.id === selected) ?? RAG_TYPES[0];

  return (
    <>
      <section className="hero">
        <div className="wrap">
          <Logo size={72} className="hero-logo" />
          <h1>
            raglib<span>_</span>
          </h1>
          <p className="lede">
            Production-grade retrieval-augmented generation for Python. Point it at your documents, pick a model, ask
            questions and get cited answers. Hybrid search, 14 RAG architectures and GraphRAG are included.
          </p>
          <div className="meta-row">
            <span className="chip">v{VERSION}</span>
            <span className="chip">MIT</span>
            <span className="chip">Python 3.9+</span>
            <span className="chip">zero required dependencies</span>
          </div>
          <Install />
          <div className="actions">
            <a className="btn" href="#/playground">Try it in your browser</a>
            <a className="btn ghost" href="#/docs">Read the docs</a>
            <a className="btn ghost" href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>
          </div>
        </div>
      </section>

      <Section num="01" title="Quickstart">
        <div className="grid c2" style={{ background: "transparent", border: 0, gap: 16 }}>
          <div>
            <p className="label">Full control</p>
            <Code code={QUICKSTART} />
          </div>
          <div>
            <p className="label">One line</p>
            <Code code={ONE_LINER} />
            <Code lang="shell" code={`raglib ask docs/ "What changed in v2?"`} />
          </div>
        </div>
      </Section>

      <Section num="02" title="Why use raglib">
        <div className="grid c3">
          {WHY.map((item) => (
            <div className="card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section num="03" title="What you can do">
        <div className="grid c4">
          {FEATURES.map((item) => (
            <div className="card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section num="04" title="14 RAG architectures, one argument" id="rag-types">
        <p className="lede">Pick a type to see what it does. Switching is one keyword: <code>RAG(..., rag_type="{type.id}")</code>.</p>
        <div className="tabs" role="tablist" aria-label="RAG types">
          {RAG_TYPES.map((item) => (
            <button
              key={item.id}
              role="tab"
              type="button"
              aria-selected={item.id === selected}
              onClick={() => setSelected(item.id)}
            >
              {item.id}
            </button>
          ))}
        </div>
        <div className="tab-panel" role="tabpanel">
          <h3>{type.name}</h3>
          <p className="muted">{type.what}</p>
          <div className="grid c3">
            <div className="card">
              <span className="label">Use when</span>
              <p>{type.when}</p>
            </div>
            <div className="card">
              <span className="label">LLM calls per question</span>
              <p className="mono">{type.calls}</p>
            </div>
            <div className="card">
              <span className="label">Code</span>
              <p className="mono">rag_type="{type.id}"</p>
            </div>
          </div>
        </div>
      </Section>

      <Section num="05" title="Tested live, not just in mocks">
        <p className="lede">
          Run on OpenRouter free chat and embedding models (October 2026). The test scripts are in the repo:{" "}
          <code>examples/live_free_models_e2e.py</code> and <code>examples/live_all_rag_types.py</code>.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Test</th>
                <th>Result</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {LIVE_RESULTS.map((row) => (
                <tr key={row.test}>
                  <td>{row.test}</td>
                  <td className="muted">{row.result}</td>
                  <td className={row.status === "pass" ? "pass" : "fixed"}>{row.status === "pass" ? "PASS" : "FIXED"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="dim" style={{ marginTop: 12, fontSize: ".85rem" }}>
          Plus 162 offline unit and end-to-end tests, including concurrency (16 sessions while ingesting) and security checks.
        </p>
      </Section>

      <Section num="06" title="Limitations">
        <p className="lede">What raglib does not do well yet, so you can decide before you build on it.</p>
        <div className="grid c2">
          {LIMITATIONS.map((item) => (
            <div className="card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section num="07" title="Start building">
        <div className="row" style={{ gap: 16 }}>
          <Install />
          <a className="btn" href="#/docs">Docs</a>
          <a className="btn ghost" href={LINKS.pypi} target="_blank" rel="noreferrer">PyPI</a>
        </div>
      </Section>
    </>
  );
}
