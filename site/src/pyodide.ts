// Loads Pyodide (CPython compiled to WebAssembly) from the jsDelivr CDN, installs raglib from
// PyPI with micropip and exposes the functions defined in public/playground.py.

export const PYODIDE_VERSION = "0.28.3";
export const RAGLIB_VERSION = "2.0.3";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

interface PyodideInterface {
  loadPackage(names: string | string[]): Promise<void>;
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  globals: { get(name: string): (...args: unknown[]) => string };
}

declare global {
  interface Window {
    loadPyodide?: (options: { indexURL: string }) => Promise<PyodideInterface>;
  }
}

export interface BuildStats {
  chunks: number;
  documents: number;
  rag_type: string;
  retrieval_mode: string;
  index_ms: number;
}

export interface Source {
  id: string;
  score: number;
  content: string;
}

export interface AskResult {
  answer: string;
  ms: number;
  trace: string[];
  citations: number[];
  sources: Source[];
  error?: string;
}

export interface RagRuntime {
  build(text: string, ragType: string, retrievalMode: string, chunkSize: number, topK: number): BuildStats;
  ask(question: string): AskResult;
  graphHtml(): string;
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

let runtime: Promise<RagRuntime> | null = null;

/** Load once per page; later calls reuse the same runtime. */
export function loadRuntime(onStep: (message: string) => void): Promise<RagRuntime> {
  if (!runtime) {
    runtime = (async () => {
      onStep("Downloading the Python runtime (about 10 MB, cached after the first visit)…");
      if (!window.loadPyodide) await loadScript(`${INDEX_URL}pyodide.js`);
      const pyodide = await window.loadPyodide!({ indexURL: INDEX_URL });
      onStep(`Installing raglib-py ${RAGLIB_VERSION} from PyPI…`);
      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(`import micropip\nawait micropip.install("raglib-py==${RAGLIB_VERSION}")`);
      onStep("Starting raglib…");
      const bridge = await fetch(new URL("playground.py", document.baseURI)).then((r) => r.text());
      pyodide.runPython(bridge);
      const build = pyodide.globals.get("build");
      const ask = pyodide.globals.get("ask");
      const graphHtml = pyodide.globals.get("graph_html");
      return {
        build: (text, ragType, retrievalMode, chunkSize, topK) =>
          JSON.parse(build(text, ragType, retrievalMode, chunkSize, topK)) as BuildStats,
        ask: (question) => JSON.parse(ask(question)) as AskResult,
        graphHtml: () => graphHtml(),
      };
    })();
    runtime.catch(() => {
      runtime = null; // allow a retry
    });
  }
  return runtime;
}
