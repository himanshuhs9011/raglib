import { useState, type ReactNode } from "react";

const KEYWORDS = new Set([
  "from", "import", "for", "in", "print", "def", "return", "if", "else", "with", "as", "async", "await", "True", "False", "None",
]);

/** Tiny Python/shell highlighter: comments, strings and keywords only (monochrome). */
function highlight(source: string): ReactNode[] {
  const out: ReactNode[] = [];
  const pattern = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b([A-Za-z_]+)\b/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(source)) !== null) {
    if (match.index > last) out.push(source.slice(last, match.index));
    if (match[1]) out.push(<span key={key++} className="tok-c">{match[1]}</span>);
    else if (match[2]) out.push(<span key={key++} className="tok-s">{match[2]}</span>);
    else if (KEYWORDS.has(match[3])) out.push(<span key={key++} className="tok-k">{match[3]}</span>);
    else out.push(match[3]);
    last = pattern.lastIndex;
  }
  if (last < source.length) out.push(source.slice(last));
  return out;
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function Code({ code, lang = "python" }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false);
  const text = code.trim();
  return (
    <div className="code">
      <div className="code-bar">
        <span className="label">{lang}</span>
        <button
          className="copy"
          type="button"
          onClick={async () => {
            if (await copyText(text)) {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }
          }}
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre>
        <code>{highlight(text)}</code>
      </pre>
    </div>
  );
}

export function Install({ command = "pip install raglib-py" }: { command?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="install">
      <code>$ {command}</code>
      <button
        type="button"
        onClick={async () => {
          if (await copyText(command)) {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }
        }}
      >
        {copied ? "copied" : "copy"}
      </button>
    </div>
  );
}
