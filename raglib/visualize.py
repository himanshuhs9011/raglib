"""Interactive graph visualization: GraphRAG knowledge graphs and GraphContextWindow graphs.

Produces ONE self-contained HTML file (no CDN, no network): force-directed layout, pan/zoom,
drag, search, click a node to see its details and neighbours, colour by type or community.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6", "#d37295", "#8cd17d",
]


def _safe_json(payload: Any) -> str:
    """JSON for embedding inside <script>: '<', '>' and '&' are escaped so text can't break out."""

    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def graph_to_html(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    title: str = "raglib graph",
    path: Optional[str] = None,
    subtitle: str = "",
) -> str:
    """Render a generic graph. Nodes: {id, label, type, group?, size?, details?}; edges: {source, target, label?, weight?}.

    Returns the HTML string and writes it to ``path`` when given.
    """

    node_ids = {node["id"] for node in nodes}
    clean_edges = [edge for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids and edge["source"] != edge["target"]]
    types = [type_ for type_, _ in Counter(node.get("type", "node") for node in nodes).most_common()]
    groups = sorted({str(node.get("group")) for node in nodes if node.get("group") is not None})
    data = {
        "title": title,
        "subtitle": subtitle,
        "nodes": list(nodes),
        "edges": clean_edges,
        "typeColors": {type_: PALETTE[index % len(PALETTE)] for index, type_ in enumerate(types)},
        "groupColors": {group: PALETTE[index % len(PALETTE)] for index, group in enumerate(groups)},
    }
    html = _TEMPLATE.replace("__TITLE__", _escape_html(title)).replace("__DATA__", _safe_json(data))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(html, encoding="utf-8")
        logger.info("Wrote graph visualization (%d nodes, %d edges) to %s", len(nodes), len(clean_edges), path)
    return html


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for the <title>."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def graphrag_to_html(store: Any, path: Optional[str] = None, max_nodes: int = 400, title: str = "Knowledge graph") -> str:
    """Visualize a GraphRAG store: entities (sized by degree), relations, communities as colours."""

    entities = sorted(store.all_entities(), key=lambda entity: store.degree(entity.id), reverse=True)[:max_nodes]
    keep = {entity.id for entity in entities}
    community_titles = {community.id: community.title or community.id for community in store.communities(0)}
    nodes = []
    for entity in entities:
        community_id = entity.community_ids.get("0")
        details = {
            "Type": entity.type,
            "Description": entity.description,
            "Mentioned in chunks": len(entity.source_chunk_ids),
            "Relations": store.degree(entity.id),
        }
        if community_id:
            details["Community"] = community_titles.get(community_id, community_id)
        nodes.append(
            {
                "id": entity.id,
                "label": entity.name,
                "type": entity.type,
                "group": community_titles.get(community_id) if community_id else None,
                "size": store.degree(entity.id),
                "details": details,
            }
        )
    edges = [
        {"source": relation.source, "target": relation.target, "label": relation.type, "weight": relation.weight, "details": relation.description}
        for relation in store.all_relations()
        if relation.source in keep and relation.target in keep
    ]
    stats = store.stats()
    subtitle = (
        f"{stats['entities']} entities, {stats['relations']} relations, {stats['communities']} communities"
        + (f" (showing top {len(nodes)} by degree)" if stats["entities"] > len(nodes) else "")
    )
    return graph_to_html(nodes, edges, title=title, path=path, subtitle=subtitle)


def context_window_to_html(graph: Any, path: Optional[str] = None, max_nodes: int = 500, title: str = "Graph context window") -> str:
    """Visualize a GraphContextWindow: documents, sections, chunks, chat turns and their links."""

    priority = {"document": 0, "section": 1, "chat_turn": 2, "chunk": 3}
    graph_nodes = sorted(graph.nodes.values(), key=lambda node: priority.get(node.node_type, 9))[:max_nodes]
    keep = {node.id for node in graph_nodes}
    sizes = {"document": 14, "section": 8, "chat_turn": 6, "chunk": 3}
    nodes: List[Dict[str, Any]] = []
    for node in graph_nodes:
        label = node.metadata.get("title") or node.metadata.get("section_title") or node.summary or node.id
        nodes.append(
            {
                "id": node.id,
                "label": str(label)[:60],
                "type": node.node_type,
                "size": sizes.get(node.node_type, 3),
                "details": {
                    "Node id": node.id,
                    "Summary": node.summary,
                    "Content": node.content[:600],
                    "Keywords": ", ".join(node.metadata.get("keywords", [])[:8]),
                },
            }
        )
    edges = []
    seen = set()
    for node in graph_nodes:
        for ref in node.connected_nodes:
            if ref.node_id not in keep:
                continue
            key = tuple(sorted((node.id, ref.node_id))) + (ref.relation.replace("PREV_", "NEXT_").replace("HAS_", "PART_"),)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": node.id, "target": ref.node_id, "label": ref.relation, "weight": ref.weight})
    counts = Counter(node.node_type for node in graph.nodes.values())
    subtitle = ", ".join(f"{count} {type_}" for type_, count in counts.most_common())
    return graph_to_html(nodes, edges, title=title, path=path, subtitle=subtitle)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
/* Layout: full-bleed graph canvas with a details rail on the right (stacked below on phones). */
:root { --bg:#f5f6f4; --panel:#ffffff; --ink:#1b1d1c; --muted:#646a67; --line:#d6dad7; --edge:rgba(60,72,66,.30); --hi:#0f1311; color-scheme:light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#111413; --panel:#1a1e1c; --ink:#e7ebe9; --muted:#9aa29e; --line:#2f3532; --edge:rgba(205,215,210,.22); --hi:#ffffff; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#111413; --panel:#1a1e1c; --ink:#e7ebe9; --muted:#9aa29e; --line:#2f3532; --edge:rgba(205,215,210,.22); --hi:#ffffff; color-scheme:dark; }
* { box-sizing:border-box; }
html,body { margin:0; height:100%; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
#app { display:grid; grid-template-columns:1fr 320px; height:100%; }
#stage { position:relative; overflow:hidden; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
canvas.dragging { cursor:grabbing; }
header { position:absolute; top:12px; left:max(16px, env(safe-area-inset-left, 0px)); right:16px; display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; pointer-events:none; }
header > * { pointer-events:auto; }
.title h1 { margin:0; font-size:17px; font-weight:650; }
.title p { margin:2px 0 0; color:var(--muted); font-size:12.5px; }
.controls { margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; }
input, select, button { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:6px 10px; }
button { cursor:pointer; }
#legend { position:absolute; left:16px; bottom:14px; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px 10px; max-height:40%; overflow:auto; font-size:12.5px; }
#legend div { display:flex; align-items:center; gap:7px; margin:2px 0; cursor:pointer; }
#legend span.dot { width:10px; height:10px; border-radius:50%; flex:none; }
#legend div.off { opacity:.35; }
aside { background:var(--panel); border-left:1px solid var(--line); padding:16px; overflow:auto; }
aside h2 { font-size:15px; margin:0 0 4px; word-break:break-word; }
aside .kind { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
aside dl { margin:12px 0; }
aside dt { color:var(--muted); font-size:12px; margin-top:10px; }
aside dd { margin:2px 0 0; white-space:pre-wrap; word-break:break-word; }
aside ul { padding-left:18px; margin:6px 0; }
aside li { cursor:pointer; margin:3px 0; }
aside li:hover { text-decoration:underline; }
.hint { color:var(--muted); }
@media (max-width: 760px) { #app { grid-template-columns:1fr; grid-template-rows:62% auto; } aside { border-left:none; border-top:1px solid var(--line); } }
:focus-visible { outline:2px solid var(--hi); outline-offset:2px; }
@media (prefers-reduced-motion: reduce) { canvas { transition:none; } }
</style>
</head>
<body>
<div id="app">
  <div id="stage">
    <canvas id="c"></canvas>
    <header>
      <div class="title"><h1 id="t"></h1><p id="st"></p></div>
      <div class="controls">
        <input id="q" type="search" placeholder="Search nodes..." aria-label="Search nodes">
        <select id="colorBy" aria-label="Colour by"><option value="type">Colour: type</option><option value="group">Colour: community</option></select>
        <button id="fit" type="button">Fit</button>
      </div>
    </header>
    <div id="legend"></div>
  </div>
  <aside id="side"><p class="hint">Click a node to see its details and connections. Drag to pan, scroll to zoom, drag a node to move it.</p></aside>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
document.getElementById("t").textContent = DATA.title;
document.getElementById("st").textContent = DATA.subtitle || (DATA.nodes.length + " nodes, " + DATA.edges.length + " edges");
const byId = new Map(), nbrs = new Map();
const nodes = DATA.nodes.map((n, i) => { const a = 2 * Math.PI * i / Math.max(1, DATA.nodes.length), r = 40 + 6 * Math.sqrt(i);
  const o = Object.assign({}, n, { x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0, r: 4 + Math.min(14, Math.sqrt(n.size || 1) * 2.2) });
  byId.set(o.id, o); nbrs.set(o.id, []); return o; });
const edges = DATA.edges.map(e => ({ s: byId.get(e.source), t: byId.get(e.target), label: e.label || "", details: e.details || "", w: e.weight || 1 }));
edges.forEach(e => { nbrs.get(e.s.id).push(e); nbrs.get(e.t.id).push(e); });
let colorBy = "type", hidden = new Set(), selected = null, hover = null, matches = null;
const view = { x: 0, y: 0, k: 1 };
function color(n) { return colorBy === "group" ? (n.group != null && DATA.groupColors[n.group]) || "#9a9aa2" : DATA.typeColors[n.type] || "#9a9aa2"; }
function visible(n) { return !hidden.has(colorBy === "group" ? String(n.group) : n.type); }
function resize() { const d = devicePixelRatio || 1, b = canvas.getBoundingClientRect(); canvas.width = b.width * d; canvas.height = b.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); draw(); }
let alpha = 1;
function tick() {
  const vis = nodes.filter(visible), cell = 120, grid = new Map();
  vis.forEach(n => { const k = Math.floor(n.x / cell) + "," + Math.floor(n.y / cell); (grid.get(k) || grid.set(k, []).get(k)).push(n); });
  vis.forEach(a => { const gx = Math.floor(a.x / cell), gy = Math.floor(a.y / cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { const bucket = grid.get((gx + dx) + "," + (gy + dy)); if (!bucket) continue;
      for (const b of bucket) { if (b === a) continue; let x = a.x - b.x, y = a.y - b.y, d2 = x * x + y * y + 0.01; if (d2 > cell * cell) continue;
        const f = 900 / d2 * alpha; a.vx += x * f / Math.sqrt(d2); a.vy += y * f / Math.sqrt(d2); } } });
  edges.forEach(e => { if (!visible(e.s) || !visible(e.t)) return; const x = e.t.x - e.s.x, y = e.t.y - e.s.y, d = Math.sqrt(x * x + y * y) || 1, f = (d - 70) * 0.02 * alpha;
    e.s.vx += x / d * f; e.s.vy += y / d * f; e.t.vx -= x / d * f; e.t.vy -= y / d * f; });
  vis.forEach(n => { n.vx -= n.x * 0.002 * alpha; n.vy -= n.y * 0.002 * alpha; if (n !== drag) { n.x += n.vx; n.y += n.vy; } n.vx *= 0.6; n.vy *= 0.6; });
  alpha = Math.max(0.02, alpha * 0.985);
}
function toScreen(n) { return [n.x * view.k + view.x, n.y * view.k + view.y]; }
function draw() {
  const b = canvas.getBoundingClientRect(), css = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, b.width, b.height);
  const focus = selected || hover, focusSet = focus ? new Set([focus.id, ...nbrs.get(focus.id).map(e => e.s === focus ? e.t.id : e.s.id)]) : null;
  ctx.lineWidth = 1;
  for (const e of edges) { if (!visible(e.s) || !visible(e.t)) continue; const [x1, y1] = toScreen(e.s), [x2, y2] = toScreen(e.t);
    const on = focus && (e.s === focus || e.t === focus); ctx.strokeStyle = on ? css.getPropertyValue("--hi") : css.getPropertyValue("--edge");
    ctx.globalAlpha = focus && !on ? 0.25 : 1; ctx.lineWidth = on ? 1.6 : Math.min(3, 0.6 + Math.log1p(e.w) * 0.5);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    if (on && view.k > 0.6 && e.label) { ctx.globalAlpha = 1; ctx.fillStyle = css.getPropertyValue("--muted"); ctx.font = "11px system-ui"; ctx.fillText(e.label, (x1 + x2) / 2 + 3, (y1 + y2) / 2 - 3); } }
  ctx.globalAlpha = 1;
  for (const n of nodes) { if (!visible(n)) continue; const [x, y] = toScreen(n), dim = (focusSet && !focusSet.has(n.id)) || (matches && !matches.has(n.id));
    ctx.globalAlpha = dim ? 0.18 : 1; ctx.fillStyle = color(n); ctx.beginPath(); ctx.arc(x, y, n.r * Math.sqrt(view.k), 0, 7); ctx.fill();
    if (n === selected) { ctx.strokeStyle = css.getPropertyValue("--hi"); ctx.lineWidth = 2; ctx.stroke(); }
    if (!dim && (view.k > 1.1 || n.r > 9 || (focusSet && focusSet.has(n.id)) || (matches && matches.has(n.id)))) {
      ctx.fillStyle = css.getPropertyValue("--ink"); ctx.font = (n === focus ? "600 " : "") + "12px system-ui"; ctx.fillText(n.label, x + n.r * Math.sqrt(view.k) + 3, y + 4); } }
  ctx.globalAlpha = 1;
}
function loop() { if (alpha > 0.021 || drag) tick(); draw(); requestAnimationFrame(loop); }
function pick(px, py) { let best = null, bd = 1e9; for (const n of nodes) { if (!visible(n)) continue; const [x, y] = toScreen(n), d = (x - px) ** 2 + (y - py) ** 2, r = (n.r * Math.sqrt(view.k) + 4) ** 2; if (d < r && d < bd) { bd = d; best = n; } } return best; }
let drag = null, panning = null, moved = false, down = null;
canvas.addEventListener("pointerdown", ev => { const b = canvas.getBoundingClientRect(), n = pick(ev.clientX - b.left, ev.clientY - b.top); moved = false; down = [ev.clientX, ev.clientY];
  canvas.setPointerCapture(ev.pointerId); canvas.classList.add("dragging"); if (n) { drag = n; alpha = Math.max(alpha, 0.3); } else panning = { x: ev.clientX - view.x, y: ev.clientY - view.y }; });
canvas.addEventListener("pointermove", ev => { const b = canvas.getBoundingClientRect(), px = ev.clientX - b.left, py = ev.clientY - b.top; if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) moved = true;
  if (drag && moved) { drag.x = (px - view.x) / view.k; drag.y = (py - view.y) / view.k; } else if (panning && moved) { view.x = ev.clientX - panning.x; view.y = ev.clientY - panning.y; } else { hover = pick(px, py); } });
canvas.addEventListener("pointerup", ev => { canvas.classList.remove("dragging"); const b = canvas.getBoundingClientRect();
  if (!moved) select(pick(ev.clientX - b.left, ev.clientY - b.top)); drag = null; panning = null; down = null; });
canvas.addEventListener("wheel", ev => { ev.preventDefault(); const b = canvas.getBoundingClientRect(), px = ev.clientX - b.left, py = ev.clientY - b.top, f = Math.exp(-ev.deltaY * 0.0015);
  view.x = px - (px - view.x) * f; view.y = py - (py - view.y) * f; view.k *= f; }, { passive: false });
function el(tag, text, cls) { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; }
function select(n) { selected = n; const side = document.getElementById("side"); side.replaceChildren();
  if (!n) { side.append(el("p", "Click a node to see its details and connections.", "hint")); return; }
  side.append(el("div", n.type, "kind"), el("h2", n.label)); const dl = el("dl");
  Object.entries(n.details || {}).forEach(([k, v]) => { if (v === "" || v == null) return; dl.append(el("dt", k), el("dd", String(v))); }); side.append(dl);
  const list = nbrs.get(n.id); side.append(el("dt", "Connections (" + list.length + ")")); const ul = el("ul");
  list.slice(0, 80).forEach(e => { const other = e.s === n ? e.t : e.s, li = el("li", (e.s === n ? "→ " : "← ") + (e.label ? e.label + " " : "") + other.label);
    if (e.details) li.title = e.details; li.onclick = () => { select(other); center(other); }; ul.append(li); }); side.append(ul); }
function center(n) { const b = canvas.getBoundingClientRect(); view.x = b.width / 2 - n.x * view.k; view.y = b.height / 2 - n.y * view.k; }
function fit() { const vis = nodes.filter(visible); if (!vis.length) return; const b = canvas.getBoundingClientRect();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity; vis.forEach(n => { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); });
  view.k = Math.min(3, 0.85 * Math.min(b.width / (x1 - x0 + 40), b.height / (y1 - y0 + 40))); view.x = b.width / 2 - (x0 + x1) / 2 * view.k; view.y = b.height / 2 - (y0 + y1) / 2 * view.k; }
function legend() { const box = document.getElementById("legend"); box.replaceChildren(); const colors = colorBy === "group" ? DATA.groupColors : DATA.typeColors;
  const keys = Object.keys(colors); if (!keys.length) { box.append(el("div", colorBy === "group" ? "No communities built yet" : "")); return; }
  keys.slice(0, 30).forEach(k => { const row = el("div"), dot = el("span", null, "dot"); dot.style.background = colors[k]; row.append(dot, el("span", k));
    if (hidden.has(k)) row.classList.add("off"); row.onclick = () => { hidden.has(k) ? hidden.delete(k) : hidden.add(k); alpha = Math.max(alpha, 0.2); legend(); }; box.append(row); }); }
document.getElementById("q").addEventListener("input", ev => { const q = ev.target.value.trim().toLowerCase();
  matches = q ? new Set(nodes.filter(n => n.label.toLowerCase().includes(q)).map(n => n.id)) : null;
  if (matches && matches.size === 1) { const n = byId.get([...matches][0]); select(n); center(n); } });
document.getElementById("colorBy").addEventListener("change", ev => { colorBy = ev.target.value; hidden.clear(); legend(); });
document.getElementById("fit").addEventListener("click", fit);
if (!Object.keys(DATA.groupColors).length) document.querySelector('#colorBy option[value="group"]').disabled = true;
addEventListener("resize", resize);
for (let i = 0; i < 220; i++) tick();
resize(); fit(); legend(); loop();
</script>
</body>
</html>
"""
