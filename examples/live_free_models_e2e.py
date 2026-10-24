"""Live end-to-end test of raglib on OpenRouter FREE models (chat + embeddings).

Stages (run one at a time to watch the 50-requests/day free quota):
  quota     - print remaining free requests
  needle    - needle-in-a-haystack: retrieval (dense / sparse / hybrid) + cited answers
  graph     - GraphRAG local (multi-hop) search + visualization
  global    - GraphRAG global (community) search
  context   - GraphContextWindow on the haystack + answer + visualization
Key is read from OPENROUTER_API_KEY only. Outputs go to examples/reports/ (git-ignored).
Set DRY=1 to run every stage offline with the mock LLM (no key, no quota).

    python examples/live_free_models_e2e.py quota
    python examples/live_free_models_e2e.py needle
"""
import json, os, random, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from a source checkout
from raglib import RAG  # noqa: E402
from raglib.schemas import Document  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(HERE, exist_ok=True)
KEY = os.getenv("OPENROUTER_API_KEY", "")
CHAT = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACKS = ["openrouter/cohere/north-mini-code:free"]
EMBED = "openrouter/nvidia/nemotron-3-embed-1b:free"
RESULTS = os.path.join(HERE, "live_results.json")
if os.getenv("DRY"):
    CHAT, FALLBACKS, EMBED = "mock", [], "mock"
    RESULTS = os.path.join(HERE, "dry_results.json")


def quota():
    if os.getenv("DRY"):
        return {"used": 0}
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]["free_model_daily_requests"]


def save(stage, data):
    allres = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
    allres[stage] = data
    json.dump(allres, open(RESULTS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def make_rag(**kw):
    return RAG(llm=CHAT, llm_fallbacks=FALLBACKS or None, embedding_llm=EMBED, llm_timeout=120, llm_max_retries=3, **kw)


# ------------------------------------------------------------------ haystack
NEEDLES = {
    "bluefin": "Project Bluefin's launch authorization code is 7-Kestrel-42 and it is stored only in the vault ledger.",
    "reykjavik": "For conferences held in Reykjavik, the maximum travel reimbursement is 1,840 euros per trip.",
    "generator": "The backup diesel generator in Building C is serviced on the 19th of every month by Voltline Ltd.",
    "okafor": "During quarterly audits, Dr. Amara Okafor always keeps a jar of pickled mango on her desk as a snack.",
}
QUESTIONS = {
    "bluefin": ("What is the launch authorization code for Project Bluefin?", ["7-kestrel-42"]),
    "reykjavik": ("How much money can I get back for travelling to a conference in Iceland's capital?", ["1,840", "1840"]),
    "generator": ("Who maintains the standby power unit in Building C and when?", ["voltline", "19"]),
    "okafor": ("What does Dr. Okafor snack on during audits?", ["pickled mango"]),
}


def build_haystack(path):
    rnd = random.Random(7)
    cities = ["Lisbon", "Oslo", "Nairobi", "Toronto", "Osaka", "Lima", "Prague", "Dublin", "Hanoi", "Cairo",
              "Bogota", "Helsinki", "Perth", "Seoul", "Zurich", "Accra"]
    projects = ["Redwood", "Marlin", "Cobalt", "Juniper", "Falcon", "Granite", "Tidal", "Nimbus", "Orchid", "Quartz"]
    buildings = ["A", "B", "D", "E", "F", "G"]
    vendors = ["Ampera Services", "GridCare", "PowerHouse Co", "Kilowatt Partners", "Northline Electric"]
    people = ["Dr. Ivan Petrov", "Ms. Leila Haddad", "Mr. Kenji Mori", "Dr. Sofia Rossi", "Mr. Daniel Osei"]
    snacks = ["roasted almonds", "green tea", "dark chocolate", "rice crackers", "dried apricots"]
    sections = []
    for i, city in enumerate(cities):
        sections.append(("Travel policy - " + city,
                         f"For conferences held in {city}, the maximum travel reimbursement is {rnd.randint(600, 2600):,} euros per trip. "
                         f"Receipts must be submitted within {rnd.randint(10, 45)} days. Economy class is required for flights under "
                         f"{rnd.randint(3, 8)} hours, and hotel bookings should use the corporate portal."))
    for p in projects:
        sections.append(("Project " + p,
                         f"Project {p} is led by the {rnd.choice(['platform', 'research', 'growth', 'security'])} team. Its launch "
                         f"authorization code is {rnd.randint(1, 9)}-{rnd.choice(['Heron', 'Lynx', 'Otter', 'Raven'])}-{rnd.randint(10, 99)}. "
                         f"Status reports are due every {rnd.choice(['Monday', 'Thursday'])} and milestones are tracked in the roadmap tool."))
    for b in buildings:
        sections.append((f"Facilities - Building {b}",
                         f"The backup diesel generator in Building {b} is serviced on the {rnd.randint(1, 28)}th of every month by "
                         f"{rnd.choice(vendors)}. Fire drills happen twice a year and the loading dock closes at {rnd.randint(5, 8)} pm."))
    for person, snack in zip(people, snacks):
        sections.append(("Staff profile - " + person,
                         f"During quarterly audits, {person} prefers {snack}. They joined the finance office "
                         f"{rnd.randint(2, 15)} years ago and coordinate vendor contracts."))
    filler_topics = ["Onboarding", "Security awareness", "Code review", "Expense tools", "Parental leave", "Remote work",
                     "Laptop refresh", "Incident response", "Data retention", "Customer escalations", "Holiday calendar",
                     "Learning budget", "Office plants", "Parking", "Mentoring", "Hackathons"]
    for topic in filler_topics:
        for part in range(1, 5):
            sections.append((f"{topic} (part {part})",
                             f"This part of the {topic.lower()} guide explains step {part}. Employees should read the checklist, "
                             f"confirm with their manager, and record completion in the HR system within {rnd.randint(2, 20)} days. "
                             f"Questions go to the {rnd.choice(['people', 'IT', 'facilities', 'finance'])} help desk."))
    rnd.shuffle(sections)
    # Needles at different depths (10%, 35%, 60%, 90%).
    for depth, key in zip([0.1, 0.35, 0.6, 0.9], NEEDLES):
        title = {"bluefin": "Project Bluefin", "reykjavik": "Travel policy - Reykjavik",
                 "generator": "Facilities - Building C", "okafor": "Staff profile - Dr. Amara Okafor"}[key]
        sections.insert(int(depth * len(sections)), (title, NEEDLES[key]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Company Operations Handbook\n\n")
        for title, body in sections:
            fh.write(f"## {title}\n\n{body}\n\n")
    return len(sections)


def needle_rank(docs, key):
    marker = {"bluefin": "7-Kestrel-42", "reykjavik": "Reykjavik", "generator": "Voltline", "okafor": "Okafor"}[key]
    for i, doc in enumerate(docs):
        if marker in doc.content:
            return i + 1
    return None


def stage_needle():
    path = os.path.join(HERE, "haystack.md")
    n_sections = build_haystack(path)
    rag = make_rag(rag_type="advanced", chunk_size=50, chunk_overlap=0, top_k=4)
    t = time.time()
    rag.add(path)
    stats = rag.stats()
    ingest_s = round(time.time() - t, 1)
    # Embed all needle questions in ONE request (saves free quota), then seed the query cache.
    qs = [q for q, _ in QUESTIONS.values()]
    vecs = rag._embedding.inner.embed_documents(qs)
    for q, v in zip(qs, vecs):
        rag._embedding._put("q:" + rag._embedding._key(q), list(v))
    retrieval = {}
    for key, (q, _) in QUESTIONS.items():
        retrieval[key] = {mode: needle_rank(rag._retriever.retrieve(q, top_k=10, mode=mode), key)
                          for mode in ("dense", "sparse", "hybrid")}
    print("haystack sections:", n_sections, "stats:", stats, "ingest s:", ingest_s)
    print("needle rank (1=best, None=not in top10):", json.dumps(retrieval))
    answers = {}
    for key in ("bluefin", "reykjavik", "generator"):
        q, expected = QUESTIONS[key]
        t = time.time()
        res = rag.query(q)
        ok = all(any(e in res.answer.lower() for e in [x]) for x in expected) if key != "reykjavik" else any(
            e in res.answer for e in expected)
        answers.setdefault("_", None)
        answers[key] = {"question": q, "answer": res.answer, "correct": ok,
                        "citations": [c.source for c in res.citations] if res.citations else [],
                        "latency_s": round(time.time() - t, 1)}
        print(json.dumps(answers[key], ensure_ascii=False))
    save("needle", {"sections": n_sections, "stats": stats, "ingest_s": ingest_s, "retrieval": retrieval, "answers": answers})


GRAPH_DOCS = [
    "Lena Park founded Quillstone Robotics in 2014 in Busan, South Korea. Quillstone builds warehouse robots. "
    "In 2021 Quillstone Robotics acquired Harbor Vision, a small lidar startup.",
    "Harbor Vision was originally created by Tomas Rivera while he was a researcher at the University of Valencia. "
    "Its lidar sensors are now used in every Quillstone robot.",
    "After leaving Harbor Vision, Tomas Rivera became chief scientist at Meridian Labs. "
    "Meridian Labs is headquartered in Lisbon and builds autonomous underwater drones for ocean mapping.",
    "Orchard Analytics is a Berlin company that sells crop-yield forecasting software to farmers. "
    "Its CEO, Mia Keller, previously worked at a bank in Frankfurt.",
]
GRAPH_Q = "In which city is the company headquartered whose chief scientist created the startup that Quillstone acquired?"


def stage_graph():
    rag = make_rag(rag_type="graph", chunk_size=200, chunk_overlap=0, top_k=4, search_mode="local")
    t = time.time()
    rag.add([Document(id=f"graph_doc_{i}", content=c, metadata={"source": f"graph_doc_{i}"}) for i, c in enumerate(GRAPH_DOCS)])
    ingest_s = round(time.time() - t, 1)
    store = rag._graph_indexer.store
    entities = sorted(e.name for e in store.all_entities())
    relations = [(r.source, r.type, r.target) for r in store.all_relations()]
    print("ingest s:", ingest_s, "entities:", len(entities), "relations:", len(relations))
    print("entities:", entities)
    for r in relations:
        print("  ", r)
    t = time.time()
    res = rag.query(GRAPH_Q)
    print("ANSWER:", res.answer)
    print("trace:", res.reasoning_trace)
    html = rag.visualize_graph(os.path.join(HERE, "live_graphrag.html"), rebuild_communities=False)
    rag.save(os.path.join(HERE, "live_graph_state"))
    save("graph", {"ingest_s": ingest_s, "entities": entities, "relations": relations, "question": GRAPH_Q,
                   "answer": res.answer, "correct": "lisbon" in res.answer.lower(),
                   "latency_s": round(time.time() - t, 1), "trace": res.reasoning_trace, "html": html})


def stage_global():
    rag = make_rag(rag_type="graph", chunk_size=200, chunk_overlap=0, top_k=4, search_mode="global",
                   graph_kwargs={"top_communities": 2, "max_points": 8})
    rag.load(os.path.join(HERE, "live_graph_state"))  # reuse extraction: no new extraction calls
    t = time.time()
    res = rag.query("What are the main themes and industries across all of these documents?")
    communities = rag._graph_indexer.store.communities()
    print("communities:", len(communities))
    for c in communities:
        print("  ", c.level, c.title if hasattr(c, "title") else "", len(c.entity_ids))
    print("ANSWER:", res.answer)
    html = rag.visualize_graph(os.path.join(HERE, "live_graphrag.html"), rebuild_communities=False)
    save("global", {"answer": res.answer, "communities": len(communities), "latency_s": round(time.time() - t, 1),
                    "trace": res.reasoning_trace, "html": html})


def stage_context():
    from raglib.graph_context.engine import GraphContextWindow

    path = os.path.join(HERE, "haystack.md")
    text = open(path, encoding="utf-8").read()
    gcw = GraphContextWindow(chunking_strategy="recursive", chunk_size=60, chunk_overlap=0, max_context_tokens=600)
    t = time.time()
    gcw.add_documents([Document(id="haystack", content=text, metadata={"source": "haystack.md"})])
    build_ms = round((time.time() - t) * 1000)
    ranks = {}
    for key, (q, _) in QUESTIONS.items():
        t = time.time()
        result = gcw.build_context_window(q)
        ranks[key] = {"rank": needle_rank(result.documents, key), "ms": round((time.time() - t) * 1000, 1),
                      "tokens": result.total_tokens}
    print("graph build ms:", build_ms, "nodes:", len(gcw.nodes) if hasattr(gcw, "nodes") else "?")
    print("needle in context window:", json.dumps(ranks))
    # One real answer: question -> graph context window -> free LLM.
    from raglib.facade import RAG as _RAG

    llm = _RAG(llm=CHAT, llm_fallbacks=FALLBACKS or None, embedding_llm="none", llm_timeout=120, llm_max_retries=3)._llm
    q, expected = QUESTIONS["okafor"]
    ctx = gcw.build_context_window(q)
    answer = llm.complete(f"Answer from the context only.\n\nContext:\n{ctx.formatted_context}\n\nQuestion: {q}")
    gcw.add_chat_turn(q, answer) if hasattr(gcw, "add_chat_turn") else None
    print("ANSWER:", answer)
    gcw.to_html(os.path.join(HERE, "live_context_window.html"))
    html = os.path.join(HERE, "live_context_window.html")
    save("context", {"build_ms": build_ms, "ranks": ranks, "question": q, "answer": answer,
                     "correct": "pickled mango" in answer.lower(), "html": html})


def stage_needle3():
    rag = RAG(llm=CHAT, llm_fallbacks=FALLBACKS or None, embedding_llm="none", llm_timeout=120, llm_max_retries=3,
              chunk_size=50, chunk_overlap=0, top_k=4)
    rag.add(os.path.join(HERE, "haystack.md"))
    q, expected = QUESTIONS["generator"]
    t = time.time()
    res = rag.query(q)
    out = {"question": q, "answer": res.answer, "correct": all(e in res.answer.lower() for e in expected),
           "citations": [c.source for c in res.citations], "latency_s": round(time.time() - t, 1), "retrieval": "bm25"}
    print(json.dumps(out, ensure_ascii=False))
    save("needle_generator", out)


if __name__ == "__main__":
    stage = sys.argv[1]
    before = quota()
    print("free requests before:", before)
    if stage != "quota":
        {"needle": stage_needle, "needle3": stage_needle3, "graph": stage_graph, "global": stage_global, "context": stage_context}[stage]()
        after = quota()
        print("free requests after:", after, "-> used", after["used"] - before["used"])
