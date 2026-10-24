"""Live test of every rag_type on OpenRouter FREE models, sharing one embedding cache.

    OPENROUTER_API_KEY=... python examples/live_all_rag_types.py [type ...]

Uses the haystack from live_free_models_e2e.py (built if missing). Results go to
examples/reports/live_rag_types.json (git-ignored). The key is read from the environment only.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from a source checkout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_free_models_e2e import CHAT, EMBED, FALLBACKS, HERE, build_haystack  # noqa: E402
from raglib import RAG  # noqa: E402
from raglib.components.tools import Tool  # noqa: E402

KEY = os.environ["OPENROUTER_API_KEY"]
HAYSTACK = os.path.join(HERE, "haystack.md")
OUT = os.path.join(HERE, "live_rag_types.json")

BLUEFIN = "What is the launch authorization code for Project Bluefin?"
TWO_HOP = "What is Project Bluefin's launch authorization code, and which company services the Building C generator?"
CASES = {
    # rag_type: (questions, expected substrings in the final answer, extra RAG kwargs)
    "naive": ([BLUEFIN], ["7-kestrel-42"], {}),
    "hybrid": (["How much is the travel reimbursement for a conference in Reykjavik?"], ["1,840|1840"], {}),
    "routing": (["What does Dr. Amara Okafor keep on her desk during audits?"], ["pickled mango"], {}),
    "corrective": (["Who services the backup generator in Building C?"], ["voltline"], {}),
    "multi_query": (["What snack does Dr. Okafor have during audits?"], ["pickled mango"], {}),
    "memory": (["Tell me about the backup generator in Building C.", "Which company services it, and on which day?"],
               ["voltline", "19"], {}),
    "tool": (["Use the currency converter to convert the Reykjavik conference travel reimbursement cap into US dollars."],
             ["1,840|1840", "2,024|2024|2,024.00"], {"tools": "converter"}),
    "multi_hop": ([TWO_HOP], ["7-kestrel-42", "voltline"], {}),
    "agentic": ([TWO_HOP], ["7-kestrel-42", "voltline"], {}),
    "self": (["On which day of the month is the Building C generator serviced?"], ["19"], {}),
    "web": (["What is the latest stable major version of the Python programming language?"], ["python"],
            {"web_search_provider": "duckduckgo"}),
    "hybrid_web": (["What is Project Bluefin's launch code, and what is the capital of Iceland?"],
                   ["7-kestrel-42", "reykjavik"], {"web_search_provider": "duckduckgo"}),
}


def quota():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]["free_model_daily_requests"]


def converter(text: str) -> str:
    """Fixed-rate EUR->USD converter (1 EUR = 1.10 USD) so the expected answer is deterministic."""

    import re

    numbers = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
    if not numbers:
        return "Give an amount in euros, e.g. '1840 EUR'."
    return f"{numbers[0]:,.2f} EUR = {numbers[0] * 1.10:,.2f} USD (fixed rate 1 EUR = 1.10 USD)"


def matches(answer: str, expected):
    low = answer.lower()
    return all(any(alt in low for alt in group.split("|")) for group in expected)


def main(types):
    if not os.path.exists(HAYSTACK):
        build_haystack(HAYSTACK)
    seed = RAG(llm=CHAT, embedding_llm=EMBED, llm_cache=False)
    shared_embedding = seed._embedding  # one cache: the haystack is embedded once for every rag_type
    results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for rag_type in types:
        questions, expected, extra = CASES[rag_type]
        extra = dict(extra)
        if extra.get("tools") == "converter":
            extra["tools"] = [Tool(name="currency_converter",
                                   description="Convert an amount in euros to US dollars. Input: an amount like '1840 EUR'.",
                                   func=converter)]
        left = quota()["remaining"]
        if left < 6:
            print(f"stopping before {rag_type}: only {left} free requests left")
            break
        before = quota()["used"]
        t = time.time()
        try:
            rag = RAG(HAYSTACK, rag_type=rag_type, llm=CHAT, llm_fallbacks=FALLBACKS, embedding_llm=shared_embedding,
                      chunk_size=50, chunk_overlap=0, top_k=4, llm_timeout=120, llm_max_retries=3, **extra)
            turns = []
            for q in questions:
                res = rag.query(q, session_id="live")
                turns.append({"q": q, "answer": res.answer, "trace": res.reasoning_trace,
                              "citations": [c.source for c in res.citations]})
            ok = matches(turns[-1]["answer"], expected)
            entry = {"ok": ok, "turns": turns, "latency_s": round(time.time() - t, 1)}
        except Exception as exc:  # noqa: BLE001
            entry = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "latency_s": round(time.time() - t, 1)}
        entry["requests"] = quota()["used"] - before
        results[rag_type] = entry
        json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n=== {rag_type}: {'PASS' if entry['ok'] else 'FAIL'}  ({entry['latency_s']}s, ~{entry['requests']} req)")
        for turn in entry.get("turns", []):
            print("Q:", turn["q"])
            print("A:", turn["answer"][:600].replace("\n", " "))
            print("trace:", turn["trace"][:14])
        if "error" in entry:
            print("ERROR:", entry["error"])
    print("\nremaining:", quota())


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
