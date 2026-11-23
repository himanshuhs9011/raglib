"""Model routing by tier: send each query to the cheapest path that can answer it.

Tiers:

* ``"faq"``   - a cached answer to the same (normalized) question is returned: no retrieval, no LLM call.
* ``"cheap"`` - hybrid retrieval without reranking, answered by ``cheap_llm`` (a small, fast model).
* ``"full"``  - the configured pipeline (rerank, the main model, the chosen ``rag_type``).

The router is any callable ``router(question, context) -> "faq" | "cheap" | "full"`` (or a
``TierDecision``); ``context`` has ``answer_cached``, ``cheap_available`` and ``personalized``. The
default ``HeuristicTierRouter`` makes no LLM call (microseconds). The chosen tier is logged, set as
``result.metadata["tier"]`` and tagged on the run's ``total`` trace event.

Trade-offs: two model integrations to maintain; a misrouted hard question gets a weaker answer on
the cheap tier (measure tier accuracy on a golden set - the router must be fast *and* accurate);
the FAQ tier repeats one answer to everyone (public tier only) until invalidated or expired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Union

from raglib.core.intents import is_comparison, is_global_question, needs_fresh_information
from raglib.core.text import count_tokens

TIER_FAQ = "faq"
TIER_CHEAP = "cheap"
TIER_FULL = "full"
TIERS = (TIER_FAQ, TIER_CHEAP, TIER_FULL)

_COMPLEX_RE = re.compile(
    r"\b(why|how does|how do|explain|analy[sz]e|compare|contrast|difference|differences|versus|vs\.?|impact|"
    r"trade-?offs?|pros and cons|step by step|summari[sz]e|evaluate|relationship|cause|caused|implications?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TierDecision:
    """A routing decision with its reason (logged and returned in ``result.metadata``)."""

    tier: str
    reason: str = ""


TierRouter = Callable[[str, Mapping[str, Any]], Union[str, TierDecision]]


class HeuristicTierRouter:
    """Fast rule-based router (no LLM call).

    * a cached answer exists -> ``faq``
    * short factual lookups (at most ``max_cheap_tokens`` tokens, no comparison / explanation /
      corpus-wide / multi-part wording) -> ``cheap`` (when a cheap model is configured)
    * everything else -> ``full``
    """

    def __init__(self, max_cheap_tokens: int = 14):
        """``max_cheap_tokens``: longest question still sent to the cheap tier."""

        self.max_cheap_tokens = max_cheap_tokens

    def __call__(self, question: str, context: Mapping[str, Any]) -> TierDecision:
        """Pick a tier for ``question``."""

        if context.get("answer_cached"):
            return TierDecision(TIER_FAQ, "cached answer")
        if not context.get("cheap_available"):
            return TierDecision(TIER_FULL, "no cheap model configured")
        tokens = count_tokens(question)
        if tokens > self.max_cheap_tokens:
            return TierDecision(TIER_FULL, f"long question ({tokens} tokens)")
        if _COMPLEX_RE.search(question) or is_comparison(question) or is_global_question(question):
            return TierDecision(TIER_FULL, "needs reasoning")
        if needs_fresh_information(question) or question.count("?") > 1 or " and " in question.lower():
            return TierDecision(TIER_FULL, "multi-part or time-sensitive")
        return TierDecision(TIER_CHEAP, "short factual lookup")


def decide(router: TierRouter, question: str, context: Dict[str, Any]) -> TierDecision:
    """Run ``router`` and normalize its answer (unknown tiers raise ``ValueError``)."""

    raw = router(question, context)
    decision = raw if isinstance(raw, TierDecision) else TierDecision(str(raw))
    tier = decision.tier.strip().lower()
    if tier not in TIERS:
        raise ValueError(f"tier router returned {decision.tier!r}; expected one of {list(TIERS)}")
    return TierDecision(tier, decision.reason)
