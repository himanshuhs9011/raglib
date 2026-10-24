"""Word-boundary query intent heuristics used when no LLM decision is available.

Matching is done on whole tokens (and whole phrases), never substrings, so
"this" never matches "hi" and "canvas" never matches "vs".
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Set

from raglib.core.text import tokenize

logger = logging.getLogger(__name__)

GREETING_TERMS: Set[str] = {
    "hi", "hello", "hey", "thanks", "thank", "thx", "bye", "goodbye", "greetings", "namaste",
}
GREETING_PHRASES = ("good morning", "good evening", "good afternoon", "thank you", "how are you")
FRESHNESS_TERMS: Set[str] = {"latest", "today", "news", "current", "currently", "recent", "now", "tonight", "yesterday"}
COMPARISON_TERMS: Set[str] = {"compare", "comparison", "versus", "vs", "difference", "differences", "trend", "trends", "benchmark"}
GLOBAL_TERMS: Set[str] = {
    "overall", "themes", "theme", "summarize", "summarise", "summary", "overview", "main",
    "across", "entire", "whole", "all", "trends", "landscape", "holistic",
}

_YEAR_RE = re.compile(r"\b20[2-9]\d\b")


def has_any_term(text: str, terms: Iterable[str]) -> bool:
    """Return True when any whole token of ``text`` is in ``terms``."""

    return bool(set(tokenize(text)) & set(terms))


def is_greeting(query: str) -> bool:
    """Return True only for short small-talk messages that need no retrieval."""

    tokens = tokenize(query)
    if not tokens or len(tokens) > 6:
        return False
    lowered = " ".join(tokens)
    if any(lowered.startswith(phrase) or lowered == phrase for phrase in GREETING_PHRASES):
        return True
    content = [token for token in tokens if token not in GREETING_TERMS and token not in {"there", "you", "so", "much", "a", "lot"}]
    return tokens[0] in GREETING_TERMS and not content


def needs_fresh_information(query: str) -> bool:
    """Return True when the query asks about recent or live information."""

    return has_any_term(query, FRESHNESS_TERMS) or bool(_YEAR_RE.search(query))


def is_comparison(query: str) -> bool:
    """Return True when the query compares entities or asks about trends."""

    return has_any_term(query, COMPARISON_TERMS)


def is_global_question(query: str) -> bool:
    """Return True for corpus-wide sensemaking questions (GraphRAG global search)."""

    tokens = set(tokenize(query))
    hits = tokens & GLOBAL_TERMS
    if not hits:
        return False
    # "all" / "main" alone are weak signals; require a second cue for them.
    strong = hits - {"all", "main"}
    return bool(strong) or len(hits) >= 2
