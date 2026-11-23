"""Tier 0 guards: rules that cost nothing (no model, no network, microseconds per check).

Input (the question, before it reaches traces, caches, memory or a model):
    ``MaxQuestionTokens``, ``UserRateLimit``, ``UserDailyBudget``, ``PIIMasker``, ``InjectionHeuristic``
Retrieval (the documents, before the answer LLM call):
    ``NothingRelevant`` - answers "couldn't find this" without calling the LLM
Output (the answer, before it is returned):
    ``MaxOutputTokens``, ``OutputLeakGuard``, ``CitationAccessGuard``, ``JSONSchemaGuard``
Agent tools:
    ``ToolPolicy`` - tool allowlist per role, max calls per question, argument limits
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from raglib.guards import pii
from raglib.guards.base import Decision, Guard, GuardContext, allow, block, modify
from raglib.guards.store import GuardStore, InMemoryGuardStore

logger = logging.getLogger(__name__)


class GuardSkippedWarning(UserWarning):
    """A guard could not run (missing model, prices, ...) and was skipped; shown once per reason."""


_WARNED: Set[str] = set()
_WARN_LOCK = threading.Lock()


def warn_once(key: str, message: str) -> None:
    """Emit ``message`` as a ``GuardSkippedWarning`` (and a log line) once per process per ``key``."""

    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    logger.warning(message)
    warnings.warn(message, GuardSkippedWarning, stacklevel=3)


def skipped(reason: str) -> Decision:
    """An allow decision recording that the guard did not run."""

    decision = allow(reason=reason, confidence=0.0)
    decision.mode = "skipped"
    return decision


def _count(text: str, counter: Optional[Callable[[str], int]]) -> int:
    if counter is not None:
        return int(counter(text))
    from raglib.core.text import count_tokens

    return count_tokens(text)


# --------------------------------------------------------------------------------------------- input


class MaxQuestionTokens(Guard):
    """Block questions longer than ``max_tokens`` (protects context budget and cost)."""

    name = "max_question_tokens"
    cacheable = True
    block_message = "Your question is too long. Please shorten it and try again."

    def __init__(self, max_tokens: int = 1000, token_counter: Optional[Callable[[str], int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.max_tokens = max_tokens
        self.token_counter = token_counter

    def check(self, value: str, context: GuardContext) -> Decision:
        tokens = _count(value, self.token_counter)
        if tokens > self.max_tokens:
            return block(self.block_message, reason=f"{tokens} tokens > {self.max_tokens}")
        return allow(reason=f"{tokens} tokens")

    def cache_scope(self, context: GuardContext) -> str:
        return str(self.max_tokens)


class _LimitGuard(Guard):
    """Shared identity handling for per-user limits (rate limit, budget).

    The user is ``key(context)`` when given, else ``RAG(rate_limit_key=...)``, else the user id
    (``query(..., user_id=...)`` or ``UserContext.user_id``). Requests with no key follow
    ``anonymous_policy``: ``"skip"`` (default: not limited, recorded as ``mode="skipped"``),
    ``"shared"`` (one bucket for all anonymous requests) or ``"block"``. Counters live in a
    ``GuardStore`` (``InMemoryGuardStore`` by default: per process, reset on restart).
    """

    anonymous_message = "Please sign in to use this service."

    def __init__(
        self,
        key: Optional[Callable[[GuardContext], Optional[str]]] = None,
        anonymous_policy: str = "skip",
        store: Optional[GuardStore] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if anonymous_policy not in {"skip", "shared", "block"}:
            raise ValueError("anonymous_policy must be 'skip', 'shared' or 'block'")
        self.key = key
        self.anonymous_policy = anonymous_policy
        self.store = store

    def bind(self, rag: Any) -> None:
        if self.store is None:
            self.store = getattr(rag, "_guard_store", None)

    def config(self) -> Dict[str, Any]:
        settings = super().config()
        settings.pop("store", None)  # where counters live is not part of the guard's behaviour
        return settings

    def _store(self) -> GuardStore:
        if self.store is None:
            self.store = InMemoryGuardStore()
        return self.store

    def identity(self, context: GuardContext) -> Tuple[Optional[str], Optional[Decision]]:
        """``(store key, None)``, or ``(None, decision)`` when the request has no key."""

        key = self.key(context) if self.key is not None else context.user_key
        if key:
            return f"{self.name}:{key}", None
        if self.anonymous_policy == "shared":
            return f"{self.name}:anonymous", None
        if self.anonymous_policy == "block":
            return None, block(self.anonymous_message, reason="no user key (anonymous_policy='block')")
        return None, skipped("no user key (anonymous_policy='skip')")


class UserRateLimit(_LimitGuard):
    """At most ``max_requests`` questions per ``per_seconds`` per user (sliding window).

    See ``_LimitGuard`` for who "the user" is and what happens to anonymous requests. With the
    default in-memory store each worker process counts on its own: for a global limit use a
    shared ``GuardStore`` or your API gateway.
    """

    name = "user_rate_limit"
    block_message = "Too many requests. Please wait a moment and try again."

    def __init__(self, max_requests: int = 30, per_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic, **kwargs: Any):
        super().__init__(**kwargs)
        if max_requests <= 0 or per_seconds <= 0:
            raise ValueError("max_requests and per_seconds must be greater than zero")
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._clock = clock

    def check(self, value: Any, context: GuardContext) -> Decision:
        key, decision = self.identity(context)
        if key is None:
            return decision  # type: ignore[return-value]
        allowed, count, retry = self._store().hit(key, self._clock(), self.per_seconds, self.max_requests)
        if not allowed:
            return block(self.block_message, reason=f"{count} requests in {self.per_seconds:g}s; retry in {retry:.0f}s")
        return allow(reason=f"{count}/{self.max_requests} in window")


class UserDailyBudget(_LimitGuard):
    """Block a user once their spend today (UTC) reaches ``max_usd``.

    Spend is each answer's ``usage["cost_usd"]`` - the same pricing ``CostReport`` aggregates - so
    it needs ``RAG(prices=PriceTable(...))`` (or ``prices=`` here). Without prices the guard is
    skipped with a one-time ``GuardSkippedWarning``. A query is checked before it runs and charged
    after, so the last query of the day may overshoot the budget by its own cost.
    """

    name = "user_daily_budget"
    block_message = "You have reached today's usage limit. Please try again tomorrow."

    def __init__(self, max_usd: float = 1.0, prices: Any = None, today: Optional[Callable[[], str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if max_usd <= 0:
            raise ValueError("max_usd must be greater than zero")
        self.max_usd = float(max_usd)
        self.prices = prices
        self._today = today or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    def bind(self, rag: Any) -> None:
        super().bind(rag)
        if self.prices is None:
            self.prices = getattr(rag, "_prices", None)

    def spent(self, user_key: str) -> float:
        """USD spent today by ``user_key`` (the same key ``RAG`` puts in ``GuardContext.user_key``)."""

        return self._store().spent(f"{self.name}:{user_key}", self._today())

    def check(self, value: Any, context: GuardContext) -> Decision:
        if self.prices is None:
            warn_once(
                "user_daily_budget:prices",
                "UserDailyBudget skipped: no prices configured, so spend cannot be measured. "
                "Enable it with RAG(prices=PriceTable(llm={...}, embedding={...})).",
            )
            return skipped("no prices configured")
        key, decision = self.identity(context)
        if key is None:
            return decision  # type: ignore[return-value]
        spent = self._store().spent(key, self._today())
        if spent >= self.max_usd:
            return block(self.block_message, reason=f"spent ${spent:.4f} of ${self.max_usd:.4f} today")
        return allow(reason=f"spent ${spent:.4f} of ${self.max_usd:.4f} today")

    def after(self, result: Any, context: GuardContext) -> None:
        cost = (getattr(result, "usage", None) or {}).get("cost_usd")
        if not cost or self.prices is None:
            return
        key, _ = self.identity(context)
        if key is not None:
            self._store().add_spend(key, self._today(), float(cost))


class PIIMasker(Guard):
    """Replace e-mails, card numbers (Luhn), Aadhaar (Verhoeff) and PAN numbers in the question with
    typed placeholders (``<EMAIL_1>``, ``<CARD_1>``, ...). Phone numbers with ``phone=True``.

    Runs before every other guard (``order=-1``), so the raw values never reach traces,
    ``on_event``, ``LatencyStats`` tags, caches, conversation memory or any model. With
    ``restore=True`` (default) the values are put back into the final answer for the caller only:
    the placeholder map exists for that one request and is never traced, logged or cached
    (``output_dir`` files and cached answers keep the placeholders). ``action="block"`` refuses
    such questions instead.
    Trade-off: regexes (with checksums) miss names, addresses and unusual formats; a masked
    question can retrieve worse when the PII was the search key (e.g. "orders of jane@x.com").
    Placeholders are numbered per request, so in a multi-turn chat ``<EMAIL_1>`` in an earlier turn
    may stand for a different value.
    """

    name = "pii_mask"
    order = -1
    block_message = "Please do not include personal data (e-mail, ID or card numbers) in your question."

    def __init__(
        self,
        action: str = "mask",
        phone: bool = False,
        restore: bool = True,
        patterns: Optional[Sequence[Tuple[str, Any, Any]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if action not in {"mask", "block"}:
            raise ValueError("action must be 'mask' or 'block'")
        self.action = action
        self.phone = phone
        self.restore = restore
        self.patterns = tuple(patterns) if patterns is not None else (pii.ALL_PII_PATTERNS if phone else pii.PII_PATTERNS)

    def check(self, value: str, context: GuardContext) -> Decision:
        masked, mapping, counts = pii.mask_reversible(value, self.patterns)
        if not counts:
            return allow()
        reason = "found " + ", ".join(f"{label} x{n}" for label, n in sorted(counts.items()))
        if self.action == "block":
            return block(self.block_message, reason=reason)
        decision = modify(masked, reason=reason)
        if self.restore:
            decision.private = {"pii_map": mapping}
        return decision


class InjectionHeuristic(Guard):
    """Keyword heuristic for prompt-injection attempts in the question.

    Blocks when the confidence reaches ``block_confidence``: a strong pattern ("ignore previous
    instructions", fake delimiters) or two or more weaker ones. A single weak signal is allowed
    and recorded with its (low) confidence - the tier 1 classifier (next phase) escalates those.
    Trade-off: keywords catch copy-paste attacks, not paraphrased or encoded ones.
    """

    name = "injection_heuristic"
    cacheable = True
    block_message = "This question looks like an attempt to change the assistant's instructions, so it was not answered."
    STRONG = frozenset({"ignore_instructions", "fake_delimiter", "system_prompt"})

    def __init__(self, block_confidence: float = 0.8, **kwargs: Any):
        super().__init__(**kwargs)
        self.block_confidence = block_confidence

    def check(self, value: str, context: GuardContext) -> Decision:
        from raglib.core.prompt_safety import injection_signals

        signals = injection_signals(value)
        if not signals:
            return allow(confidence=0.9)  # no keywords: likely fine, never certain
        confidence = 0.9 if (self.STRONG & set(signals) or len(signals) >= 2) else 0.5
        reason = "signals: " + ", ".join(signals)
        if confidence >= self.block_confidence:
            return block(self.block_message, reason=reason, confidence=confidence)
        return allow(reason=reason + " (below block confidence)", confidence=1.0 - confidence)

    def cache_scope(self, context: GuardContext) -> str:
        return str(self.block_confidence)


# ----------------------------------------------------------------------------------------- retrieval

# Sensible defaults per score type: low enough to only stop clearly unrelated questions.
NOTHING_RELEVANT_THRESHOLDS: Dict[str, float] = {
    "dense": 0.15,  # cosine similarity of the query and chunk embeddings
    "cohere": 0.02,  # Cohere Rerank relevance (0..1)
    "cross_encoder": -7.0,  # ms-marco cross-encoder logit (sigmoid(-7) ~ 0.001)
}


class NothingRelevant(Guard):
    """Answer "couldn't find this in the documents" without calling the LLM when nothing relevant was retrieved.

    ``score="auto"`` uses the reranker score when a Cohere / cross-encoder reranker ran, else the
    dense cosine similarity (``metadata["dense_score"]``). BM25-only, fusion-only, web and tool
    results have no absolute relevance scale, so the check is skipped (allowed) for them.
    ``threshold`` is one number (for the chosen ``score``) or a dict per score type; defaults in
    ``NOTHING_RELEVANT_THRESHOLDS``. No documents at all also exits early (unless retrieval was
    skipped on purpose, e.g. small talk).
    Trade-off: saves the answer LLM call for off-topic questions; a threshold set too high refuses
    answerable questions - check ``result.sources[i].metadata`` scores on your own data first.
    """

    name = "nothing_relevant"
    kind = "quality"
    stages = ("retrieval",)
    block_message = "I couldn't find this in the documents."

    def __init__(self, threshold: Union[float, Mapping[str, float], None] = None, score: str = "auto", **kwargs: Any):
        super().__init__(**kwargs)
        if score not in {"auto", "dense", "rerank"}:
            raise ValueError("score must be 'auto', 'dense' or 'rerank'")
        self.score = score
        self.thresholds = dict(NOTHING_RELEVANT_THRESHOLDS)
        self.fixed: Optional[float] = None
        self._explicit = threshold is not None
        self._dense_unscaled = False  # the embedding has no semantic similarity scale (offline mock)
        if isinstance(threshold, Mapping):
            self.thresholds.update({str(k): float(v) for k, v in threshold.items()})
            self._explicit = "dense" in threshold
        elif threshold is not None:
            self.fixed = float(threshold)

    def bind(self, rag: Any) -> None:
        from raglib.embedding.mock_embedding import MockEmbedding
        from raglib.facade import unwrap_embedding

        embedding = unwrap_embedding(getattr(rag, "_embedding", None))
        self._dense_unscaled = isinstance(embedding, MockEmbedding) and not self._explicit

    def check(self, value: Sequence[Any], context: GuardContext) -> Decision:
        documents = list(value or [])
        if context.retrieval_skipped:
            return allow(reason="retrieval skipped on purpose")
        if not documents:
            return block(self.block_message, reason="no documents retrieved")
        measured = self._top_score(documents, context.reranker)
        if measured is None:
            if self._dense_unscaled:
                return skipped("offline MockEmbedding has no similarity scale (pass threshold= to check anyway)")
            return skipped("no absolute relevance score (BM25 / fusion / web / tool results)")
        score_type, top, threshold = measured
        if top < threshold:
            return block(self.block_message, reason=f"top {score_type} score {top:.3f} < {threshold:g}")
        return allow(reason=f"top {score_type} score {top:.3f} >= {threshold:g}")

    def _top_score(self, documents: Sequence[Any], reranker: Optional[str]) -> Optional[Tuple[str, float, float]]:
        def every(key: str) -> Optional[List[float]]:
            values = [doc.metadata.get(key) for doc in documents]
            if any(v is None for v in values):
                return None  # a document without the score (web / tool / graph) may still be relevant
            return [float(v) for v in values]

        if self.score in {"auto", "rerank"} and reranker in {"cohere", "cross_encoder"}:
            scores = every("rerank_score")
            if scores:
                threshold = self.fixed if self.fixed is not None else self.thresholds[reranker]
                return f"rerank:{reranker}", max(scores), threshold
            if self.score == "rerank":
                return None
        elif self.score == "rerank":
            scores = every("rerank_score")
            if scores and self.fixed is not None:
                return "rerank", max(scores), self.fixed
            return None
        scores = every("dense_score")
        if scores and not self._dense_unscaled:
            threshold = self.fixed if self.fixed is not None else self.thresholds["dense"]
            return "dense", max(scores), threshold
        return None


# -------------------------------------------------------------------------------------------- output


class MaxOutputTokens(Guard):
    """Cap the answer at ``max_tokens`` (``action="truncate"`` cuts it, ``"block"`` refuses it).

    Trade-off: on ``rag.query`` this limits what the user sees, not what you pay - the tokens were
    already generated. ``RAG(llm_max_tokens=...)`` (set by ``preset="production"``) is what caps
    cost: the model stops generating there. In ``rag.stream`` the truncating variant also stops
    reading the model's stream once the limit is reached.
    """

    name = "max_output_tokens"
    kind = "quality"
    stages = ("output",)
    block_message = "The answer was too long to show. Please ask a narrower question."

    def __init__(self, max_tokens: int = 1024, action: str = "truncate", token_counter: Optional[Callable[[str], int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if action not in {"truncate", "block"}:
            raise ValueError("action must be 'truncate' or 'block'")
        self.max_tokens, self.action, self.token_counter = max_tokens, action, token_counter

    @property
    def streaming_safe(self) -> bool:  # type: ignore[override]
        return self.action == "truncate"

    def check(self, value: str, context: GuardContext) -> Decision:
        tokens = _count(value, self.token_counter)
        if tokens <= self.max_tokens:
            return allow(reason=f"{tokens} tokens")
        if self.action == "block":
            return block(self.block_message, reason=f"{tokens} tokens > {self.max_tokens}")
        return modify(self._prefix(value, self.max_tokens).rstrip() + " …", reason=f"truncated {tokens} -> {self.max_tokens} tokens")

    def check_stream(self, chunk: str, context: GuardContext, state: Dict[str, Any]) -> Decision:
        used = state.get("tokens", 0)
        tokens = _count(chunk, self.token_counter)
        if used + tokens <= self.max_tokens:
            state["tokens"] = used + tokens
            return allow(reason=f"{used + tokens} tokens")
        state["exhausted"] = True
        kept = self._prefix(chunk, max(0, self.max_tokens - used)).rstrip()
        return modify(kept + " …", reason=f"truncated at {self.max_tokens} tokens")

    def _prefix(self, text: str, budget: int) -> str:
        words = text.split(" ")
        low, high = 0, len(words)
        while low < high:  # longest word prefix within the budget
            middle = (low + high + 1) // 2
            if _count(" ".join(words[:middle]), self.token_counter) <= budget:
                low = middle
            else:
                high = middle - 1
        return " ".join(words[:low])


class OutputLeakGuard(Guard):
    """Mask (or block) personal data and secrets in the answer.

    Secrets (API keys, cloud keys, private keys, JWTs) are always masked. Personal data that
    appears verbatim in the sources the user was allowed to retrieve is kept when
    ``allow_source_pii=True`` (a support e-mail in your docs is not a leak); anything else -
    e.g. PII the model recalled or invented - is masked. Phone numbers with ``phone=True``.
    Masking is streaming-safe (checked piece by piece in ``rag.stream``); ``action="block"`` is not.
    Trade-off: regex detection, see ``raglib.guards.pii``.
    """

    name = "output_leak"
    stages = ("output",)
    block_message = "The answer contained sensitive data and was withheld."

    def __init__(
        self,
        action: str = "mask",
        check_pii: bool = True,
        check_secrets: bool = True,
        allow_source_pii: bool = True,
        phone: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if action not in {"mask", "block"}:
            raise ValueError("action must be 'mask' or 'block'")
        self.action, self.check_pii, self.check_secrets, self.allow_source_pii = action, check_pii, check_secrets, allow_source_pii
        self.phone = phone

    @property
    def streaming_safe(self) -> bool:  # type: ignore[override]
        return self.action == "mask"

    def check(self, value: str, context: GuardContext) -> Decision:
        patterns: List[Any] = []
        if self.check_secrets:
            patterns.extend(pii.SECRET_PATTERNS)
        if self.check_pii:
            patterns.extend(pii.ALL_PII_PATTERNS if self.phone else pii.PII_PATTERNS)
        found = pii.find(value, patterns)
        if self.allow_source_pii and found:
            sources = "\n".join(getattr(doc, "content", "") for doc in context.documents)
            secret_labels = {label for label, _, _ in pii.SECRET_PATTERNS}
            found = [
                item for item in found
                if item[0] in secret_labels or value[item[1]:item[2]] not in sources
            ]
        if not found:
            return allow()
        counts: Dict[str, int] = {}
        for label, _, _ in found:
            counts[label] = counts.get(label, 0) + 1
        reason = "found " + ", ".join(f"{label} x{n}" for label, n in sorted(counts.items()))
        if self.action == "block":
            return block(self.block_message, reason=reason)
        pieces, position = [], 0
        for label, start, end in found:
            pieces.append(value[position:start] + f"[{label}]")
            position = end
        pieces.append(value[position:])
        return modify("".join(pieces), reason=reason)


_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


class CitationAccessGuard(Guard):
    """Citations may only reference documents the user can access; unknown ``[n]`` markers are removed.

    With access control on, every cited source is checked against the user's permission filter
    (defense in depth: retrieval already filters, but caches, custom retrievers or tools can
    slip). A cited document the user may not read blocks the answer (security: fail closed).
    A marker pointing past the source list (``[7]`` with 3 sources) is stripped from the text.
    Streaming-safe: in ``rag.stream`` a block stops the stream at the offending piece.

    A source's permission comes from its access tags (``access_level``, ``department``,
    ``allowed_roles``), so copies of a chunk that lost the internal ``_acl_*`` keys (e.g. GraphRAG's
    chunk view) are judged correctly. GraphRAG summaries (entities, relations, community reports,
    global points) carry the strictest tags of the chunks they were built from
    (``AccessControl.strictest``), so they are checked the same way; a summary without tags is
    denied, like any untagged document.

    ``summary_rule="all_sources"`` uses the more precise rule for summaries: visible when the user may
    read *every* source chunk (looked up by ``source_chunk_ids``), so a user in two departments can
    see a summary that mixes both. Summaries with more than 200 sources (ids not all listed) keep
    the strictest-tags rule. The default ``"strictest"`` needs no chunk lookups.
    """

    name = "citation_access"
    stages = ("output",)
    streaming_safe = True
    stream_holdback = 8  # only whole "[123]" markers matter: streams with about one word of lag
    block_message = "The answer referenced a document you do not have access to, so it was withheld."

    def __init__(self, summary_rule: str = "strictest", **kwargs: Any):
        super().__init__(**kwargs)
        if summary_rule not in {"strictest", "all_sources"}:
            raise ValueError("summary_rule must be 'strictest' or 'all_sources'")
        self.summary_rule = summary_rule

    def bind(self, rag: Any) -> None:
        self._access = getattr(rag, "_access", None)
        retriever = getattr(rag, "_retriever", None)
        self._chunk_lookup = getattr(retriever, "get", None)

    def _visible(self, document: Any, permission_filter: Mapping[str, Any]) -> bool:
        from raglib.vectorstores.base_store import matches_filter

        metadata = dict(getattr(document, "metadata", {}) or {})
        access = getattr(self, "_access", None)
        ids = metadata.get("source_chunk_ids")
        lookup = getattr(self, "_chunk_lookup", None)
        if (
            self.summary_rule == "all_sources" and ids and lookup is not None
            and metadata.get("source_chunk_count", len(ids)) == len(ids)
        ):
            chunks = [lookup(chunk_id) for chunk_id in ids]
            return all(
                chunk is not None and self._visible(chunk, permission_filter) for chunk in chunks
            )
        if access is not None:
            metadata = access.tag(metadata)  # re-derive the _acl_* keys from the access tags
        return matches_filter(metadata, permission_filter, array_contains=True)

    def check(self, value: str, context: GuardContext) -> Decision:
        sources = list(context.documents)
        cited = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(value or "")})
        if context.permission_filter is not None:
            denied = [
                n for n in cited
                if 1 <= n <= len(sources) and not self._visible(sources[n - 1], context.permission_filter)
            ]
            if denied:
                return block(self.block_message, reason=f"cited inaccessible source(s) {denied}")
        invalid = [n for n in cited if not 1 <= n <= len(sources)]
        if invalid:
            cleaned = _CITATION_RE.sub(lambda m: "" if int(m.group(1)) in invalid else m.group(0), value)
            return modify(cleaned, reason=f"removed unknown citation(s) {invalid}")
        return allow(reason=f"{len(cited)} citation(s) checked")


class JSONSchemaGuard(Guard):
    """Require the answer to be JSON matching ``schema`` (uses ``jsonschema`` when installed).

    Without ``jsonschema`` a built-in subset is checked: ``type``, ``required``, ``properties``,
    ``items``, ``enum``. A code-fenced answer (```json ... ```) is unwrapped first. Failures
    block with ``message`` (``kind="quality"``: a guard error fails open).
    """

    name = "json_schema"
    kind = "quality"
    stages = ("output",)
    block_message = "The answer was not in the expected format."

    def __init__(self, schema: Mapping[str, Any], **kwargs: Any):
        super().__init__(**kwargs)
        self.schema = dict(schema)

    def check(self, value: str, context: GuardContext) -> Decision:
        text = (value or "").strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced:
            text = fenced.group(1)
        try:
            data = json.loads(text)
        except ValueError as exc:
            return block(self.block_message, reason=f"not JSON ({exc.msg})")
        errors = _validate(data, self.schema)
        if errors:
            return block(self.block_message, reason="; ".join(errors[:3]))
        if text != (value or "").strip():
            return modify(text, reason="unwrapped code fence")
        return allow(reason="valid JSON")


_JSON_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
    "number": (int, float), "integer": int,
}


def _validate(data: Any, schema: Mapping[str, Any]) -> List[str]:
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return _validate_subset(data, schema, "$")
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path)) or '$'}: {e.message}" for e in validator.iter_errors(data)]


def _validate_subset(data: Any, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        ok = False
        for name in types:
            python = _JSON_TYPES.get(name)
            if python is None:
                ok = True
            elif name in {"number", "integer"} and isinstance(data, bool):
                continue
            elif isinstance(data, python):
                ok = True
        if not ok:
            return [f"{path}: expected {expected}"]
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: not one of {schema['enum']}")
    if isinstance(data, dict):
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: missing '{key}'")
        for key, sub in (schema.get("properties") or {}).items():
            if key in data:
                errors.extend(_validate_subset(data[key], sub, f"{path}.{key}"))
    if isinstance(data, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(data):
            errors.extend(_validate_subset(item, schema["items"], f"{path}[{index}]"))
    return errors


# --------------------------------------------------------------------------------------------- tools


class ToolPolicy(Guard):
    """What the agent may call: tool allowlist per role, calls per question, argument limits.

    ``allowed_tools``: ``{"role": ["search_docs", ...]}``; a user gets the union over their roles.
    ``default_tools`` applies to users without a matching role (``None`` = every tool, ``[]`` = none).
    ``max_tool_calls`` caps tool calls per question; ``max_argument_chars`` caps the argument
    length; ``argument_checks={"tool": fn(argument) -> bool}`` adds your own rules (False blocks).
    A blocked call is reported to the agent as the tool's observation, so it can try something else.
    """

    name = "tool_policy"
    stages = ("tool",)
    block_message = "This tool call is not allowed."

    def __init__(
        self,
        allowed_tools: Optional[Mapping[str, Sequence[str]]] = None,
        default_tools: Optional[Sequence[str]] = None,
        max_tool_calls: Optional[int] = 8,
        max_argument_chars: Optional[int] = 1000,
        argument_checks: Optional[Mapping[str, Callable[[str], bool]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.allowed_tools = {role: set(names) for role, names in (allowed_tools or {}).items()}
        self.default_tools = None if default_tools is None else set(default_tools)
        self.max_tool_calls = max_tool_calls
        self.max_argument_chars = max_argument_chars
        self.argument_checks = dict(argument_checks or {})

    def allowed_for(self, user: Any) -> Optional[Set[str]]:
        """Tool names ``user`` may call (``None`` = no restriction)."""

        roles = set(getattr(user, "roles", ()) or ())
        granted: Set[str] = set()
        matched = False
        for role in roles:
            if role in self.allowed_tools:
                matched = True
                granted |= self.allowed_tools[role]
        if matched:
            return granted
        return None if self.default_tools is None else set(self.default_tools)

    def filter_tools(self, tools: Mapping[str, Any], user: Any) -> Dict[str, Any]:
        allowed = self.allowed_for(user)
        return dict(tools) if allowed is None else {name: tool for name, tool in tools.items() if name in allowed}

    def check(self, value: Tuple[str, str, int], context: GuardContext) -> Decision:
        name, argument, calls_so_far = value
        allowed = self.allowed_for(context.user)
        if allowed is not None and name not in allowed:
            return block(f"The tool '{name}' is not allowed for this user.", reason=f"tool '{name}' not in allowlist")
        if self.max_tool_calls is not None and calls_so_far >= self.max_tool_calls:
            return block("The tool call limit for this question was reached.", reason=f"call limit: {calls_so_far} calls >= {self.max_tool_calls}")
        argument = argument or ""
        if self.max_argument_chars is not None and len(argument) > self.max_argument_chars:
            return block("The tool argument was too long.", reason=f"argument {len(argument)} chars > {self.max_argument_chars}")
        check = self.argument_checks.get(name)
        if check is not None and not check(argument):
            return block(f"The argument for '{name}' was rejected.", reason=f"argument check for '{name}' failed")
        return allow(reason=f"call {calls_so_far + 1}")


# ------------------------------------------------------------------------------------------ defaults


def default_input_guards() -> List[Guard]:
    """Production defaults: question length, PII masking, injection keywords (rate limits and
    budgets need your numbers, so they are not on by default)."""

    return [PIIMasker(), MaxQuestionTokens(), InjectionHeuristic()]


def default_retrieval_guards() -> List[Guard]:
    return [NothingRelevant()]


def default_output_guards() -> List[Guard]:
    return [OutputLeakGuard(), CitationAccessGuard()]


_BY_NAME: Dict[str, Callable[[], Guard]] = {
    "max_question_tokens": MaxQuestionTokens,
    "pii": PIIMasker,
    "pii_mask": PIIMasker,
    "injection": InjectionHeuristic,
    "injection_heuristic": InjectionHeuristic,
    "rate_limit": UserRateLimit,
    "user_rate_limit": UserRateLimit,
    "budget": UserDailyBudget,
    "user_daily_budget": UserDailyBudget,
    "nothing_relevant": NothingRelevant,
    "max_output_tokens": MaxOutputTokens,
    "output_leak": OutputLeakGuard,
    "citation_access": CitationAccessGuard,
}

# Tier 1 model guards by name (opt-in; TopicGuard needs your topics, so it has no name).
_MODEL_GUARDS = {
    "injection_classifier": "InjectionClassifier",
    "toxicity": "ToxicityClassifier",
    "grounding": "GroundingGuard",
}

_DEFAULTS = {"input": default_input_guards, "retrieval": default_retrieval_guards, "output": default_output_guards}


def resolve_guards(spec: Union[str, Guard, Sequence[Union[str, Guard]], None], stage: str) -> List[Guard]:
    """``None`` -> no guards; ``"default"`` -> the stage's defaults; names and ``Guard`` objects mix."""

    if spec is None or spec is False:
        return []
    if isinstance(spec, (str, Guard)):
        spec = [spec]
    guards: List[Guard] = []
    for item in spec:
        if isinstance(item, Guard):
            guards.append(item)
        elif item == "default":
            guards.extend(_DEFAULTS[stage]())
        elif isinstance(item, str) and item in _BY_NAME:
            guards.append(_BY_NAME[item]())
        elif isinstance(item, str) and item in _MODEL_GUARDS:
            from raglib.guards import classifiers

            guards.append(getattr(classifiers, _MODEL_GUARDS[item])())
        else:
            raise ValueError(f"Unknown guard {item!r}; use a Guard object, 'default', or one of {sorted({*_BY_NAME, *_MODEL_GUARDS})}")
    wrong = [g.name for g in guards if stage not in g.stages]
    if wrong:
        raise ValueError(f"{stage}_guards got guards for another stage: {wrong}")
    return guards
