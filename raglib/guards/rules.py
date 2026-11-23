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
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from raglib.guards import pii
from raglib.guards.base import Decision, Guard, GuardContext, allow, block, modify

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


class UserRateLimit(Guard):
    """At most ``max_requests`` questions per ``per_seconds`` per user (sliding window, in process).

    Users are keyed by ``UserContext.user_id``, else the ``session_id``, else ``"anonymous"`` (one
    shared bucket - pass a user or a session id per caller). Limits live in process memory: with
    several workers each worker has its own window (use your API gateway for a global limit).
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
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, value: Any, context: GuardContext) -> Decision:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(context.user_key, deque())
            while hits and now - hits[0] >= self.per_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry = max(0.0, self.per_seconds - (now - hits[0]))
                return block(self.block_message, reason=f"{len(hits)} requests in {self.per_seconds:g}s; retry in {retry:.0f}s")
            hits.append(now)
            if len(self._hits) > 100_000:  # drop idle users so the table cannot grow without bound
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.per_seconds][:50_000]:
                    self._hits.pop(key, None)
            return allow(reason=f"{len(hits)}/{self.max_requests} in window")


class UserDailyBudget(Guard):
    """Block a user once their spend today (UTC) reaches ``max_usd``.

    Spend is each answer's ``usage["cost_usd"]`` - the same pricing ``CostReport`` aggregates - so
    it needs ``RAG(prices=PriceTable(...))`` (or ``prices=`` here). Without prices the guard is
    skipped with a one-time ``GuardSkippedWarning``. A query is checked before it runs and charged
    after, so the last query of the day may overshoot the budget by its own cost. In process memory.
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
        self._spent: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def bind(self, rag: Any) -> None:
        if self.prices is None:
            self.prices = getattr(rag, "_prices", None)

    def spent(self, user_key: str) -> float:
        """USD spent today by ``user_key``."""

        with self._lock:
            return self._spent.get((user_key, self._today()), 0.0)

    def check(self, value: Any, context: GuardContext) -> Decision:
        if self.prices is None:
            warn_once(
                "user_daily_budget:prices",
                "UserDailyBudget skipped: no prices configured, so spend cannot be measured. "
                "Enable it with RAG(prices=PriceTable(llm={...}, embedding={...})).",
            )
            return skipped("no prices configured")
        spent = self.spent(context.user_key)
        if spent >= self.max_usd:
            return block(self.block_message, reason=f"spent ${spent:.4f} of ${self.max_usd:.4f} today")
        return allow(reason=f"spent ${spent:.4f} of ${self.max_usd:.4f} today")

    def after(self, result: Any, context: GuardContext) -> None:
        cost = (getattr(result, "usage", None) or {}).get("cost_usd")
        if not cost:
            return
        day = self._today()
        with self._lock:
            key = (context.user_key, day)
            self._spent[key] = self._spent.get(key, 0.0) + float(cost)
            stale = [k for k in self._spent if k[1] != day]
            for k in stale:
                del self._spent[k]


class PIIMasker(Guard):
    """Mask e-mails, phone numbers, Aadhaar, PAN and card numbers in the question (``[EMAIL]``, ...).

    Runs first, so the raw values never reach traces, ``on_event``, ``LatencyStats`` tags, caches,
    conversation memory or any model. ``action="block"`` refuses such questions instead.
    Trade-off: regexes (with checksums) miss names, addresses and unusual formats; a masked
    question can retrieve worse when the PII was the search key (e.g. "orders of jane@x.com").
    """

    name = "pii_mask"
    block_message = "Please do not include personal data (e-mail, phone, ID or card numbers) in your question."

    def __init__(self, action: str = "mask", patterns: Sequence[Tuple[str, Any, Any]] = pii.PII_PATTERNS, **kwargs: Any):
        super().__init__(**kwargs)
        if action not in {"mask", "block"}:
            raise ValueError("action must be 'mask' or 'block'")
        self.action = action
        self.patterns = patterns

    def check(self, value: str, context: GuardContext) -> Decision:
        masked, counts = pii.mask(value, self.patterns)
        if not counts:
            return allow()
        reason = "found " + ", ".join(f"{label} x{n}" for label, n in sorted(counts.items()))
        if self.action == "block":
            return block(self.block_message, reason=reason)
        return modify(masked, reason=reason)


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

    Trade-off: this limits what the user sees, not what you pay - the tokens were already
    generated. To cap cost, set ``RAG(llm_max_tokens=...)`` as well.
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

    def check(self, value: str, context: GuardContext) -> Decision:
        tokens = _count(value, self.token_counter)
        if tokens <= self.max_tokens:
            return allow(reason=f"{tokens} tokens")
        if self.action == "block":
            return block(self.block_message, reason=f"{tokens} tokens > {self.max_tokens}")
        words = value.split(" ")
        low, high = 0, len(words)
        while low < high:  # longest word prefix within the limit
            middle = (low + high + 1) // 2
            if _count(" ".join(words[:middle]), self.token_counter) <= self.max_tokens:
                low = middle
            else:
                high = middle - 1
        return modify(" ".join(words[:low]).rstrip() + " …", reason=f"truncated {tokens} -> {self.max_tokens} tokens")


class OutputLeakGuard(Guard):
    """Mask (or block) personal data and secrets in the answer.

    Secrets (API keys, cloud keys, private keys, JWTs) are always masked. Personal data that
    appears verbatim in the sources the user was allowed to retrieve is kept when
    ``allow_source_pii=True`` (a support e-mail in your docs is not a leak); anything else -
    e.g. PII the model recalled or invented - is masked.
    Trade-off: regex detection, see ``raglib.guards.pii``.
    """

    name = "output_leak"
    stages = ("output",)
    block_message = "The answer contained sensitive data and was withheld."

    def __init__(self, action: str = "mask", check_pii: bool = True, check_secrets: bool = True, allow_source_pii: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        if action not in {"mask", "block"}:
            raise ValueError("action must be 'mask' or 'block'")
        self.action, self.check_pii, self.check_secrets, self.allow_source_pii = action, check_pii, check_secrets, allow_source_pii

    def check(self, value: str, context: GuardContext) -> Decision:
        patterns: List[Any] = []
        if self.check_secrets:
            patterns.extend(pii.SECRET_PATTERNS)
        if self.check_pii:
            patterns.extend(pii.PII_PATTERNS)
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
    """

    name = "citation_access"
    stages = ("output",)
    block_message = "The answer referenced a document you do not have access to, so it was withheld."

    def check(self, value: str, context: GuardContext) -> Decision:
        from raglib.vectorstores.base_store import matches_filter

        sources = list(context.documents)
        cited = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(value or "")})
        if context.permission_filter is not None:
            denied = [
                n for n in cited
                if 1 <= n <= len(sources)
                and not matches_filter(getattr(sources[n - 1], "metadata", {}) or {}, context.permission_filter, array_contains=True)
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
        else:
            raise ValueError(f"Unknown guard {item!r}; use a Guard object, 'default', or one of {sorted(_BY_NAME)}")
    wrong = [g.name for g in guards if stage not in g.stages]
    if wrong:
        raise ValueError(f"{stage}_guards got guards for another stage: {wrong}")
    return guards
