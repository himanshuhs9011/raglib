"""Incremental output guards for ``rag.stream``: check the answer piece by piece, keep streaming.

Tokens go into a pending buffer. A piece is released (checked by every streaming-safe output
guard, then yielded) when it ends at a completed sentence, or - for long sentences - at a space
at least ``holdback`` characters before the end of the buffer. The release point is moved back
so it never cuts through anything a guard looks for (an API key, a card number, a ``[n]``
citation, a ``<EMAIL_1>`` placeholder): a secret split across tokens is still checked whole.

``holdback`` (``RAG(stream_holdback_chars=...)``, default ``DEFAULT_STREAM_HOLDBACK`` = 64) must be
at least ``MIN_STREAM_HOLDBACK`` = 40: the longest text that can still *become* a secret while
containing a space is a PEM header (``-----BEGIN ENCRYPTED PRIVATE KEY-----``, 37 characters; a
card number with separators is at most 23, a placeholder 24). Below that, a space inside a
not-yet-complete header or card number could be released before the pattern matches.

Trade-off: the text shown lags the model by up to one sentence (or ``holdback`` characters).
A block mid-stream cannot take back the pieces already shown: the stream stops and the guard's
message is appended.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from raglib.core.tracing import record_stage
from raglib.guards.base import BLOCK, MODIFY, Decision, Guard, GuardContext, guard_sort_key, run_guarded
from raglib.guards.pii import ALL_PII_PATTERNS, PLACEHOLDER_RE, SECRET_PATTERNS

_SENTENCE_END = re.compile(r"[.!?;:][\"')\]*]*\s+|\n\s*\n")
_GUARDED = [pattern for _, pattern, _ in (*SECRET_PATTERNS, *ALL_PII_PATTERNS)] + [PLACEHOLDER_RE, re.compile(r"\[\d{1,3}\]")]
_RANK = {"allow": 0, "modify": 1, "block": 2}
MIN_STREAM_HOLDBACK = 40
DEFAULT_STREAM_HOLDBACK = 64


def check_holdback(chars: int) -> int:
    """Validate ``stream_holdback_chars`` (see the module docstring for the minimum)."""

    if not isinstance(chars, int) or chars < MIN_STREAM_HOLDBACK:
        raise ValueError(
            f"stream_holdback_chars must be an integer >= {MIN_STREAM_HOLDBACK} (the longest secret prefix "
            "that can contain a space, a PEM header), got " + repr(chars)
        )
    return chars


class StreamingOutput:
    """Release checked pieces of a streamed answer (see the module docstring)."""

    def __init__(self, guards: Sequence[Guard], context: GuardContext, holdback: int = DEFAULT_STREAM_HOLDBACK):
        self.guards = sorted(guards, key=guard_sort_key)
        self.context = context
        configured = check_holdback(holdback)
        # Only guards that look for secrets need the full window; citation checks need a few chars.
        needs = [guard.stream_holdback if guard.stream_holdback is not None else configured for guard in self.guards]
        self.holdback = max(needs) if needs else configured
        self.blocked: Optional[Decision] = None
        self.stopped = False  # True after a block or once MaxOutputTokens is exhausted
        self._pending = ""
        self._shown: List[str] = []
        self._states: Dict[str, Dict[str, Any]] = {guard.name: {} for guard in self.guards}
        self._summary: Dict[str, Decision] = {}

    @property
    def answer(self) -> str:
        """What the user saw (the block message included)."""

        return "".join(self._shown)

    def feed(self, token: str) -> str:
        if self.stopped:
            return ""
        self._pending += token
        return self._release(self._cut())

    def close(self) -> str:
        """Release (and check) whatever is still pending; call once the model is done."""

        if self.stopped:
            return ""
        return self._release(len(self._pending))

    def decisions(self) -> List[Decision]:
        """One decision per guard for the whole stream; each is also traced as ``guard.<name>``."""

        out = []
        for guard in self.guards:
            decision = self._summary.get(guard.name)
            if decision is None:
                continue
            record_stage(
                f"guard.{guard.name}", decision.latency_ms, stage="output", tier=decision.tier, action=decision.action,
                mode=decision.mode, confidence=round(decision.confidence, 3), cached=False, reason=decision.reason,
                streamed=True, **({"error": decision.error} if decision.error else {}),
            )
            out.append(decision)
        return out

    # ------------------------------------------------------------------ internals

    def _cut(self) -> int:
        text = self._pending
        cut = 0
        for match in _SENTENCE_END.finditer(text):
            cut = match.end()
        limit = len(text) - self.holdback
        if limit > cut:
            space = max(text.rfind(" ", 0, limit + 1), text.rfind("\n", 0, limit + 1))
            if space + 1 > cut:
                cut = space + 1
        if cut <= 0:
            return 0
        moved = True
        while moved and cut > 0:  # never cut through something a guard looks for
            moved = False
            for pattern in _GUARDED:
                for match in pattern.finditer(text):
                    if match.start() < cut < match.end():
                        cut, moved = match.start(), True
        return cut

    def _release(self, cut: int) -> str:
        if cut <= 0:
            return ""
        piece, self._pending = self._pending[:cut], self._pending[cut:]
        for guard in self.guards:
            state = self._states[guard.name]
            started = time.perf_counter()
            decision = run_guarded(guard, lambda g=guard, p=piece, s=state: g.check_stream(p, self.context, s), piece)
            decision.guard, decision.stage = guard.name, "output"
            decision.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            self._merge(decision)
            if decision.action == MODIFY:
                piece = decision.value
            elif decision.action == BLOCK:
                self.blocked, self.stopped = decision, True
                shown = "".join(self._shown)
                message = decision.message or guard.block_message
                tail = ("\n\n" if shown else "") + message
                self._shown.append(tail)
                return tail
            if state.get("exhausted"):
                self.stopped = True
        self._shown.append(piece)
        return piece

    def _merge(self, decision: Decision) -> None:
        previous = self._summary.get(decision.guard)
        if previous is None:
            self._summary[decision.guard] = replace(decision, value=None, private=None)
            return
        reasons = [r for r in (previous.reason, decision.reason) if r]
        merged = previous if _RANK[previous.action] >= _RANK[decision.action] else decision
        self._summary[decision.guard] = replace(
            merged,
            value=None,
            private=None,
            reason="; ".join(dict.fromkeys(reasons))[:300],
            latency_ms=round(previous.latency_ms + decision.latency_ms, 3),
            confidence=min(previous.confidence, decision.confidence),
            error=previous.error or decision.error,
        )


def streaming_mode(guards: Sequence[Guard]) -> str:
    """``"none"`` (no output guards), ``"incremental"`` (all streaming-safe) or ``"buffered"``."""

    if not guards:
        return "none"
    return "incremental" if all(guard.streaming_safe for guard in guards) else "buffered"


def stream_context(context: GuardContext, question: str, documents: Sequence[Any]) -> GuardContext:
    return replace(context, stage="output", question=question, documents=list(documents))
