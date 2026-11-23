"""Secure defaults: the protections every ``RAG(...)`` gets, which never change a normal answer.

* ``citation_access`` - ``CitationAccessGuard`` whenever ``access_control`` is on (an answer that
  cites a document the user may not read is withheld; with correct retrieval filters this never
  happens, so normal answers are unchanged).
* ``tool_limit`` - ``ToolPolicy(max_tool_calls=10)`` for the agent: stops a looping agent. The agent's
  own ``max_steps`` (default 4) stays far below it, so normal runs are unchanged.
* ``injection_flags`` - ``InjectionHeuristic(flag_only=True)`` on the question.
* ``secret_flags`` - ``OutputLeakGuard(check_pii=False, flag_only=True)``: API keys, tokens and
  private keys in the answer.

The two flag-only guards never block or modify: their findings go to ``result.guard_flags`` and a
``guard_flag`` event on ``on_event``. Flag-only output guards run once on the finished answer, so
``rag.stream`` still yields token by token. Source delimiters / escaping in every prompt and the
agent's web-search argument guard are not guards: they are always applied.

Turn one off with ``RAG(secure_defaults={"injection_flags": False})``, all with
``secure_defaults=False``. A guard of the same class that you configure yourself (e.g. a blocking
``InjectionHeuristic`` in ``input_guards``) replaces the default one.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from raglib.guards.base import Guard
from raglib.guards.rules import CitationAccessGuard, InjectionHeuristic, OutputLeakGuard, ToolPolicy, resolve_guards

logger = logging.getLogger("raglib.guards")

SECURE_DEFAULTS = ("citation_access", "tool_limit", "injection_flags", "secret_flags")
DEFAULT_MAX_TOOL_CALLS = 10
GUARD_TIP_ENV = "RAGLIB_GUARD_TIP"
GUARDRAILS_DOCS_URL = "https://github.com/himanshuhs9011/raglib#guardrails"

_tip_lock = threading.Lock()
_tip_shown = False


def secure_defaults_options(value: Union[bool, Mapping[str, bool], None]) -> Dict[str, bool]:
    """``True`` / ``None`` -> all on, ``False`` -> all off, a mapping -> per protection (missing = on)."""

    if value is None or value is True:
        return dict.fromkeys(SECURE_DEFAULTS, True)
    if value is False:
        return dict.fromkeys(SECURE_DEFAULTS, False)
    if not isinstance(value, Mapping):
        raise TypeError("secure_defaults must be True, False or a dict like {'injection_flags': False}")
    unknown = set(value) - set(SECURE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown secure_defaults keys {sorted(unknown)}; known: {list(SECURE_DEFAULTS)}")
    return {name: bool(value.get(name, True)) for name in SECURE_DEFAULTS}


def _has(guards: List[Guard], kind: type) -> bool:
    return any(isinstance(guard, kind) for guard in guards)


def with_secure_defaults(
    options: Mapping[str, bool],
    access_control: bool,
    input_guards: Any = None,
    retrieval_guards: Any = None,
    output_guards: Any = None,
    tool_policy: Any = None,
) -> Tuple[List[Guard], List[Guard], List[Guard], Optional[Any]]:
    """The configured guards plus the enabled secure defaults (never duplicating a configured kind)."""

    inputs = resolve_guards(input_guards, "input")
    retrievals = resolve_guards(retrieval_guards, "retrieval")
    outputs = resolve_guards(output_guards, "output")
    if options.get("injection_flags") and not _has(inputs, InjectionHeuristic):
        inputs.append(InjectionHeuristic(flag_only=True))
    if options.get("secret_flags") and not _has(outputs, OutputLeakGuard):
        outputs.append(OutputLeakGuard(check_pii=False, flag_only=True))
    if options.get("citation_access") and access_control and not _has(outputs, CitationAccessGuard):
        outputs.append(CitationAccessGuard())
    if tool_policy is None and options.get("tool_limit"):
        tool_policy = ToolPolicy(max_tool_calls=DEFAULT_MAX_TOOL_CALLS, max_argument_chars=None)
    return inputs, retrievals, outputs, tool_policy


def guard_tip() -> None:
    """Log once per process (INFO) where to find the guardrails; ``RAGLIB_GUARD_TIP=0`` silences it."""

    global _tip_shown
    if os.getenv(GUARD_TIP_ENV, "").strip().lower() in {"0", "false", "no", "off"}:
        return
    with _tip_lock:
        if _tip_shown:
            return
        _tip_shown = True
    logger.info(
        "raglib: only the always-on protections are active (injection and secret flags, citation access with "
        "access_control, a %d-call agent tool limit). For PII masking, a no-LLM 'not in the documents' exit and "
        "leak masking use preset='production' or input_guards= / output_guards=. See %s "
        "(set %s=0 to hide this message).",
        DEFAULT_MAX_TOOL_CALLS, GUARDRAILS_DOCS_URL, GUARD_TIP_ENV,
    )


def reset_guard_tip() -> None:
    """Show the tip again on the next ``RAG(...)`` (tests)."""

    global _tip_shown
    with _tip_lock:
        _tip_shown = False
