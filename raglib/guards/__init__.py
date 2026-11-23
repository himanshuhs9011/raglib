"""Layered guardrails, cheapest first.

Configure them on ``RAG``::

    from raglib import RAG
    from raglib.guards import PIIMasker, MaxQuestionTokens, UserRateLimit, NothingRelevant, OutputLeakGuard

    rag = RAG(
        "docs/",
        input_guards=[PIIMasker(), MaxQuestionTokens(500), UserRateLimit(20, per_seconds=60)],
        retrieval_guards=[NothingRelevant(message="Sorry, that is not covered by our documents.")],
        output_guards="default",            # PII / secret leak check + citation access check
    )

``preset="production"`` turns on the defaults for every stage. Each guard returns allow / block
(with a user-facing message) / modify (e.g. masked text) and a confidence; every decision is in
``result.metadata["guards"]``, the trace (stage ``guard.<name>``), ``on_event``, ``LatencyStats``
and ``CostReport``. Tier 0 (this module's rules) costs nothing; tier 1 and 2 checks (classifiers,
LLM judge) come in later releases and only run when the cheaper tiers are unsure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

from raglib.guards.base import (
    ALLOW,
    BLOCK,
    MODIFY,
    Decision,
    Guard,
    GuardContext,
    GuardRunner,
    GuardScope,
    allow,
    block,
    current_scope,
    guard_scope,
    modify,
)
from raglib.guards.classifiers import GroundingGuard, InjectionClassifier, ModelGuard, TopicGuard, ToxicityClassifier
from raglib.guards.judge import JudgeBudget, LLMJudge
from raglib.guards.models import RECOMMENDED_LOCAL_MODELS, ModerationClient
from raglib.guards.parallel import GuardCancelled, ParallelInputGuards
from raglib.guards.rules import (
    NOTHING_RELEVANT_THRESHOLDS,
    CitationAccessGuard,
    GuardSkippedWarning,
    InjectionHeuristic,
    JSONSchemaGuard,
    MaxOutputTokens,
    MaxQuestionTokens,
    NothingRelevant,
    OutputLeakGuard,
    PIIMasker,
    ToolPolicy,
    UserDailyBudget,
    UserRateLimit,
    default_input_guards,
    default_output_guards,
    default_retrieval_guards,
    resolve_guards,
)
from raglib.guards.store import GuardStore, InMemoryGuardStore
from raglib.guards.streaming import StreamingOutput

GuardSpec = Union[str, Guard, Sequence[Union[str, Guard]], None]


@dataclass
class GuardConfig:
    """The guards of one ``RAG`` instance (built from its ``*_guards`` options)."""

    input: List[Guard] = field(default_factory=list)
    retrieval: List[Guard] = field(default_factory=list)
    output: List[Guard] = field(default_factory=list)
    tool_policy: Optional[ToolPolicy] = None
    runner: GuardRunner = field(default_factory=GuardRunner)

    @classmethod
    def build(
        cls,
        input_guards: GuardSpec = None,
        retrieval_guards: GuardSpec = None,
        output_guards: GuardSpec = None,
        tool_policy: Union[ToolPolicy, str, None] = None,
        cache_ttl: Optional[float] = 3600.0,
    ) -> Optional["GuardConfig"]:
        """``None`` when no guard is configured (queries then take the unguarded path)."""

        if tool_policy == "default":
            tool_policy = ToolPolicy()
        elif tool_policy is not None and not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy, 'default' or None")
        config = cls(
            input=resolve_guards(input_guards, "input"),
            retrieval=resolve_guards(retrieval_guards, "retrieval"),
            output=resolve_guards(output_guards, "output"),
            tool_policy=tool_policy,
            runner=GuardRunner(cache_ttl=cache_ttl),
        )
        if not (config.input or config.retrieval or config.output or config.tool_policy):
            return None
        return config

    def all(self) -> List[Guard]:
        return [*self.input, *self.retrieval, *self.output, *([self.tool_policy] if self.tool_policy else [])]

    def bind(self, rag: Any) -> None:
        for guard in self.all():
            guard.bind(rag)

    def signature(self) -> str:
        """Hash of every guard's class and settings: part of the answer-cache key, so adding,
        removing or changing a guard never serves answers cached under the old configuration."""

        description = {
            "input": [guard.config() for guard in self.input],
            "retrieval": [guard.config() for guard in self.retrieval],
            "output": [guard.config() for guard in self.output],
            "tool_policy": self.tool_policy.config() if self.tool_policy is not None else None,
        }
        raw = json.dumps(description, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ALLOW",
    "BLOCK",
    "MODIFY",
    "NOTHING_RELEVANT_THRESHOLDS",
    "CitationAccessGuard",
    "Decision",
    "Guard",
    "GuardConfig",
    "GuardContext",
    "GuardRunner",
    "GuardScope",
    "GroundingGuard",
    "GuardCancelled",
    "GuardSkippedWarning",
    "GuardStore",
    "InMemoryGuardStore",
    "StreamingOutput",
    "InjectionClassifier",
    "JudgeBudget",
    "LLMJudge",
    "InjectionHeuristic",
    "JSONSchemaGuard",
    "MaxOutputTokens",
    "MaxQuestionTokens",
    "ModelGuard",
    "ModerationClient",
    "NothingRelevant",
    "OutputLeakGuard",
    "PIIMasker",
    "ParallelInputGuards",
    "RECOMMENDED_LOCAL_MODELS",
    "ToolPolicy",
    "TopicGuard",
    "ToxicityClassifier",
    "UserDailyBudget",
    "UserRateLimit",
    "allow",
    "block",
    "current_scope",
    "default_input_guards",
    "default_output_guards",
    "default_retrieval_guards",
    "guard_scope",
    "modify",
    "resolve_guards",
]
