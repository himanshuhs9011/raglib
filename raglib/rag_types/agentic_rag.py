"""Agentic RAG: an LLM agent loop (ReAct-style) that chooses tools until it has enough evidence."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from raglib.components.tools import Tool
from raglib.core.base import BaseRAG, PipelineContext
from raglib.core.fusion import interleave
from raglib.schemas import Document

logger = logging.getLogger(__name__)

FINISH = "finish"


class AgenticRAG(BaseRAG):
    """Think -> act (call a tool) -> observe, for up to ``max_steps``; then answer with citations.

    Built-in tools are added automatically when their backend is configured:
    ``search_docs`` (corpus), ``search_web`` (web provider) and ``graph_search`` (GraphRAG).
    Pass ``tools=[Tool(...)]`` to add your own (APIs, SQL, calculators, ...).
    The agent speaks a provider-agnostic JSON action protocol, so it works with every LLM.
    """

    pipeline_name = "agentic"
    uses_decision_engine = False

    def __init__(self, tools: Optional[Sequence[Tool]] = None, max_steps: int = 4, **kwargs: Any):
        """Initialize the agent."""

        super().__init__(**kwargs)
        self.extra_tools = list(tools or [])
        self.max_steps = max(1, max_steps)

    def available_tools(self, ctx: PipelineContext) -> Dict[str, Tool]:
        """Return the tools for this run (built-ins + user tools)."""

        tools: Dict[str, Tool] = {}
        if self.retriever is not None:
            tools["search_docs"] = Tool(
                name="search_docs",
                description="Search the private document collection. Input: a focused search query.",
                func=lambda q: self.retriever.retrieve(q, filters=ctx.filters),  # type: ignore[union-attr]
            )
        if self.web_retriever is not None:
            tools["search_web"] = Tool(
                name="search_web",
                description="Search the public internet for recent or external information. Input: a web query.",
                func=self.web_retriever.retrieve,
            )
        if self.graph_retriever is not None:
            tools["graph_search"] = Tool(
                name="graph_search",
                description="Query the knowledge graph for entities, their relationships and corpus-wide themes. "
                "Input: a question or entity names.",
                func=self.graph_retriever.retrieve,
            )
        for tool in self.extra_tools:
            tools[tool.name] = tool
        return tools

    def retrieve(self, ctx: PipelineContext) -> List[Document]:
        """Run the agent loop and return the gathered evidence."""

        if self.llm_client is None:
            raise ValueError("AgenticRAG requires llm_client")
        tools = self.available_tools(ctx)
        if not tools:
            raise ValueError("AgenticRAG has no tools: configure a retriever, web search, graph, or tools=[...]")

        steps: List[Dict[str, Any]] = []
        evidence: List[List[Document]] = []
        for step_number in range(1, self.max_steps + 1):
            action = self._next_action(ctx.search_query, tools, steps)
            if action is None:
                ctx.trace.append(f"step_{step_number}:unparseable_action")
                break
            name, tool_input = action
            if name == FINISH:
                ctx.trace.append(f"step_{step_number}:finish")
                break
            tool = tools.get(name)
            if tool is None:
                steps.append({"action": name, "input": tool_input, "observation": f"Unknown tool '{name}'."})
                ctx.trace.append(f"step_{step_number}:unknown_tool:{name}")
                continue
            with ctx.tracer.stage("tool", tool=name, step=step_number) as meta:
                try:
                    docs = tool.run(tool_input or ctx.search_query)
                    observation = self._summarize(docs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tool %s failed: %s", name, exc)
                    docs, observation = [], f"Tool error: {exc}"
                meta["docs"] = len(docs)
            evidence.append(docs)
            steps.append({"action": name, "input": tool_input, "observation": observation})
            ctx.trace.append(f"step_{step_number}:{name}({(tool_input or '')[:50]})->{len(docs)}_docs")

        if not any(evidence) and "search_docs" in tools:
            ctx.trace.append("fallback:search_docs")
            evidence.append(tools["search_docs"].run(ctx.search_query))
        ctx.metadata["agent_steps"] = steps
        ctx.trace.insert(0, f"plan_steps:{len(steps)}")
        # Each step targets a different part of the question: round-robin keeps every step's best evidence.
        return interleave([docs for docs in evidence if docs])

    def _next_action(self, question: str, tools: Dict[str, Tool], steps: List[Dict[str, Any]]):
        """Ask the LLM for the next action; returns (tool_name, input) or None."""

        tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools.values())
        history = "\n".join(
            f"Step {index}: {step['action']}({step['input']!r})\nObservation: {step['observation']}"
            for index, step in enumerate(steps, start=1)
        ) or "(no steps yet)"
        prompt = (
            "You are a research agent gathering evidence to answer a question.\n"
            f"Question: {question}\n\n"
            f"Available tools:\n{tool_lines}\n- {FINISH}: stop when the observations contain enough evidence.\n\n"
            f"Previous steps:\n{history}\n\n"
            "Choose the next action. Prefer specific queries; do not repeat a previous search.\n"
            'Reply with JSON only: {"thought": "<brief reasoning>", "action": "<tool name or finish>", "input": "<tool input>"}'
        )
        assert self.llm_client is not None
        data = self.llm_client.complete_json(prompt, default=None)
        if not isinstance(data, dict):
            return None
        name = str(data.get("action") or "").strip()
        if not name:
            return None
        return name, str(data.get("input") or "").strip()

    @staticmethod
    def _summarize(documents: List[Document], limit: int = 3) -> str:
        """Compact observation text for the agent scratchpad."""

        if not documents:
            return "No results."
        parts = [f"[{i}] {doc.content[:220].strip()}" for i, doc in enumerate(documents[:limit], start=1)]
        more = f" (+{len(documents) - limit} more)" if len(documents) > limit else ""
        return " ".join(parts) + more
