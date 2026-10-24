"""Tool abstraction for agentic RAG."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Union

from raglib.schemas import Document

logger = logging.getLogger(__name__)

ToolOutput = Union[str, Document, List[Document], List[str], None]


@dataclass
class Tool:
    """A callable the agent can invoke with a single string input.

    ``func`` may return text, a Document, or a list of either; outputs are normalized
    to Documents so they can be cited like retrieved chunks.
    """

    name: str
    description: str
    func: Callable[[str], ToolOutput]

    def run(self, tool_input: str) -> List[Document]:
        """Invoke the tool and normalize its output to Documents."""

        output: Any = self.func(tool_input)
        return normalize_tool_output(self.name, tool_input, output)


def normalize_tool_output(tool_name: str, tool_input: str, output: Any) -> List[Document]:
    """Convert arbitrary tool output into a list of Documents."""

    if output is None:
        return []
    items = output if isinstance(output, list) else [output]
    documents: List[Document] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, Document):
            documents.append(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        digest = hashlib.sha256(f"{tool_name}:{tool_input}:{text}".encode("utf-8")).hexdigest()[:12]
        documents.append(
            Document(
                id=f"tool-{tool_name}-{digest}",
                content=text,
                metadata={"tool": tool_name, "tool_input": tool_input, "title": f"{tool_name} result {index}"},
                source=f"tool:{tool_name}",
            )
        )
    return documents
