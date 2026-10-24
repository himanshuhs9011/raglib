"""Interactive terminal chat session for raglib RAG facade."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List

from raglib.schemas import GenerationResult

if TYPE_CHECKING:
    from raglib.facade import RAG

logger = logging.getLogger(__name__)


class InteractiveSession:
    """Terminal-based interactive Q&A session over a RAG instance."""

    BANNER = (
        "+------------------------------------------------------+\n"
        "|              raglib interactive session             |\n"
        "|  Type your question and press Enter.                |\n"
        "|  Commands: help | history | reset | clear | exit     |\n"
        "+------------------------------------------------------+"
    )

    HELP_TEXT = (
        "Available commands:\n"
        "  help      - show this message\n"
        "  history   - show all questions asked in this session\n"
        "  reset     - forget the conversation so far\n"
        "  clear     - clear the terminal screen\n"
        "  exit / quit / q / bye / stop - end the session\n"
    )

    def __init__(self, rag: "RAG", session_id: str = "interactive"):
        """Store the RAG facade instance and initialize session state."""

        self._rag = rag
        self._session_id = session_id
        self._history: List[Dict[str, str]] = []
        self._turn = 0

    def start(self) -> None:
        """Print banner and run the input loop until the user exits."""

        print(self.BANNER)
        stats = self._rag.stats()
        print(f"  Chunks indexed   : {stats['chunks']}")
        print(f"  LLM              : {stats['llm']}")
        print(f"  RAG strategy     : {stats['rag_type']}")
        print()

        while True:
            try:
                raw = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession ended.")
                break

            if not raw:
                continue

            normalized = raw.lower()
            if normalized in {"exit", "quit", "q", "bye", "stop"}:
                print("Session ended.")
                break
            if normalized == "help":
                print(self.HELP_TEXT)
                continue
            if normalized == "history":
                self._print_history()
                continue
            if normalized == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if normalized == "reset":
                self._rag.reset_memory(self._session_id)
                self._history.clear()
                print("  Conversation memory cleared.\n")
                continue

            self._turn += 1
            print("\nThinking...\n")
            try:
                result = self._rag.query(raw, session_id=self._session_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("InteractiveSession query failed")
                print(f"[Error] {exc}\n")
                continue

            self._print_result(result)
            self._history.append({"q": raw, "a": result.answer})

    def _print_history(self) -> None:
        """Print all Q/A turns accumulated during this session."""

        if not self._history:
            print("  No questions asked yet in this session.\n")
            return

        for index, turn in enumerate(self._history, start=1):
            answer_preview = turn["a"][:120]
            suffix = "..." if len(turn["a"]) > 120 else ""
            print(f"  [{index}] Q: {turn['q']}")
            print(f"       A: {answer_preview}{suffix}")
        print()

    def _print_result(self, result: GenerationResult) -> None:
        """Print answer, compact source summary, and optional reasoning trace."""

        print(f"Assistant: {result.answer}\n")

        if result.citations:
            for citation in result.citations:
                print(f"  [{citation.index}] {self._truncate_source_id(citation.document_id)}")
        elif result.sources:
            summary = ", ".join(self._truncate_source_id(document.id) for document in result.sources[:3])
            print(f"  Sources: {summary}")

        if result.reasoning_trace:
            print(f"  Trace  : {' -> '.join(result.reasoning_trace)}")

        print()

    def _truncate_source_id(self, value: str, max_length: int = 48) -> str:
        """Return a compact source id string for terminal display."""

        if len(value) <= max_length:
            return value
        return f"{value[:max_length - 3]}..."
