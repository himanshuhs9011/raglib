"""Conversation memory: per-session, token-bounded turn history."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from raglib.core.text import count_tokens
from raglib.schemas import Document

logger = logging.getLogger(__name__)

DEFAULT_SESSION = "default"


@dataclass
class MemoryTurn:
    """Represents one conversational turn in memory."""

    query: str
    answer: str
    documents: List[Document] = field(default_factory=list)


class MemoryModule:
    """Bounded history for ONE conversation (turn- and token-capped)."""

    def __init__(self, max_turns: int = 10, max_tokens: int = 1200):
        """Initialize bounded memory storage."""

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._turns: Deque[MemoryTurn] = deque(maxlen=max_turns)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of stored turns."""

        return len(self._turns)

    @property
    def turns(self) -> List[MemoryTurn]:
        """Return stored turns, oldest first."""

        return list(self._turns)

    def add(self, query: str, answer: str, documents: Optional[List[Document]] = None) -> None:
        """Add a turn to memory."""

        with self._lock:
            self._turns.append(MemoryTurn(query=query, answer=answer, documents=list(documents or [])))

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Return the most recent turns that fit the token budget, oldest first."""

        budget = max_tokens if max_tokens is not None else self.max_tokens
        blocks: List[str] = []
        used = 0
        for turn in reversed(self._turns):
            block = f"User: {turn.query}\nAssistant: {turn.answer}"
            tokens = count_tokens(block)
            if blocks and used + tokens > budget:
                break
            blocks.append(block)
            used += tokens
        return "\n\n".join(reversed(blocks))

    def clear(self) -> None:
        """Clear all memory turns."""

        with self._lock:
            self._turns.clear()


class ConversationStore:
    """Maps ``session_id`` -> MemoryModule, so concurrent users never share history.

    Sessions are LRU-evicted beyond ``max_sessions``. Subclass and override
    ``get``/``clear`` to back this with Redis or a database.
    """

    def __init__(self, max_turns: int = 10, max_tokens: int = 1200, max_sessions: int = 10_000):
        """Initialize the session map."""

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, MemoryModule]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str] = None) -> MemoryModule:
        """Return (creating if needed) the memory for a session."""

        key = session_id or DEFAULT_SESSION
        with self._lock:
            memory = self._sessions.get(key)
            if memory is None:
                memory = MemoryModule(max_turns=self.max_turns, max_tokens=self.max_tokens)
                self._sessions[key] = memory
                if len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(key)
            return memory

    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear one session, or every session when ``session_id`` is None."""

        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)

    def sessions(self) -> List[str]:
        """Return known session ids."""

        return list(self._sessions.keys())
