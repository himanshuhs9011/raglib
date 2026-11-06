"""Conversation memory: per-session, token-bounded turn history.

``ConversationStore`` keeps sessions in process memory. ``SQLiteConversationStore`` persists
them to a SQLite file (standard library, no extra dependency), so history survives restarts
and is shared by every worker process on the machine. For Redis or a SQL server, subclass
``ConversationStore`` and override ``get``/``clear``/``sessions`` (``get`` returns any
``MemoryModule``; override ``add``/``turns``/``clear`` on it).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
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
    """Bounded history for ONE conversation (turn- and token-capped).

    ``store_documents=False`` (default) keeps only the question and answer of each turn;
    holding every turn's retrieved documents costs memory for thousands of sessions and
    is not used to build the conversation context.
    """

    def __init__(self, max_turns: int = 10, max_tokens: int = 1200, store_documents: bool = False):
        """Initialize bounded memory storage."""

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.store_documents = store_documents
        self._turns: Deque[MemoryTurn] = deque(maxlen=max_turns)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of stored turns."""

        return len(self.turns)

    @property
    def turns(self) -> List[MemoryTurn]:
        """Return stored turns, oldest first."""

        with self._lock:
            return list(self._turns)

    def add(self, query: str, answer: str, documents: Optional[List[Document]] = None) -> None:
        """Add a turn to memory."""

        kept = list(documents or []) if self.store_documents else []
        with self._lock:
            self._turns.append(MemoryTurn(query=query, answer=answer, documents=kept))

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Return the most recent turns that fit the token budget, oldest first."""

        budget = max_tokens if max_tokens is not None else self.max_tokens
        blocks: List[str] = []
        used = 0
        for turn in reversed(self.turns):  # snapshot: safe while other requests add turns
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
    ``get``/``clear``/``sessions`` to back this with Redis or a database
    (see ``SQLiteConversationStore``).
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 1200,
        max_sessions: int = 10_000,
        store_documents: bool = False,
    ):
        """Initialize the session map."""

        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_sessions = max_sessions
        self.store_documents = store_documents
        self._sessions: "OrderedDict[str, MemoryModule]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str] = None) -> MemoryModule:
        """Return (creating if needed) the memory for a session."""

        key = session_id or DEFAULT_SESSION
        with self._lock:
            memory = self._sessions.get(key)
            if memory is None:
                memory = MemoryModule(
                    max_turns=self.max_turns, max_tokens=self.max_tokens, store_documents=self.store_documents
                )
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

        with self._lock:
            return list(self._sessions.keys())


class _SQLiteMemory(MemoryModule):
    """One session's history stored in SQLite."""

    def __init__(self, store: "SQLiteConversationStore", session_id: str):
        super().__init__(max_turns=store.max_turns, max_tokens=store.max_tokens, store_documents=store.store_documents)
        self._store = store
        self._session_id = session_id

    @property
    def turns(self) -> List[MemoryTurn]:
        """Return the last ``max_turns`` turns, oldest first."""

        rows = self._store._fetch(self._session_id, self.max_turns)
        turns = []
        for query, answer, documents in rows:
            docs = [Document(**item) for item in json.loads(documents)] if documents else []
            turns.append(MemoryTurn(query=query, answer=answer, documents=docs))
        return turns

    def add(self, query: str, answer: str, documents: Optional[List[Document]] = None) -> None:
        """Append a turn and trim the session to ``max_turns``."""

        payload = json.dumps([asdict(doc) for doc in documents or []], default=str) if self.store_documents else None
        self._store._append(self._session_id, query, answer, payload, self.max_turns)

    def clear(self) -> None:
        """Delete this session's turns."""

        self._store.clear(self._session_id)


class SQLiteConversationStore(ConversationStore):
    """Conversation history persisted in a SQLite file.

    History survives restarts and is shared by every process that opens the same file
    (e.g. several uvicorn workers). Sessions idle longer than ``ttl_seconds`` are deleted
    lazily. Use ``":memory:"`` for a throwaway database (tests).
    """

    def __init__(
        self,
        path: str = "raglib_conversations.db",
        max_turns: int = 10,
        max_tokens: int = 1200,
        store_documents: bool = False,
        ttl_seconds: Optional[float] = None,
    ):
        """Open (or create) the database."""

        import sqlite3  # imported lazily: browser runtimes (Pyodide) ship without it

        super().__init__(max_turns=max_turns, max_tokens=max_tokens, store_documents=store_documents)
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._db_lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        with self._db_lock, self._conn:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS raglib_turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, created REAL NOT NULL, "
                "query TEXT NOT NULL, answer TEXT NOT NULL, documents TEXT)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS raglib_turns_session ON raglib_turns(session, id)")

    def get(self, session_id: Optional[str] = None) -> MemoryModule:
        """Return the memory view for a session."""

        self._expire()
        return _SQLiteMemory(self, session_id or DEFAULT_SESSION)

    def clear(self, session_id: Optional[str] = None) -> None:
        """Delete one session, or every session when ``session_id`` is None."""

        with self._db_lock, self._conn:
            if session_id is None:
                self._conn.execute("DELETE FROM raglib_turns")
            else:
                self._conn.execute("DELETE FROM raglib_turns WHERE session = ?", (session_id,))

    def sessions(self) -> List[str]:
        """Return session ids that have history."""

        with self._db_lock:
            return [row[0] for row in self._conn.execute("SELECT DISTINCT session FROM raglib_turns ORDER BY session")]

    def close(self) -> None:
        """Close the database connection."""

        with self._db_lock:
            self._conn.close()

    def _fetch(self, session_id: str, limit: int):
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT query, answer, documents FROM raglib_turns WHERE session = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return list(reversed(rows))

    def _append(self, session_id: str, query: str, answer: str, documents: Optional[str], max_turns: int) -> None:
        with self._db_lock, self._conn:
            self._conn.execute(
                "INSERT INTO raglib_turns (session, created, query, answer, documents) VALUES (?, ?, ?, ?, ?)",
                (session_id, time.time(), query, answer, documents),
            )
            self._conn.execute(
                "DELETE FROM raglib_turns WHERE session = ? AND id NOT IN "
                "(SELECT id FROM raglib_turns WHERE session = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, max_turns),
            )

    def _expire(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = time.time() - self.ttl_seconds
        with self._db_lock, self._conn:
            self._conn.execute(
                "DELETE FROM raglib_turns WHERE session IN "
                "(SELECT session FROM raglib_turns GROUP BY session HAVING MAX(created) < ?)",
                (cutoff,),
            )
