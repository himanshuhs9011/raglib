"""SQL over tabular files, for questions that aggregate across every row.

Retrieval answers from the top-k chunks, so "how many engineers are based in Tokyo?" over a
5,000-row table can never be counted correctly: the model only sees a few chunks. ``SQLTool``
loads CSV/TSV/XLSX/JSON tables into an in-memory SQLite database and lets the LLM write one
``SELECT`` that is executed exactly::

    from raglib import RAG, SQLTool
    rag = RAG("people.csv", llm="openai", rag_type="agentic", tools=[SQLTool.from_files("people.csv")])
    rag.query("How many engineers are based in Tokyo?")

Safety: the database is a private in-memory copy, and an SQLite authorizer allows only
reading (``SELECT``, functions, CTEs); writes, ``ATTACH``, ``PRAGMA`` and every other
statement are refused. Queries are aborted after ``timeout_seconds`` and results are capped
at ``max_rows``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from raglib.components.tools import Tool
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

# SQLite authorizer action codes allowed: SELECT (21), READ (20), FUNCTION (31), RECURSIVE (33).
_READ_ACTIONS = {21, 20, 31, 33}
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _identifier(name: str, fallback: str) -> str:
    """Make a safe, lowercase SQL identifier."""

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip()).strip("_").lower()
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _coerce(value: Any) -> Any:
    """Convert numeric-looking strings to numbers so SUM/AVG/comparisons work."""

    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


class SQLTool:
    """Let the agent answer aggregate questions with read-only SQL over loaded tables."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        name: str = "query_tables",
        max_rows: int = 50,
        timeout_seconds: float = 5.0,
        sample_rows: int = 3,
    ):
        """Create an empty database; add tables with ``add_csv`` / ``add_xlsx`` / ``add_rows``.

        ``llm_client`` writes SQL from natural-language questions; when the tool is used by
        ``AgenticRAG`` without one, the agent's own LLM is used.
        """

        import sqlite3  # imported lazily: browser runtimes (Pyodide) ship without it

        self._sqlite = sqlite3
        self.llm_client = llm_client
        self.name = name
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.sample_rows = sample_rows
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self._tables: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_files(cls, paths: Any, llm_client: Optional[BaseLLMClient] = None, **kwargs: Any) -> "SQLTool":
        """Build a tool from .csv/.tsv/.xlsx/.json/.jsonl files (a path or a list of paths)."""

        tool = cls(llm_client=llm_client, **kwargs)
        for path in [paths] if isinstance(paths, (str, os.PathLike)) else list(paths):
            tool.add_file(os.fspath(path))
        return tool

    def add_file(self, path: str, table: Optional[str] = None) -> List[str]:
        """Load one file by extension; returns the table name(s) created."""

        lower = path.lower()
        if lower.endswith((".csv", ".tsv")):
            return [self.add_csv(path, table)]
        if lower.endswith(".xlsx"):
            return self.add_xlsx(path, table)
        if lower.endswith((".json", ".jsonl", ".ndjson")):
            return [self.add_json(path, table)]
        raise ValueError(f"SQLTool supports .csv, .tsv, .xlsx, .json and .jsonl files, got {path}")

    def add_csv(self, path: str, table: Optional[str] = None, delimiter: Optional[str] = None) -> str:
        """Load a CSV/TSV file (first row = header)."""

        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            raw = handle.read()
        if delimiter is None:
            if path.lower().endswith(".tsv"):
                delimiter = "\t"
            else:
                try:
                    delimiter = csv.Sniffer().sniff("\n".join(raw.splitlines()[:20]), delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
        rows = [row for row in csv.reader(raw.splitlines(), delimiter=delimiter) if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError(f"{path} has no rows")
        return self.add_rows(table or os.path.splitext(os.path.basename(path))[0], rows[0], rows[1:])

    def add_xlsx(self, path: str, table: Optional[str] = None) -> List[str]:
        """Load every non-empty sheet of a workbook (``pip install raglib-py[xlsx]``)."""

        try:
            from openpyxl import load_workbook  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("XLSX support requires openpyxl: pip install raglib-py[xlsx]") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        names = []
        try:
            base = table or os.path.splitext(os.path.basename(path))[0]
            sheets = [sheet for sheet in workbook.worksheets]
            for sheet in sheets:
                rows = [row for row in sheet.iter_rows(values_only=True) if any(cell not in (None, "") for cell in row)]
                if len(rows) < 2:
                    continue
                name = base if len(sheets) == 1 else f"{base}_{sheet.title}"
                names.append(self.add_rows(name, [str(cell or "") for cell in rows[0]], rows[1:]))
        finally:
            workbook.close()
        return names

    def add_json(self, path: str, table: Optional[str] = None) -> str:
        """Load a JSON array of objects, or a JSONL file of objects."""

        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = handle.read()
        if path.lower().endswith((".jsonl", ".ndjson")):
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            records = json.loads(raw)
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError(f"{path} must contain a list of JSON objects")
        header: List[str] = []
        for record in records:
            for key in record:
                if key not in header:
                    header.append(key)
        rows = [[json.dumps(r.get(k)) if isinstance(r.get(k), (dict, list)) else r.get(k) for k in header] for r in records]
        return self.add_rows(table or os.path.splitext(os.path.basename(path))[0], header, rows)

    def add_rows(self, table: str, header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
        """Create table ``table`` from a header and rows; returns the (sanitized) table name."""

        name = _identifier(table, "table")
        base, suffix = name, 2
        while name in self._tables:
            name, suffix = f"{base}_{suffix}", suffix + 1
        columns: List[str] = []
        for index, column in enumerate(header):
            column_name = _identifier(column, f"column_{index + 1}")
            while column_name in columns:
                column_name += "_"
            columns.append(column_name)
        data = [[_coerce(row[i]) if i < len(row) else None for i in range(len(columns))] for row in rows]
        types = []
        for index in range(len(columns)):
            values = [row[index] for row in data if row[index] is not None]
            if values and all(isinstance(value, int) for value in values):
                types.append("INTEGER")
            elif values and all(isinstance(value, (int, float)) for value in values):
                types.append("REAL")
            else:
                types.append("TEXT")
        column_sql = ", ".join(f'"{column}" {kind}' for column, kind in zip(columns, types))
        with self._lock:
            self._conn.execute(f'CREATE TABLE "{name}" ({column_sql})')
            placeholders = ", ".join("?" for _ in columns)
            self._conn.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', data)
            self._conn.commit()
        self._tables[name] = columns
        logger.info("SQLTool loaded table %s (%d rows, %d columns)", name, len(data), len(columns))
        return name

    # ------------------------------------------------------------------ querying
    @property
    def tables(self) -> Dict[str, List[str]]:
        """Table name -> column names."""

        return dict(self._tables)

    def schema(self) -> str:
        """Describe every table: columns with types, row count and a few sample rows."""

        parts = []
        with self._lock:
            for name in self._tables:
                ddl = self._conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,)).fetchone()[0]
                count = self._conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                samples = self._conn.execute(f'SELECT * FROM "{name}" LIMIT ?', (self.sample_rows,)).fetchall()
                lines = [f"{ddl};  -- {count} rows"]
                lines.extend(f"-- sample: {sample}" for sample in samples)
                parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def run_sql(self, sql: str) -> Dict[str, Any]:
        """Execute one read-only statement; returns ``{"columns", "rows", "truncated"}``."""

        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("Empty SQL")
        if ";" in statement:
            raise ValueError("Only one SQL statement is allowed")
        deadline = time.monotonic() + self.timeout_seconds

        sqlite3 = self._sqlite

        def authorize(action: int, *_: Any) -> int:
            return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY

        with self._lock:
            self._conn.set_authorizer(authorize)
            self._conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            try:
                cursor = self._conn.execute(statement)
                rows = cursor.fetchmany(self.max_rows + 1)
                columns = [description[0] for description in cursor.description or []]
            except sqlite3.DatabaseError as exc:
                if "interrupted" in str(exc).lower():
                    raise TimeoutError(f"SQL query exceeded {self.timeout_seconds}s") from exc
                raise
            finally:
                self._conn.set_authorizer(None)
                self._conn.set_progress_handler(None, 0)
        return {"columns": columns, "rows": rows[: self.max_rows], "truncated": len(rows) > self.max_rows}

    def write_sql(self, question: str, error: Optional[str] = None, previous: Optional[str] = None) -> str:
        """Ask the LLM for one SQLite SELECT answering ``question``."""

        if self.llm_client is None:
            raise ValueError("SQLTool needs llm_client to turn questions into SQL (or pass SQL directly)")
        repair = f"\nYour previous query:\n{previous}\nfailed with: {error}\nFix it.\n" if error else ""
        prompt = (
            "You write one SQLite SELECT query that answers the question from these tables.\n"
            f"{self.schema()}\n\n"
            f"Question: {question}\n{repair}"
            "Rules: a single read-only SELECT (CTEs allowed); use only the tables and columns above; "
            "match text case-insensitively (LOWER(col) = LOWER('value')) unless case matters.\n"
            "Return only the SQL."
        )
        raw = self.llm_client.complete(prompt=prompt, system="You are an expert SQLite analyst.")
        match = _FENCE_RE.search(raw or "")
        return (match.group(1) if match else raw or "").strip().rstrip(";")

    def answer(self, question_or_sql: str) -> List[Document]:
        """Run SQL (if given) or write SQL for a question, and return the result as a Document."""

        text = (question_or_sql or "").strip()
        is_sql = bool(re.match(r"(?is)^\s*(select|with)\b", text))
        sql = text if is_sql else self.write_sql(text)
        try:
            result = self.run_sql(sql)
        except (self._sqlite.DatabaseError, ValueError) as exc:
            if is_sql and self.llm_client is None:
                raise
            sql = self.write_sql(text, error=str(exc), previous=sql)
            result = self.run_sql(sql)
        return [self._to_document(question_or_sql, sql, result)]

    def _to_document(self, question: str, sql: str, result: Dict[str, Any]) -> Document:
        """Render a query result as a citable Document."""

        header = " | ".join(result["columns"])
        body = "\n".join(" | ".join("NULL" if value is None else str(value) for value in row) for row in result["rows"])
        note = f"\n(showing the first {self.max_rows} rows)" if result["truncated"] else ""
        content = f"SQL query over the loaded tables:\n{sql}\n\nResult:\n{header}\n{body or '(no rows)'}{note}"
        return Document(
            id="sql-" + hashlib.sha256(f"{sql}\x00{body}".encode("utf-8")).hexdigest()[:16],
            content=content,
            metadata={"tool": self.name, "sql": sql, "title": f"{self.name}: {question[:60]}", "rows": len(result["rows"])},
            source=f"tool:{self.name}",
        )

    def as_tool(self) -> Tool:
        """Return the agent ``Tool`` (``AgenticRAG`` / ``RAG(tools=[...])`` call this automatically)."""

        names = ", ".join(f"{table}({', '.join(columns)})" for table, columns in self._tables.items())
        return Tool(
            name=self.name,
            description=(
                "Run exact SQL over structured tables - use it to count, sum, average, rank or filter rows "
                f"(retrieval cannot count across all rows). Tables: {names}. "
                "Input: the question in plain language, or a SQLite SELECT query."
            ),
            func=self.answer,
        )
