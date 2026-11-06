"""Loaders for HTML files, tabular data (CSV/TSV, XLSX), JSON/JSONL and EPUB books.

Tabular rows are rendered one per line as ``column: value; column: value`` so every line
is self-describing: after chunking, a chunk that holds row 4,812 still says which value
is the city and which is the profession. For questions that aggregate over all rows
("how many engineers are in Tokyo?") use ``raglib.components.sql_tool.SQLTool`` instead
of retrieval, since top-k retrieval only ever sees a few chunks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import posixpath
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence
from xml.etree import ElementTree

from raglib.loaders.html_text import html_to_text
from raglib.schemas import Document

logger = logging.getLogger(__name__)


def _file_id(prefix: str, path: str) -> str:
    """Stable document id derived from the absolute file path."""

    return f"{prefix}-{hashlib.sha256(os.path.abspath(path).encode('utf-8')).hexdigest()[:16]}"


def _read_text(path: str) -> str:
    """Read a text file as UTF-8 (with BOM handling), falling back to Latin-1."""

    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as handle:
            return handle.read()


def format_row(header: Sequence[str], row: Sequence[Any]) -> str:
    """Render one table row as ``column: value; ...`` (empty cells are skipped)."""

    parts = []
    for index, value in enumerate(row):
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        name = str(header[index]).strip() if index < len(header) and str(header[index]).strip() else f"column_{index + 1}"
        parts.append(f"{name}: {text}")
    return "; ".join(parts)


def _rows_to_text(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Dict[str, Any]:
    """Render rows and count them."""

    lines = [line for line in (format_row(header, row) for row in rows) if line]
    return {"text": "\n".join(lines), "rows": len(lines)}


class HTMLFileLoader:
    """Load a local .html/.htm/.xhtml file as structured text."""

    def load(self, path: str) -> List[Document]:
        """Return one Document (title in metadata)."""

        title, text = html_to_text(_read_text(path))
        if not text:
            logger.warning("HTMLFileLoader found no text in file=%s", path)
            return []
        metadata: Dict[str, Any] = {"file": path, "loader": "html"}
        if title:
            metadata["title"] = title
        return [Document(id=_file_id("html", path), content=text, metadata=metadata, source="html")]


class CSVLoader:
    """Load .csv/.tsv files: the first row is the header, each data row becomes one line."""

    def __init__(self, delimiter: Optional[str] = None, has_header: bool = True):
        """``delimiter=None`` sniffs it (tab for .tsv)."""

        self.delimiter = delimiter
        self.has_header = has_header

    def load(self, path: str) -> List[Document]:
        """Return one Document holding every row."""

        raw = _read_text(path)
        if not raw.strip():
            return []
        delimiter = self.delimiter or ("\t" if path.lower().endswith(".tsv") else self._sniff(raw))
        rows = list(csv.reader(raw.splitlines(), delimiter=delimiter))
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return []
        header = rows[0] if self.has_header else [f"column_{i + 1}" for i in range(len(rows[0]))]
        body = rows[1:] if self.has_header else rows
        rendered = _rows_to_text(header, body)
        metadata = {"file": path, "loader": "csv", "columns": [str(name) for name in header], "rows": rendered["rows"]}
        return [Document(id=_file_id("csv", path), content=rendered["text"], metadata=metadata, source="csv")]

    @staticmethod
    def _sniff(raw: str) -> str:
        """Guess the delimiter from the first lines."""

        try:
            return csv.Sniffer().sniff("\n".join(raw.splitlines()[:20]), delimiters=",;\t|").delimiter
        except csv.Error:
            return ","


class JSONLoader:
    """Load .json, .jsonl and .ndjson files.

    A JSON array of objects (or a JSONL file) becomes one line per record; any other JSON
    value is pretty-printed.
    """

    def load(self, path: str) -> List[Document]:
        """Return one Document."""

        raw = _read_text(path).strip()
        if not raw:
            return []
        records: Optional[List[Any]] = None
        if path.lower().endswith((".jsonl", ".ndjson")):
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
            if isinstance(data, list):
                records = data
            else:
                text = json.dumps(data, ensure_ascii=False, indent=2)
                return [Document(id=_file_id("json", path), content=text, metadata={"file": path, "loader": "json"}, source="json")]
        lines = []
        for record in records:
            if isinstance(record, dict):
                lines.append(format_row(list(record.keys()), [self._scalar(value) for value in record.values()]))
            else:
                lines.append(self._scalar(record))
        text = "\n".join(line for line in lines if line)
        metadata = {"file": path, "loader": "json", "rows": len(records)}
        return [Document(id=_file_id("json", path), content=text, metadata=metadata, source="json")] if text else []

    @staticmethod
    def _scalar(value: Any) -> str:
        """Render nested values compactly."""

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return "" if value is None else str(value)


class XLSXLoader:
    """Load .xlsx workbooks (``pip install raglib-py[xlsx]``): one Document per non-empty sheet."""

    def load(self, path: str) -> List[Document]:
        """Return one Document per sheet; the first non-empty row is the header."""

        try:
            from openpyxl import load_workbook  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("XLSX loading requires openpyxl. Install with: pip install raglib-py[xlsx]") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        documents: List[Document] = []
        try:
            for index, sheet in enumerate(workbook.worksheets, start=1):
                rows = [row for row in sheet.iter_rows(values_only=True) if any(cell not in (None, "") for cell in row)]
                if not rows:
                    continue
                header = ["" if cell is None else str(cell) for cell in rows[0]]
                rendered = _rows_to_text(header, rows[1:])
                if not rendered["text"]:
                    continue
                documents.append(
                    Document(
                        id=f"{_file_id('xlsx', path)}-sheet-{index}",
                        content=rendered["text"],
                        metadata={
                            "file": path,
                            "loader": "xlsx",
                            "sheet": sheet.title,
                            "columns": header,
                            "rows": rendered["rows"],
                        },
                        source="xlsx",
                    )
                )
        finally:
            workbook.close()
        return documents


_OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "container": "urn:oasis:names:tc:opendocument:xmlns:container"}


class EPUBLoader:
    """Load .epub books (standard library only): one Document per chapter, in reading order."""

    def load(self, path: str) -> List[Document]:
        """Return one Document per spine item that contains text."""

        documents: List[Document] = []
        base_id = _file_id("epub", path)
        with zipfile.ZipFile(path) as book:
            for number, (name, raw) in enumerate(self._spine(book), start=1):
                title, text = html_to_text(raw)
                if not text:
                    continue
                metadata: Dict[str, Any] = {"file": path, "loader": "epub", "chapter": number, "item": name}
                if title:
                    metadata["title"] = title
                documents.append(Document(id=f"{base_id}-ch-{number}", content=text, metadata=metadata, source="epub"))
        if not documents:
            logger.warning("EPUBLoader found no text in file=%s", path)
        return documents

    @staticmethod
    def _spine(book: zipfile.ZipFile) -> List[Any]:
        """Return (name, xhtml) pairs in spine order (falls back to every HTML file)."""

        try:
            container = ElementTree.fromstring(book.read("META-INF/container.xml"))
            rootfile = container.find(".//container:rootfile", _OPF_NS)
            opf_path = rootfile.get("full-path") if rootfile is not None else None
            if not opf_path:
                raise KeyError("no rootfile")
            opf = ElementTree.fromstring(book.read(opf_path))
            base = posixpath.dirname(opf_path)
            manifest = {
                item.get("id"): item.get("href")
                for item in opf.findall(".//opf:manifest/opf:item", _OPF_NS)
            }
            names = []
            for itemref in opf.findall(".//opf:spine/opf:itemref", _OPF_NS):
                href = manifest.get(itemref.get("idref"))
                if href:
                    names.append(posixpath.normpath(posixpath.join(base, href.split("#", 1)[0])))
        except (KeyError, ElementTree.ParseError):
            names = sorted(name for name in book.namelist() if re.search(r"\.x?html?$", name, re.IGNORECASE))
        pairs = []
        for name in names:
            try:
                pairs.append((name, book.read(name).decode("utf-8", errors="replace")))
            except KeyError:
                logger.debug("EPUB spine item missing: %s", name)
        return pairs
