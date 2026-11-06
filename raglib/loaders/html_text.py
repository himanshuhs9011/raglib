"""HTML to structured plain text (shared by the web, HTML-file and EPUB loaders).

Keeps the document's structure so chunking can split on paragraphs: block elements end
with line breaks, headings become ``#`` lines, list items ``- `` lines and table rows
``a | b | c`` lines. Navigation, footers, sidebars, forms and scripts are dropped, and
when the page has a ``<main>`` or ``<article>`` element only that content is kept.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Tuple

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form", "iframe", "button", "select"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "table", "thead", "tbody", "ul", "ol", "dl",
    "dt", "dd", "blockquote", "figure", "figcaption", "address", "details", "summary", "body", "hr",
}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_CONTENT_ROOTS = {"main", "article"}


class _StructuredTextParser(HTMLParser):
    """Collect visible text with block structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._pre_depth = 0
        self._root_depth = 0
        self._all: List[str] = []
        self._root: List[str] = []
        self._row: List[str] = []
        self._cell: List[str] = []
        self._in_cell = False

    def _emit(self, text: str) -> None:
        if self._in_cell:
            self._cell.append(text)
            return
        self._all.append(text)
        if self._root_depth:
            self._root.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            if tag in {"br", "hr"} and not self._skip_depth:
                self._emit("\n")
            return
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _CONTENT_ROOTS:
            self._root_depth += 1
            self._emit("\n\n")
        elif tag == "pre":
            self._pre_depth += 1
            self._emit("\n\n")
        elif tag in _HEADINGS:
            self._emit("\n\n" + "#" * _HEADINGS[tag] + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell = []
        elif tag in _BLOCK_TAGS:
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in _CONTENT_ROOTS:
            self._emit("\n\n")
            self._root_depth = max(0, self._root_depth - 1)
        elif tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self._emit("\n\n")
        elif tag in {"td", "th"}:
            self._in_cell = False
            self._row.append(" ".join("".join(self._cell).split()))
        elif tag == "tr":
            if any(self._row):
                self._emit("\n" + " | ".join(self._row))
            self._row = []
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth or not data:
            return
        if self._pre_depth:
            self._emit(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        chunks = self._root if "".join(self._root).strip() else self._all
        return _normalize("".join(chunks))


def _normalize(text: str) -> str:
    """Trim spaces around line breaks and collapse runs of blank lines."""

    lines = [line.rstrip() for line in text.replace("\r", "").split("\n")]
    lines = [line.lstrip() if not line.startswith("    ") else line for line in lines]
    joined = "\n".join(lines)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def html_to_text(raw_html: str) -> Tuple[str, str]:
    """Return ``(title, text)`` for an HTML document."""

    parser = _StructuredTextParser()
    try:
        parser.feed(raw_html)
        parser.close()
        text = parser.text()
    except Exception:  # noqa: BLE001 - malformed markup: fall back to tag stripping
        text = ""
    if not text:
        without = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", " ", raw_html, flags=re.IGNORECASE)
        text = _normalize(html.unescape(re.sub(r"<[^>]+>", " ", without)))
    return " ".join(parser.title.split()), text
