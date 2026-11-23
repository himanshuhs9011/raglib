"""Parent sections for small-to-big retrieval (``RAG(parent_retrieval="heading" | "fixed")``).

Small child chunks are what gets embedded and searched (precise matches); the larger parent
section each child belongs to is what the LLM reads (whole context). ``build_sections`` cuts a
loaded document into those parent sections.

Trade-off (accuracy vs tokens): answers see whole sections instead of fragments, but each source
sent to the LLM is larger - up to ``parent_chunk_size`` tokens - so prompts grow and fewer
sources fit in ``max_context_tokens``. Use small children (e.g. ``chunk_size=200``) with it.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from raglib.core.text import count_units
from raglib.schemas import Document

PARENT_MODES = ("heading", "fixed")

# Markdown ATX headings ("# Title") and numbered headings ("2.1 Scope") on their own line.
_HEADING_RE = re.compile(r"(?m)^(?:#{1,6}[ \t]+\S.*|\d+(?:\.\d+)*\.?[ \t]+[A-Z][^\n]{0,80})$")


def _windows(text: str, size: int) -> List[str]:
    """Cut text into pieces of at most ``size`` units on paragraph, then line, then word borders."""

    from raglib.loaders.chunk_splitter import ChunkSplitter

    splitter = ChunkSplitter(chunk_size=size, overlap=0, min_tokens=1, strategy="recursive")
    return splitter._recursive_split(text, splitter._separators, size)  # noqa: SLF001 - shared splitting rules


def split_sections(text: str, mode: str, parent_chunk_size: int) -> List[Tuple[str, str]]:
    """``(title, text)`` sections: at headings (``"heading"``) or in fixed windows (``"fixed"``).

    Heading sections longer than ``parent_chunk_size`` are cut further; a document without
    headings falls back to fixed windows.
    """

    if mode not in PARENT_MODES:
        raise ValueError(f"parent_retrieval must be one of {list(PARENT_MODES)}")
    text = text.strip()
    if not text:
        return []
    pieces: List[Tuple[str, str]] = []
    if mode == "heading":
        starts = [match.start() for match in _HEADING_RE.finditer(text)]
        if starts:
            if starts[0] > 0:
                starts.insert(0, 0)
            bounds = starts + [len(text)]
            for start, end in zip(bounds, bounds[1:]):
                block = text[start:end].strip()
                if block:
                    first_line = block.splitlines()[0].lstrip("# ").strip()
                    pieces.append((first_line[:120], block))
    if not pieces:
        pieces = [("", text)]
    sections: List[Tuple[str, str]] = []
    for title, block in pieces:
        if count_units(block) <= parent_chunk_size:
            sections.append((title, block))
        else:
            sections.extend((title, window) for window in _windows(block, parent_chunk_size))
    return sections


def build_sections(document: Document, mode: str, parent_chunk_size: int) -> List[Document]:
    """Parent-section documents of ``document``; ids are ``{doc id}#s{n}_{content hash}``."""

    sections: List[Document] = []
    for index, (title, text) in enumerate(split_sections(document.content, mode, parent_chunk_size), start=1):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        metadata = {**document.metadata, "parent_id": document.id, "section_index": index}
        if title:
            metadata["section_title"] = title
        sections.append(
            Document(id=f"{document.id}#s{index}_{digest}", content=text, metadata=metadata, source=document.source)
        )
    return sections
