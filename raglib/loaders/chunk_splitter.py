"""Chunk splitting utilities with fixed, recursive, LLM and embedding (semantic) strategies."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from raglib.core.text import has_no_space_script, resolve_token_counter, unit_spans
from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

SUPPORTED_CHUNKING_STRATEGIES = ("fixed", "recursive", "llm", "embedding")
SEMANTIC_BREAKPOINTS = ("percentile", "fixed")
EMBEDDING_FAILURE_POLICIES = ("fallback", "raise")
FALLBACK_STRATEGY_LABEL = "recursive_fallback"

# Words ending in a period that do not end a sentence ("Dr. Sharma", "e.g. tables"). Lower case.
DEFAULT_ABBREVIATIONS: Tuple[str, ...] = (
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.", "mt.", "e.g.", "i.e.", "etc.", "vs.", "inc.",
    "ltd.", "co.", "corp.", "no.", "nos.", "fig.", "figs.", "approx.", "dept.", "est.", "vol.", "pp.",
)

_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+\u2022]|\d{1,3}[.)])[ \t]+\S")

FallbackHook = Callable[[Document, Exception], None]


class EmbeddingChunkingError(RuntimeError):
    """Embedding-based chunking failed and ``on_embedding_failure="raise"`` was set."""


@dataclass
class _Piece:
    """One chunk's text and, when known, its character offsets in the document content."""

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
CHUNKING_STRATEGY_ALIASES = {
    "fixed": "fixed",
    "fixed_size": "fixed",
    "sliding": "fixed",
    "sentence": "fixed",
    "recursive": "recursive",
    "recursive_character": "recursive",
    "llm": "llm",
    "semantic": "llm",
    "llm_semantic": "llm",
    "embedding": "embedding",
    "embedding_semantic": "embedding",
    "semantic_embedding": "embedding",
}

# Sentence boundaries: whitespace after . ! ? (or the Devanagari danda / double danda), any line
# break, right after CJK full stops (no space follows them), and whitespace after Thai/CJK text.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?\u0964\u0965])[ \t]+"
    r"|\s*\n\s*"
    r"|(?<=[\u3002\uff01\uff1f])\s*"
    r"|(?<=[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u9fff])[ \t]+"
)

# Prefer coarser structure first, then finer units (LangChain-style recursive split).
DEFAULT_RECURSIVE_SEPARATORS: Sequence[str] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "。",
    "！",
    "？",
    "; ",
    ", ",
    " ",
    "",
)


class ChunkSplitter:
    """Split documents into overlapping chunks using a selectable strategy."""

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        min_tokens: int = 20,
        strategy: str = "fixed",
        llm_client: Optional[BaseLLMClient] = None,
        separators: Optional[Sequence[str]] = None,
        embedding: Optional[BaseEmbedding] = None,
        semantic_breakpoint: str = "percentile",
        semantic_threshold: Optional[float] = None,
        semantic_min_tokens: Optional[int] = None,
        token_counter: Any = "units",
        on_embedding_failure: str = "fallback",
        abbreviations: Optional[Iterable[str]] = None,
        on_fallback: Optional[FallbackHook] = None,
    ):
        """Configure chunk size, overlap, strategy, and optional LLM client / embedding model.

        Sizes (``chunk_size``, ``overlap``, ``min_tokens``, ``semantic_min_tokens``) are measured
        with ``token_counter``: ``"units"`` (default: whitespace words, one per CJK/Thai character -
        *not* model tokens; English is ~1.3 tokens per word), ``"tiktoken"`` (real model tokens),
        ``"auto"`` (tiktoken when installed, else units) or a callable ``text -> int``.

        ``strategy="embedding"`` (semantic chunking with embeddings): at every gap between two
        sentences, embed the two sentences before it and the two after it, and start a new chunk
        where their similarity drops - so chunks end where the topic changes. ``semantic_breakpoint``:

        * ``"percentile"`` (default): break where the distance to the next sentence is above the
          ``semantic_threshold``-th percentile of all distances in the document (default 95).
        * ``"fixed"``: break where the cosine similarity is below ``semantic_threshold`` (default 0.75).

        Chunks are kept between ``semantic_min_tokens`` (default ``chunk_size // 4``) and
        ``chunk_size``; no overlap is added (boundaries are topic changes). Pass a
        ``CachedEmbedding`` (RAG does) so re-ingesting unchanged text costs nothing.

        The embedding strategy keeps the original formatting: every chunk is a slice of the
        document (``metadata["start_char"]`` / ``["end_char"]``), so line breaks, tables and lists
        survive. A Markdown table, or a run of list items, is one unit that is only split (at row /
        item borders) when it is larger than ``chunk_size``. Sentences do not end after
        ``abbreviations`` (default ``DEFAULT_ABBREVIATIONS``: "Dr.", "e.g.", "Inc.", ...).

        ``on_embedding_failure``: ``"fallback"`` (default) splits the document with the recursive
        strategy instead, labels its chunks ``chunking_strategy="recursive_fallback"`` and calls
        ``on_fallback(document, error)``; ``"raise"`` raises ``EmbeddingChunkingError`` (and
        ``ValueError`` at construction when no embedding model is given).

        Trade-off (accuracy vs cost): ~15-25% better retrieval on structured documents in published
        tests, but every sentence is embedded - roughly 2x the corpus in embedding tokens (pairs of
        sentences around each gap) on top of the chunk embeddings, and slower ingestion. Use it for
        high-value documents (legal, medical, financial), not millions of blog pages.
        The ``"llm"`` strategy (alias ``"semantic"``) asks an LLM for the boundaries instead.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if min_tokens <= 0:
            raise ValueError("min_tokens must be greater than zero")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_tokens = min_tokens
        self._strategy = self._normalize_strategy(strategy)
        self._llm_client = llm_client
        self._separators = list(separators) if separators is not None else list(DEFAULT_RECURSIVE_SEPARATORS)
        if on_embedding_failure not in EMBEDDING_FAILURE_POLICIES:
            raise ValueError(f"on_embedding_failure must be one of {list(EMBEDDING_FAILURE_POLICIES)}")
        self._token_counter, self.token_counter_name = resolve_token_counter(token_counter)
        self._on_embedding_failure = on_embedding_failure
        self._abbreviations = frozenset(
            item.strip().lower() for item in (DEFAULT_ABBREVIATIONS if abbreviations is None else abbreviations) if item.strip()
        )
        self.on_fallback = on_fallback
        if semantic_breakpoint not in SEMANTIC_BREAKPOINTS:
            raise ValueError(f"semantic_breakpoint must be one of {list(SEMANTIC_BREAKPOINTS)}")
        if semantic_breakpoint == "percentile" and semantic_threshold is not None and not 0 < semantic_threshold < 100:
            raise ValueError("a percentile semantic_threshold must be between 0 and 100")
        if semantic_breakpoint == "fixed" and semantic_threshold is not None and not -1 <= semantic_threshold <= 1:
            raise ValueError("a fixed semantic_threshold is a cosine similarity between -1 and 1")
        self._embedding = embedding
        self._semantic_breakpoint = semantic_breakpoint
        self._semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else (95.0 if semantic_breakpoint == "percentile" else 0.75)
        )
        self._semantic_min_tokens = max(1, semantic_min_tokens if semantic_min_tokens is not None else chunk_size // 4)
        if self._strategy == "embedding" and self._embedding is None:
            if on_embedding_failure == "raise":
                raise ValueError("chunking_strategy='embedding' needs an embedding model (on_embedding_failure='raise')")
            logger.warning("chunking_strategy='embedding' without an embedding model; falling back to recursive")

        if self._strategy == "llm" and self._llm_client is None:
            logger.warning(
                "chunking_strategy='llm' without llm_client; semantic splits will fall back to recursive"
            )

    @property
    def strategy(self) -> str:
        """Return the normalized chunking strategy name."""

        return self._strategy

    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with preserved metadata and prev/next links.

        Chunk ids are ``{parent_id}_chunk_{n}_{content_hash}``: stable for unchanged text
        (idempotent re-ingestion) and different when the text changes (safe upserts).
        """

        chunks: List[Document] = []
        previous_last: Optional[Document] = None
        previous_document: Optional[Document] = None
        for document in documents:
            pieces, label = self._split_document(document)
            doc_chunks: List[Document] = []
            for index, piece in enumerate(pieces, start=1):
                chunk_text = piece.text
                digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:8]
                metadata = {
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                    "chunk_count": len(pieces),
                    "chunk_tokens": self._count_tokens(chunk_text),
                    "chunking_strategy": label,
                }
                if piece.start is not None:
                    # document.content[start_char:end_char] == chunk content (character offsets).
                    metadata["start_char"], metadata["end_char"] = piece.start, piece.end
                doc_chunks.append(
                    Document(
                        id=f"{document.id}_chunk_{index}_{digest}",
                        content=chunk_text,
                        metadata=metadata,
                        score=document.score,
                        source=document.source,
                    )
                )
            for position, chunk in enumerate(doc_chunks):
                if position > 0:
                    chunk.metadata["prev_id"] = doc_chunks[position - 1].id
                if position + 1 < len(doc_chunks):
                    chunk.metadata["next_id"] = doc_chunks[position + 1].id
            # Consecutive pages of one file (PDF pages, slides): link the boundary chunks, so
            # neighbour expansion (neighbor_window) can follow text across a page break.
            if doc_chunks and previous_last is not None and self._consecutive(previous_document, document):
                previous_last.metadata["next_id"] = doc_chunks[0].id
                doc_chunks[0].metadata["prev_id"] = previous_last.id
            if doc_chunks:
                previous_last, previous_document = doc_chunks[-1], document
            chunks.extend(doc_chunks)

        logger.info(
            "ChunkSplitter strategy=%s created %d chunks from %d documents",
            self._strategy,
            len(chunks),
            len(documents),
        )
        return chunks

    @staticmethod
    def _consecutive(previous: Optional[Document], current: Document) -> bool:
        """True when ``current`` is the next page/slide of the same file as ``previous``."""

        if previous is None:
            return False
        file_a, file_b = previous.metadata.get("file"), current.metadata.get("file")
        if not file_a or file_a != file_b:
            return False
        for key in ("page", "slide", "chapter"):
            a, b = previous.metadata.get(key), current.metadata.get(key)
            if isinstance(a, int) and isinstance(b, int):
                return b == a + 1
        return False

    def _split_document(self, document: Document) -> Tuple[List[_Piece], str]:
        """Split one document; returns its pieces and the strategy label that produced them."""

        content = document.content
        text = content.strip()
        if not text:
            return [], self._strategy
        if self._strategy == "embedding":
            pieces = self._split_embedding_pieces(content, document)
            if pieces is not None:
                return pieces, self._strategy
            parts = self._split_recursive(text)
            return self._locate(content, self._normalize_chunks(parts)), FALLBACK_STRATEGY_LABEL

        if self._strategy == "recursive":
            parts = self._split_recursive(text)
        elif self._strategy == "llm":
            parts = self._split_llm(text)
        else:
            parts = self._split_fixed(text)
        return self._locate(content, self._normalize_chunks(parts)), self._strategy

    @staticmethod
    def _locate(content: str, parts: List[str]) -> List[_Piece]:
        """Attach character offsets to chunks that are verbatim slices of ``content`` (in order)."""

        pieces: List[_Piece] = []
        cursor = 0
        for part in parts:
            found = content.find(part, cursor)
            if found < 0:
                found = content.find(part)
            if found < 0:
                pieces.append(_Piece(part))  # rebuilt text (e.g. recursive overlap): no offsets
                continue
            pieces.append(_Piece(part, found, found + len(part)))
            cursor = found + 1  # overlapping chunks start before the previous one ends
        return pieces

    def _split_fixed(self, text: str) -> List[str]:
        """Pack whole sentences (or lines) into chunks of at most ``chunk_size`` units, with overlap.

        A sentence or line longer than ``chunk_size`` is hard-split first, so no chunk is
        ever larger than the budget (e.g. a log, CSV or one-line-per-record file without
        sentence punctuation). Original whitespace and line breaks are preserved.
        """

        sentences: List[str] = []
        for sentence in self._split_sentence_segments(text) or [text]:
            if self._count_tokens(sentence) > self._chunk_size:
                sentences.extend(self._hard_split(sentence, overlap=0))
            else:
                sentences.append(sentence)

        sentence_token_counts = [self._count_tokens(sentence) for sentence in sentences]
        chunks: List[str] = []

        cursor = 0
        while cursor < len(sentences):
            start = cursor
            end = cursor
            total_tokens = 0

            while end < len(sentences):
                sentence_tokens = sentence_token_counts[end]
                if total_tokens > 0 and (total_tokens + sentence_tokens) > self._chunk_size:
                    break
                total_tokens += sentence_tokens
                end += 1
                if total_tokens >= self._chunk_size:
                    break

            if end == start:
                end = min(start + 1, len(sentences))

            chunk_text = "".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(sentences):
                break

            overlap_tokens = 0
            overlap_sentences = 0
            back = end - 1
            # Overlap whole sentences, but never so much that the next chunk would overflow.
            while back > start and overlap_tokens + sentence_token_counts[back] <= self._overlap:
                overlap_tokens += sentence_token_counts[back]
                overlap_sentences += 1
                back -= 1

            cursor = max(start + 1, end - overlap_sentences)

        return chunks

    def _split_recursive(self, text: str) -> List[str]:
        """Recursively split text using hierarchical separators, then apply overlap."""

        # Leave room for the overlap that is prepended afterwards, so chunks stay within chunk_size.
        limit = max(self._chunk_size - self._overlap, 1)
        raw_chunks = self._recursive_split(text, self._separators, limit)
        return self._apply_overlap(raw_chunks)

    def _recursive_split(self, text: str, separators: Sequence[str], limit: Optional[int] = None) -> List[str]:
        """Recursively split oversized text using the next available separator."""

        budget: int = limit or self._chunk_size
        cleaned = text.strip()
        if not cleaned:
            return []

        if self._count_tokens(cleaned) <= budget:
            return [cleaned]

        if not separators:
            return self._hard_split(cleaned, overlap=0, limit=budget)

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            return self._hard_split(cleaned, overlap=0, limit=budget)

        pieces = cleaned.split(separator)
        pieces = [piece.strip() for piece in pieces if piece and piece.strip()]
        if len(pieces) <= 1:
            return self._recursive_split(cleaned, next_separators, budget)

        merged: List[str] = []
        current: List[str] = []
        current_tokens = 0
        sep_tokens = 1 if separator.strip() in {"。", "！", "？"} else 0

        for piece in pieces:
            piece_tokens = self._count_tokens(piece)
            if piece_tokens > budget:
                if current:
                    merged.append(separator.join(current).strip())
                    current = []
                    current_tokens = 0
                merged.extend(self._recursive_split(piece, next_separators, budget))
                continue

            # CJK full stops are size units themselves (". " or "\n" merge into the previous word).
            joined_tokens = current_tokens + piece_tokens + sep_tokens if current else piece_tokens
            if current and joined_tokens > budget:
                merged.append(separator.join(current).strip())
                current = [piece]
                current_tokens = piece_tokens
            else:
                current.append(piece)
                current_tokens = joined_tokens

        if current:
            merged.append(separator.join(current).strip())

        return [chunk for chunk in merged if chunk]

    def _hard_token_split(self, text: str) -> List[str]:
        """Force-split by size units when no separator can reduce size."""

        return self._hard_split(text, overlap=self._overlap)

    def _hard_split(self, text: str, overlap: int, limit: Optional[int] = None) -> List[str]:
        """Cut ``text`` into windows of ``chunk_size`` units (words, or CJK/Thai characters).

        Slices the original string, so whitespace and line breaks inside a window survive.
        """

        return [text[start:end] for start, end in self._hard_split_spans(text, 0, len(text), overlap, limit)]

    def _hard_split_spans(
        self, text: str, start: int, end: int, overlap: int = 0, limit: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """``(start, end)`` offsets of windows of at most ``limit`` (default ``chunk_size``) inside ``text[start:end]``."""

        size = limit or self._chunk_size
        spans = [(a + start, b + start) for a, b in unit_spans(text[start:end])]
        if not spans:
            return []
        if self.token_counter_name == "units":
            if len(spans) <= size:
                return [(spans[0][0], spans[-1][1])]
            windows: List[Tuple[int, int]] = []
            step = max(size - overlap, 1)
            index = 0
            while index < len(spans):
                last = min(index + size, len(spans))
                windows.append((spans[index][0], spans[last - 1][1]))
                if last >= len(spans):
                    break
                index += step
            return windows

        # A real tokenizer: grow each window by binary search on unit boundaries.
        windows = []
        index = 0
        while index < len(spans):
            low, high = index + 1, len(spans)
            while low < high:
                middle = (low + high + 1) // 2
                if self._count_tokens(text[spans[index][0] : spans[middle - 1][1]]) <= size:
                    low = middle
                else:
                    high = middle - 1
            last = low
            windows.append((spans[index][0], spans[last - 1][1]))
            if last >= len(spans):
                break
            back = 0
            while back + 1 < last - index and self._count_tokens(text[spans[last - back - 1][0] : spans[last - 1][1]]) <= overlap:
                back += 1
            index = max(index + 1, last - back)
        return windows

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Prefix each chunk with the last ``overlap`` units of the previous chunk."""

        if self._overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            previous = chunks[index - 1]
            overlap_text = self._tail_within(previous, self._overlap)
            current = chunks[index].strip()
            if overlap_text and not current.startswith(overlap_text):
                joiner = "" if has_no_space_script(overlap_text[-1:]) and has_no_space_script(current[:1]) else " "
                current = f"{overlap_text}{joiner}{current}".strip()
            overlapped.append(current)
        return overlapped

    def _semantic_units(self, text: str) -> List[Tuple[int, int]]:
        """Character spans of the units the embedding strategy compares, in order.

        A Markdown table (consecutive ``| ... |`` rows) or a run of list items (``-``, ``*``, ``+``,
        ``1.``, ``1)`` with indented continuation lines, blank lines allowed between items) is one
        unit; other text is split into sentences, not after an abbreviation. Spans are trimmed of
        surrounding whitespace, so ``text[start:end]`` is the unit's exact text.
        """

        lines: List[Tuple[int, int]] = []
        position = 0
        for line in text.splitlines(keepends=True):
            lines.append((position, position + len(line)))
            position += len(line)
        units: List[Tuple[int, int]] = []
        prose_start: Optional[int] = None
        index = 0

        def flush_prose(until: int) -> None:
            nonlocal prose_start
            if prose_start is not None:
                units.extend(self._sentence_spans(text, prose_start, until))
                prose_start = None

        while index < len(lines):
            line_start, line_end = lines[index]
            line = text[line_start:line_end]
            kind = "table" if _TABLE_ROW_RE.match(line.rstrip("\r\n")) else "list" if _LIST_ITEM_RE.match(line) else None
            if kind is None:
                if prose_start is None and line.strip():
                    prose_start = line_start
                index += 1
                continue
            flush_prose(line_start)
            block_end = line_end
            index += 1
            while index < len(lines):
                candidate = text[lines[index][0] : lines[index][1]]
                stripped = candidate.rstrip("\r\n")
                if kind == "table" and _TABLE_ROW_RE.match(stripped):
                    pass
                elif kind == "list" and (_LIST_ITEM_RE.match(candidate) or (candidate[:1] in " \t" and candidate.strip())):
                    pass
                elif kind == "list" and not candidate.strip():
                    following = next((text[a:b] for a, b in lines[index + 1 :] if text[a:b].strip()), "")
                    if not _LIST_ITEM_RE.match(following):
                        break
                else:
                    break
                block_end = lines[index][1]
                index += 1
            units.append(_trim(text, line_start, block_end))
        flush_prose(len(text))
        return [(a, b) for a, b in units if b > a]

    def _sentence_spans(self, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Trimmed sentence spans in ``text[start:end]``; no break right after an abbreviation."""

        spans: List[Tuple[int, int]] = []
        position = start
        region = text[start:end]
        for match in _SENTENCE_BOUNDARY_RE.finditer(region):
            boundary = start + match.end()
            if boundary <= position:
                continue
            candidate = text[position : start + match.start()]
            if not candidate.strip():
                continue
            if "\n" not in match.group(0) and self._ends_with_abbreviation(candidate):
                continue  # "Dr. Sharma": keep going until the next boundary
            spans.append(_trim(text, position, start + match.start()))
            position = boundary
        if text[position:end].strip():
            spans.append(_trim(text, position, end))
        return [(a, b) for a, b in spans if b > a]

    def _ends_with_abbreviation(self, sentence: str) -> bool:
        """True when the last word of ``sentence`` is a known abbreviation (case-insensitive)."""

        words = sentence.split()
        return bool(words) and words[-1].lower() in self._abbreviations

    def _fit_units(self, text: str, units: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Split units larger than ``chunk_size``: blocks at line borders, then hard windows."""

        fitted: List[Tuple[int, int]] = []
        for start, end in units:
            if self._count_tokens(text[start:end]) <= self._chunk_size:
                fitted.append((start, end))
                continue
            current: Optional[Tuple[int, int]] = None
            position = start
            for line in text[start:end].splitlines(keepends=True):
                line_span = _trim(text, position, position + len(line))
                position += len(line)
                if line_span[1] <= line_span[0]:
                    continue
                if self._count_tokens(text[line_span[0] : line_span[1]]) > self._chunk_size:
                    if current is not None:
                        fitted.append(current)
                        current = None
                    fitted.extend(self._hard_split_spans(text, line_span[0], line_span[1]))
                    continue
                if current is not None and self._count_tokens(text[current[0] : line_span[1]]) <= self._chunk_size:
                    current = (current[0], line_span[1])
                else:
                    if current is not None:
                        fitted.append(current)
                    current = line_span
            if current is not None:
                fitted.append(current)
        return fitted

    def _split_embedding_pieces(self, content: str, document: Document) -> Optional[List[_Piece]]:
        """Semantic chunking on character offsets; ``None`` means "use the recursive fallback"."""

        if self._embedding is None:
            self._report_fallback(document, RuntimeError("no embedding model configured"))
            return None
        units = self._fit_units(content, self._semantic_units(content))
        if not units:
            return []
        if len(units) == 1:
            start, end = units[0]
            return [_Piece(content[start:end], start, end)]
        texts = [content[start:end] for start, end in units]
        # For the gap after unit i, compare the two units before it with the two after it
        # (single sentences are too short to embed reliably; overlapping windows blur the boundary).
        # right(i) == left(i + 2), so de-duplicated this is about one embedding per unit.
        gaps = range(len(units) - 1)
        lefts = [" ".join(texts[max(0, i - 1) : i + 1]) for i in gaps]
        rights = [" ".join(texts[i + 1 : i + 3]) for i in gaps]
        unique = list(dict.fromkeys(lefts + rights))
        try:
            vectors = dict(zip(unique, self._embedding.embed_documents(unique)))
            similarities = [_cosine(vectors[left], vectors[right]) for left, right in zip(lefts, rights)]
        except Exception as exc:  # noqa: BLE001 - policy decides: raise or a visible fallback
            if self._on_embedding_failure == "raise":
                raise EmbeddingChunkingError(f"embedding chunking failed for document {document.id!r}: {exc}") from exc
            self._report_fallback(document, exc)
            return None
        if self._semantic_breakpoint == "percentile":
            distances = [1.0 - value for value in similarities]
            cutoff = _percentile(distances, self._semantic_threshold)
            breaks = {i for i, distance in enumerate(distances) if distance > cutoff}
        else:
            breaks = {i for i, value in enumerate(similarities) if value < self._semantic_threshold}

        groups: List[List[int]] = []
        current: List[int] = []
        for index in range(len(units)):
            if current and self._count_tokens(content[units[current[0]][0] : units[index][1]]) > self._chunk_size:
                groups.append(current)
                current = []
            current.append(index)
            if index in breaks and self._count_tokens(content[units[current[0]][0] : units[index][1]]) >= self._semantic_min_tokens:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        spans = [(units[group[0]][0], units[group[-1]][1]) for group in groups]
        spans = self._merge_small(content, spans, self._semantic_min_tokens)
        spans = self._merge_small(content, spans, self._min_tokens)
        return [_Piece(content[start:end], start, end) for start, end in spans]

    def _merge_small(self, content: str, spans: List[Tuple[int, int]], minimum: int) -> List[Tuple[int, int]]:
        """Merge spans smaller than ``minimum`` into a neighbour when the result fits ``chunk_size``."""

        merged: List[Tuple[int, int]] = []
        for start, end in spans:
            if merged and self._count_tokens(content[start:end]) < minimum:
                previous = merged[-1]
                if self._count_tokens(content[previous[0] : end]) <= self._chunk_size:
                    merged[-1] = (previous[0], end)
                    continue
            merged.append((start, end))
        if len(merged) >= 2 and self._count_tokens(content[merged[0][0] : merged[0][1]]) < minimum:
            if self._count_tokens(content[merged[0][0] : merged[1][1]]) <= self._chunk_size:
                merged[0:2] = [(merged[0][0], merged[1][1])]
        return merged

    def _report_fallback(self, document: Document, error: Exception) -> None:
        """Log and report a recursive fallback (RAG counts it and emits an ``on_event`` warning)."""

        logger.warning("Embedding chunking fell back to recursive for document %s: %s", document.id, error)
        if self.on_fallback is not None:
            try:
                self.on_fallback(document, error)
            except Exception as exc:  # noqa: BLE001 - reporting must never break ingestion
                logger.warning("on_fallback hook failed: %s", exc)

    def _tail_within(self, text: str, budget: int) -> str:
        """The longest suffix of ``text`` (on unit boundaries) whose size is at most ``budget``."""

        spans = unit_spans(text)
        if not spans:
            return ""
        if self.token_counter_name == "units":
            return text.strip() if len(spans) <= budget else text[spans[-budget][0] :].strip()
        count = 0
        while count < len(spans) and self._count_tokens(text[spans[len(spans) - count - 1][0] :]) <= budget:
            count += 1
        return text[spans[len(spans) - count][0] :].strip() if count else ""

    def _split_llm(self, text: str) -> List[str]:
        """Use an LLM to propose semantic chunk boundaries with recursive fallback."""

        if self._count_tokens(text) <= self._chunk_size:
            return [text]

        if self._llm_client is None:
            logger.warning("No llm_client for LLM chunking; falling back to recursive")
            return self._split_recursive(text)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_recursive(text)

        numbered = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1))
        prompt = (
            "Split the numbered sentences into coherent semantic chunks for retrieval.\n"
            f"Each chunk should stay under approximately {self._chunk_size} words/tokens.\n"
            "Prefer keeping related ideas together and splitting at topic boundaries.\n"
            "Return ONLY lines in this exact format (1-based inclusive indexes):\n"
            "CHUNK: start-end\n"
            "Example:\n"
            "CHUNK: 1-3\n"
            "CHUNK: 4-6\n\n"
            f"Sentences:\n{numbered}"
        )
        system = "You are a document chunking assistant. Output only CHUNK lines."

        try:
            raw = self._llm_client.complete(prompt=prompt, system=system)
            ranges = self._parse_chunk_ranges(raw, sentence_count=len(sentences))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM chunking failed (%s); falling back to recursive", exc)
            return self._split_recursive(text)

        if not ranges:
            logger.warning("LLM chunking returned no usable ranges; falling back to recursive")
            return self._split_recursive(text)

        chunks: List[str] = []
        for start, end in ranges:
            chunk_text = " ".join(sentences[start - 1 : end]).strip()
            if has_no_space_script(chunk_text):
                chunk_text = "".join(sentences[start - 1 : end]).strip()
            if not chunk_text:
                continue
            if self._count_tokens(chunk_text) > self._chunk_size * 2:
                chunks.extend(self._split_recursive(chunk_text))
            else:
                chunks.append(chunk_text)

        if not chunks:
            return self._split_recursive(text)

        return self._apply_overlap(chunks)

    def _parse_chunk_ranges(self, raw: str, sentence_count: int) -> List[tuple[int, int]]:
        """Parse CHUNK: start-end lines into validated inclusive ranges."""

        ranges: List[tuple[int, int]] = []
        pattern = re.compile(r"CHUNK\s*:\s*(\d+)\s*[-:]\s*(\d+)", re.IGNORECASE)
        for match in pattern.finditer(raw or ""):
            start = int(match.group(1))
            end = int(match.group(2))
            if start < 1 or end < start or end > sentence_count:
                continue
            ranges.append((start, end))

        if not ranges:
            return []

        # Fill gaps / enforce coverage when the model skips sentences.
        ranges.sort(key=lambda item: item[0])
        covered: List[tuple[int, int]] = []
        cursor = 1
        for start, end in ranges:
            if start > cursor:
                covered.append((cursor, start - 1))
            covered.append((max(start, cursor), end))
            cursor = end + 1
        if cursor <= sentence_count:
            covered.append((cursor, sentence_count))
        return covered

    def _normalize_chunks(self, chunks: List[str]) -> List[str]:
        """Merge undersized trailing chunks without exceeding the size budget."""

        cleaned = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        if not cleaned:
            return []

        if (
            len(cleaned) >= 2
            and self._count_tokens(cleaned[-1]) < self._min_tokens
            and (self._count_tokens(cleaned[-2]) + self._count_tokens(cleaned[-1])) <= self._chunk_size
        ):
            cleaned[-2] = f"{cleaned[-2]} {cleaned[-1]}".strip()
            cleaned.pop()

        normalized: List[str] = []
        for chunk in cleaned:
            token_count = self._count_tokens(chunk)
            if (
                normalized
                and token_count < self._min_tokens
                and (self._count_tokens(normalized[-1]) + token_count) <= self._chunk_size
            ):
                normalized[-1] = f"{normalized[-1]} {chunk}".strip()
                continue
            normalized.append(chunk)
        return normalized

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentence-like units (stripped)."""

        return [segment.strip() for segment in self._split_sentence_segments(text) if segment.strip()]

    def _split_sentence_segments(self, text: str) -> List[str]:
        """Split text at sentence/line boundaries, keeping each segment's trailing whitespace.

        ``"".join(segments)`` reproduces the text, so chunks keep their line breaks.
        """

        segments: List[str] = []
        position = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(text):
            end = match.end()
            if end <= position:
                continue
            if text[position:end].strip():
                segments.append(text[position:end])
            elif segments:
                segments[-1] += text[position:end]
            position = end
        if text[position:].strip():
            segments.append(text[position:])
        return segments

    def _count_tokens(self, text: str) -> int:
        """Size of ``text`` with the configured ``token_counter`` (default: units, not model tokens)."""

        return self._token_counter(text)

    @property
    def token_counter(self) -> Callable[[str], int]:
        """The function that measures chunk sizes (shared with RAG's embedding cost estimate)."""

        return self._token_counter

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        """Normalize and validate a chunking strategy name."""

        key = (strategy or "fixed").strip().lower()
        normalized = CHUNKING_STRATEGY_ALIASES.get(key)
        if normalized is None:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Supported values: {sorted(SUPPORTED_CHUNKING_STRATEGIES)}"
            )
        return normalized


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0 for a zero vector)."""

    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    """Shrink ``(start, end)`` so the span does not begin or end with whitespace."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
