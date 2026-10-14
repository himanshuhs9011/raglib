"""Chunk splitting utilities with fixed, recursive, and LLM strategies."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

SUPPORTED_CHUNKING_STRATEGIES = ("fixed", "recursive", "llm")
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
}

# Prefer coarser structure first, then finer units (LangChain-style recursive split).
DEFAULT_RECURSIVE_SEPARATORS: Sequence[str] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
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
    ):
        """Configure chunk size, overlap, strategy, and optional LLM client."""

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

        if self._strategy == "llm" and self._llm_client is None:
            logger.warning(
                "chunking_strategy='llm' without llm_client; semantic splits will fall back to recursive"
            )

    @property
    def strategy(self) -> str:
        """Return the normalized chunking strategy name."""

        return self._strategy

    def split(self, documents: List[Document]) -> List[Document]:
        """Split input documents into overlapping chunks with preserved metadata."""

        chunks: List[Document] = []
        for document in documents:
            split_texts = self._split_document(document)
            for index, chunk_text in enumerate(split_texts, start=1):
                chunk_tokens = self._count_tokens(chunk_text)
                chunks.append(
                    Document(
                        id=f"{document.id}_chunk_{index}",
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "parent_id": document.id,
                            "chunk_index": index,
                            "chunk_tokens": chunk_tokens,
                            "chunking_strategy": self._strategy,
                        },
                        score=document.score,
                        source=document.source,
                    )
                )

        logger.info(
            "ChunkSplitter strategy=%s created %d chunks from %d documents",
            self._strategy,
            len(chunks),
            len(documents),
        )
        return chunks

    def _split_document(self, document: Document) -> List[str]:
        """Split one document with the configured strategy."""

        text = document.content.strip()
        if not text:
            return []

        if self._strategy == "recursive":
            parts = self._split_recursive(text)
        elif self._strategy == "llm":
            parts = self._split_llm(text)
        else:
            parts = self._split_fixed(text)

        return self._normalize_chunks(parts)

    def _split_fixed(self, text: str) -> List[str]:
        """Split text into sentence-aware fixed-size overlapping chunks."""

        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text]

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

            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(sentences):
                break

            overlap_tokens = 0
            overlap_sentences = 0
            back = end - 1
            while back >= start and overlap_tokens < self._overlap:
                overlap_tokens += sentence_token_counts[back]
                overlap_sentences += 1
                back -= 1

            cursor = max(start + 1, end - overlap_sentences)

        return chunks

    def _split_recursive(self, text: str) -> List[str]:
        """Recursively split text using hierarchical separators, then apply overlap."""

        raw_chunks = self._recursive_split(text, self._separators)
        return self._apply_overlap(raw_chunks)

    def _recursive_split(self, text: str, separators: Sequence[str]) -> List[str]:
        """Recursively split oversized text using the next available separator."""

        cleaned = text.strip()
        if not cleaned:
            return []

        if self._count_tokens(cleaned) <= self._chunk_size:
            return [cleaned]

        if not separators:
            return self._hard_token_split(cleaned)

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            return self._hard_token_split(cleaned)

        pieces = cleaned.split(separator)
        pieces = [piece.strip() for piece in pieces if piece and piece.strip()]
        if len(pieces) <= 1:
            return self._recursive_split(cleaned, next_separators)

        merged: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = self._count_tokens(piece)
            if piece_tokens > self._chunk_size:
                if current:
                    merged.append(separator.join(current).strip())
                    current = []
                    current_tokens = 0
                merged.extend(self._recursive_split(piece, next_separators))
                continue

            if current and (current_tokens + piece_tokens) > self._chunk_size:
                merged.append(separator.join(current).strip())
                current = [piece]
                current_tokens = piece_tokens
            else:
                current.append(piece)
                current_tokens += piece_tokens

        if current:
            merged.append(separator.join(current).strip())

        return [chunk for chunk in merged if chunk]

    def _hard_token_split(self, text: str) -> List[str]:
        """Force-split by whitespace tokens when no separator can reduce size."""

        tokens = text.split()
        if not tokens:
            return []

        chunks: List[str] = []
        step = max(self._chunk_size - self._overlap, 1)
        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(tokens):
                break
            start += step
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Add trailing-token overlap between consecutive recursive chunks."""

        if self._overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            previous_tokens = chunks[index - 1].split()
            overlap_text = " ".join(previous_tokens[-self._overlap :]) if previous_tokens else ""
            current = chunks[index].strip()
            if overlap_text and not current.startswith(overlap_text):
                current = f"{overlap_text} {current}".strip()
            overlapped.append(current)
        return overlapped

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
            if not chunk_text:
                continue
            if self._count_tokens(chunk_text) > self._chunk_size * 2:
                # Guard against oversized LLM groups.
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
        """Split text into sentence-like units while preserving punctuation."""

        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [piece.strip() for piece in pieces if piece and piece.strip()]

    def _count_tokens(self, text: str) -> int:
        """Estimate token count using whitespace-separated words."""

        return len(text.split())

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
