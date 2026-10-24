"""Document loader dispatcher for paths, folders, URLs, and raw text input."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

from raglib.loaders.pdf_loader import PDFLoader
from raglib.loaders.pptx_loader import PPTXLoader
from raglib.loaders.text_loader import TextLoader
from raglib.loaders.web_loader import WebLoader
from raglib.loaders.word_loader import WordLoader
from raglib.schemas import Document
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)

SourceInput = Union[str, os.PathLike[str], Document, Sequence[Union[str, os.PathLike[str], Document]], None]


class DocumentLoader:
    """Dispatch source input to the correct format loader."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx"}

    def __init__(
        self,
        vision_client: Optional[BaseVisionClient] = None,
        pdf_loader: Optional[PDFLoader] = None,
        word_loader: Optional[WordLoader] = None,
        pptx_loader: Optional[PPTXLoader] = None,
        text_loader: Optional[TextLoader] = None,
        web_loader: Optional[WebLoader] = None,
        allow_private_urls: bool = False,
        allowed_dirs: Optional[Sequence[Union[str, os.PathLike]]] = None,
    ):
        """Initialize all concrete loaders used by the dispatcher.

        ``allowed_dirs`` restricts file/folder sources to those directories (use it when
        sources come from untrusted users); ``allow_private_urls`` disables the SSRF guard.
        """

        self.allowed_dirs = [os.path.realpath(os.fspath(path)) for path in allowed_dirs] if allowed_dirs else None

        self.pdf_loader = pdf_loader or PDFLoader(vision_client=vision_client)
        self.word_loader = word_loader or WordLoader()
        self.pptx_loader = pptx_loader or PPTXLoader()
        self.text_loader = text_loader or TextLoader()
        self.web_loader = web_loader or WebLoader(allow_private_hosts=allow_private_urls)

    def load(self, source: SourceInput) -> List[Document]:
        """Load input source(s) into a list of normalized Document objects."""

        documents: List[Document] = []
        for batch in self.iter_load(source):
            documents.extend(batch)
        return documents

    def iter_load(self, source: SourceInput) -> Iterator[List[Document]]:
        """Yield documents one source unit (file, URL, raw text) at a time.

        Lets callers index huge folders incrementally instead of holding every file in memory.
        Folder files that fail to load are skipped with a warning; explicit single sources raise.
        """

        if source is None:
            return
        if isinstance(source, (list, tuple, set)):
            for item in source:
                yield from self.iter_load(item)
            return

        if isinstance(source, Document):
            if source.content.strip():
                yield [source]
            return

        is_explicit_path = isinstance(source, os.PathLike)
        if is_explicit_path:
            source = os.fspath(source)
        if not isinstance(source, str):
            raise TypeError("source must be a path, URL, raw string, Document, or list of these")
        normalized = source.strip()
        if not normalized:
            return

        if self._is_url(normalized):
            yield self.web_loader.load(normalized)
            return

        if os.path.exists(normalized):
            self._check_allowed_path(normalized)
            if os.path.isdir(normalized):
                yield from self._iter_folder(normalized)
                return
            yield self._load_file(normalized)
            return

        if is_explicit_path or self._looks_like_missing_file_path(normalized):
            raise FileNotFoundError(f"Source file or directory not found: '{normalized}'")

        stable_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        yield [Document(f"raw-{stable_id}", normalized, {"loader": "raw_text"}, 0.0, "text")]

    def iter_files(self, folder_path: str) -> Iterator[str]:
        """Yield supported file paths under a folder (sorted, recursive)."""

        for root, dirs, filenames in os.walk(folder_path):
            dirs.sort()
            for filename in sorted(filenames):
                file_path = os.path.join(root, filename)
                extension = Path(file_path).suffix.lower()
                if extension and extension not in self.SUPPORTED_EXTENSIONS:
                    continue
                yield file_path

    def _iter_folder(self, folder_path: str) -> Iterator[List[Document]]:
        """Yield the documents of each file in a folder, skipping unreadable files."""

        for file_path in self.iter_files(folder_path):
            try:
                self._check_allowed_path(file_path)  # symlinked files must stay inside allowed_dirs
                yield self._load_file(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping file=%s due to loader error: %s", file_path, exc)
                yield []

    def _check_allowed_path(self, path: str) -> None:
        """Enforce ``allowed_dirs`` (symlinks and '..' are resolved first)."""

        if self.allowed_dirs is None:
            return
        real = os.path.realpath(path)
        for base in self.allowed_dirs:
            if real == base or real.startswith(base.rstrip(os.sep) + os.sep):
                return
        raise PermissionError(f"Source path is outside allowed_dirs: {path}")

    def _looks_like_missing_file_path(self, value: str) -> bool:
        """Detect single-line strings that clearly refer to a missing file path."""

        if "\n" in value or "\r" in value or len(value) > 260:
            return False
        if " " in value and not any(sep in value for sep in ("/", "\\")):
            return False
        suffix = Path(value).suffix.lower()
        return suffix in self.SUPPORTED_EXTENSIONS

    def _load_folder(self, folder_path: str) -> List[Document]:
        """Load all supported files recursively from a folder path."""

        return [document for batch in self._iter_folder(folder_path) for document in batch]

    def _load_file(self, path: str) -> List[Document]:
        """Load one file according to its extension."""

        extension = Path(path).suffix.lower()
        if extension == ".pdf":
            return self.pdf_loader.load(path)
        if extension == ".docx":
            return self.word_loader.load(path)
        if extension == ".pptx":
            return self.pptx_loader.load(path)
        if extension in {".txt", ".md"}:
            return self.text_loader.load(path)

        logger.warning("Unknown extension '%s' for file=%s; attempting text loader fallback", extension, path)
        return self.text_loader.load(path)

    def _is_url(self, value: str) -> bool:
        """Return True when the value looks like an HTTP or HTTPS URL."""

        lowered = value.lower()
        return lowered.startswith("http://") or lowered.startswith("https://")
