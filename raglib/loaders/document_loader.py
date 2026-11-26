"""Document loader dispatcher for paths, folders, URLs, and raw text input."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

from raglib.loaders.exceptions import DocumentLoadError, InstructionError
from raglib.loaders.pdf_loader import PDFLoader
from raglib.loaders.pptx_loader import PPTXLoader
from raglib.loaders.structured_loaders import (
    CSVLoader,
    EPUBLoader,
    HTMLFileLoader,
    JSONLoader,
    XLSXLoader,
)
from raglib.loaders.text_loader import TextLoader
from raglib.loaders.web_loader import WebLoader
from raglib.loaders.word_loader import WordLoader
from raglib.schemas import Document
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB per file
DEFAULT_MAX_PAGES = 5000  # PDF pages per file
ZIP_EXPANSION_LIMIT = 10  # a zip-based file may unpack to at most 10 x max_file_bytes
MAX_ZIP_MEMBERS = 100_000
_MAGIC = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04", ".pptx": b"PK\x03\x04", ".xlsx": b"PK\x03\x04", ".epub": b"PK\x03\x04"}

SourceInput = Union[str, os.PathLike[str], Document, Sequence[Union[str, os.PathLike[str], Document]], None]

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".adoc", ".tex", ".log", ".org",
    # source code and configuration (loaded as plain text)
    ".py", ".pyi", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".java", ".kt", ".scala", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m", ".r", ".lua", ".pl", ".sh",
    ".bash", ".ps1", ".bat", ".sql", ".graphql", ".proto", ".css", ".scss", ".vue", ".svelte",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml",
}
# Directories that never hold documents (VCS internals, dependencies, caches, virtualenvs).
SKIPPED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build", ".next", ".cache",
}
_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class DocumentLoader:
    """Dispatch source input to the correct format loader."""

    SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
        ".pdf", ".docx", ".pptx", ".html", ".htm", ".xhtml", ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
        ".xlsx", ".epub",
    }

    def __init__(
        self,
        vision_client: Optional[BaseVisionClient] = None,
        pdf_loader: Optional[PDFLoader] = None,
        word_loader: Optional[WordLoader] = None,
        pptx_loader: Optional[PPTXLoader] = None,
        text_loader: Optional[TextLoader] = None,
        web_loader: Optional[WebLoader] = None,
        html_loader: Optional[HTMLFileLoader] = None,
        csv_loader: Optional[CSVLoader] = None,
        json_loader: Optional[JSONLoader] = None,
        xlsx_loader: Optional[XLSXLoader] = None,
        epub_loader: Optional[EPUBLoader] = None,
        allow_private_urls: bool = False,
        allowed_dirs: Optional[Sequence[Union[str, os.PathLike]]] = None,
        max_file_bytes: Optional[int] = DEFAULT_MAX_FILE_BYTES,
        max_pages: Optional[int] = DEFAULT_MAX_PAGES,
    ):
        """Initialize all concrete loaders used by the dispatcher.

        ``allowed_dirs`` restricts file/folder sources to those directories (use it when
        sources come from untrusted users); ``allow_private_urls`` disables the SSRF guard.

        Limits (``None`` turns one off): ``max_file_bytes`` (default 100 MB) per file, checked
        before parsing; zip-based files (DOCX, PPTX, XLSX, EPUB) may unpack to at most ten times
        that, checked before unpacking (zip bombs); ``max_pages`` (default 5000) per PDF, checked
        before text is extracted. A file whose first bytes do not match its extension (a ".pdf"
        that is not a PDF) is refused. Any of these, or a parser failure, raises ``DocumentLoadError``.
        """

        self.allowed_dirs = [os.path.realpath(os.fspath(path)) for path in allowed_dirs] if allowed_dirs else None
        for name, value in (("max_file_bytes", max_file_bytes), ("max_pages", max_pages)):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        self.max_file_bytes = max_file_bytes
        self.max_pages = max_pages

        self.pdf_loader = pdf_loader or PDFLoader(vision_client=vision_client)
        self.word_loader = word_loader or WordLoader()
        self.pptx_loader = pptx_loader or PPTXLoader()
        self.text_loader = text_loader or TextLoader()
        self.web_loader = web_loader or WebLoader(allow_private_hosts=allow_private_urls, pdf_loader=self.pdf_loader)
        self.html_loader = html_loader or HTMLFileLoader()
        self.csv_loader = csv_loader or CSVLoader()
        self.json_loader = json_loader or JSONLoader()
        self.xlsx_loader = xlsx_loader or XLSXLoader()
        self.epub_loader = epub_loader or EPUBLoader()
        if self.max_pages is not None and getattr(self.pdf_loader, "max_pages", None) is None:
            self.pdf_loader.max_pages = self.max_pages

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

        if "\n" not in normalized and len(normalized) < 200:
            logger.info("Source %r is not an existing file or URL; indexing it as raw text", normalized[:60])
        stable_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        yield [Document(f"raw-{stable_id}", normalized, {"loader": "raw_text"}, 0.0, "text")]

    def iter_files(self, folder_path: str) -> Iterator[str]:
        """Yield supported file paths under a folder (sorted, recursive).

        Hidden and tooling directories (``.git``, ``node_modules``, virtualenvs, caches) are
        skipped. Files without an extension (README, LICENSE, ...) are included and are
        dropped later if they turn out to be binary.
        """

        for root, dirs, filenames in os.walk(folder_path):
            dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS and not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
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
        """Detect single-line strings that clearly refer to a file or folder path.

        A typo such as ``RAG("./my_docs")`` must raise ``FileNotFoundError`` instead of
        indexing the string ``"./my_docs"`` as a document.
        """

        if "\n" in value or "\r" in value or len(value) > 260:
            return False
        has_separator = "/" in value or "\\" in value
        if Path(value).suffix.lower() in self.SUPPORTED_EXTENSIONS and (has_separator or " " not in value):
            return True
        if any(char.isspace() for char in value):
            return False
        if value.startswith(("./", ".\\", "../", "..\\", "/", "\\", "~")) or _DRIVE_RE.match(value):
            return True
        return has_separator

    def _load_folder(self, folder_path: str) -> List[Document]:
        """Load all supported files recursively from a folder path."""

        return [document for batch in self._iter_folder(folder_path) for document in batch]

    def _load_file(self, path: str) -> List[Document]:
        """Check the file against the limits, then parse it; parser failures raise ``DocumentLoadError``."""

        extension = Path(path).suffix.lower()
        self._check_file(path, extension)
        try:
            return self._parse_file(path, extension)
        except (DocumentLoadError, InstructionError, ImportError):
            raise
        except Exception as exc:  # noqa: BLE001 - a malformed file must not surface as a parser crash
            raise DocumentLoadError(f"could not parse {Path(path).name!r} ({type(exc).__name__})") from exc

    def _check_file(self, path: str, extension: str) -> None:
        """Size, type (first bytes) and unpacked-size checks before any parser sees the file."""

        name = Path(path).name
        size = os.path.getsize(path)
        if self.max_file_bytes is not None and size > self.max_file_bytes:
            raise DocumentLoadError(
                f"{name!r} is {size:,} bytes, over max_file_bytes={self.max_file_bytes:,} (RAG(max_file_bytes=...))"
            )
        expected = _MAGIC.get(extension)
        if expected is None or size == 0:
            return
        with open(path, "rb") as handle:
            head = handle.read(1024)
        if extension == ".pdf":
            if b"%PDF-" not in head:
                raise DocumentLoadError(f"{name!r} has the extension .pdf but is not a PDF file")
            return
        if not head.startswith(expected):
            raise DocumentLoadError(f"{name!r} has the extension {extension} but is not a {extension[1:].upper()} (zip) file")
        if self.max_file_bytes is None:
            return
        import zipfile

        limit = self.max_file_bytes * ZIP_EXPANSION_LIMIT
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
        except zipfile.BadZipFile as exc:
            raise DocumentLoadError(f"{name!r} is not a valid {extension[1:].upper()} file") from exc
        unpacked = sum(member.file_size for member in members)
        if unpacked > limit or len(members) > MAX_ZIP_MEMBERS:
            raise DocumentLoadError(
                f"{name!r} unpacks to {unpacked:,} bytes in {len(members):,} parts, over the limit "
                f"({limit:,} bytes = {ZIP_EXPANSION_LIMIT} x max_file_bytes, {MAX_ZIP_MEMBERS:,} parts): possibly a zip bomb"
            )

    def _parse_file(self, path: str, extension: str) -> List[Document]:
        """Load one file according to its extension."""

        if extension == ".pdf":
            return self.pdf_loader.load(path)
        if extension == ".docx":
            return self.word_loader.load(path)
        if extension == ".pptx":
            return self.pptx_loader.load(path)
        if extension in {".html", ".htm", ".xhtml"}:
            return self.html_loader.load(path)
        if extension in {".csv", ".tsv"}:
            return self.csv_loader.load(path)
        if extension in {".json", ".jsonl", ".ndjson"}:
            return self.json_loader.load(path)
        if extension == ".xlsx":
            return self.xlsx_loader.load(path)
        if extension == ".epub":
            return self.epub_loader.load(path)
        if extension in TEXT_EXTENSIONS:
            return self.text_loader.load(path)

        logger.info("No dedicated loader for '%s' (file=%s); reading it as text", extension or "(no extension)", path)
        return self.text_loader.load(path)

    def _is_url(self, value: str) -> bool:
        """Return True when the value looks like an HTTP or HTTPS URL."""

        lowered = value.lower()
        return lowered.startswith("http://") or lowered.startswith("https://")
