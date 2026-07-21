"""Document ingestion for PDF and plain-text files."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .exceptions import DocumentError
from .models import DocumentPage, SourceDocument


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_txt(path: Path) -> SourceDocument:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            normalized = _normalize_text(text)
            if not normalized:
                raise DocumentError(f"Text file contains no usable text: {path}")
            return SourceDocument(path=path, pages=(DocumentPage(number=1, text=normalized),))
        except UnicodeError as exc:
            last_error = exc
    raise DocumentError(f"Could not decode text file {path}: {last_error}")


def _read_pdf(path: Path) -> SourceDocument:
    pages: list[DocumentPage] = []
    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise DocumentError(f"Password-protected PDFs are not supported: {path}")
            for page_number, page in enumerate(document, start=1):
                text = _normalize_text(page.get_text("text", sort=True))
                if text:
                    pages.append(DocumentPage(number=page_number, text=text))
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not extract text from PDF {path}: {exc}") from exc

    if not pages:
        raise DocumentError(
            f"PDF contains no extractable text: {path}. It may be scanned and require OCR first."
        )
    return SourceDocument(path=path, pages=tuple(pages))


def load_document(path: str | Path) -> SourceDocument:
    """Load a PDF or TXT document and return normalized page text."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise DocumentError(f"Input document does not exist: {source_path}")

    suffix = source_path.suffix.casefold()
    if suffix == ".pdf":
        return _read_pdf(source_path)
    if suffix == ".txt":
        return _read_txt(source_path)
    raise DocumentError(f"Unsupported input type {suffix!r}; expected .pdf or .txt")
