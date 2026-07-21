"""Deterministic, provenance-aware text chunking."""

from __future__ import annotations

from bisect import bisect_right

from .exceptions import DocumentError
from .models import SourceDocument, TextChunk


def _choose_boundary(text: str, start: int, target_end: int) -> int:
    """Prefer paragraph, line, sentence, then word boundaries near the target."""
    if target_end >= len(text):
        return len(text)
    search_floor = start + max(1, int((target_end - start) * 0.65))
    candidates = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")
    for delimiter in candidates:
        position = text.rfind(delimiter, search_floor, target_end)
        if position != -1:
            return position + len(delimiter)
    return target_end


def chunk_document(
    document: SourceDocument,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[TextChunk]:
    """Split document text while retaining source page ranges."""
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    pieces: list[str] = []
    page_starts: list[int] = []
    page_numbers: list[int] = []
    cursor = 0
    for page in document.pages:
        marker = f"[Page {page.number}]\n"
        page_text = f"{marker}{page.text.strip()}\n\n"
        page_starts.append(cursor)
        page_numbers.append(page.number)
        pieces.append(page_text)
        cursor += len(page_text)

    full_text = "".join(pieces).strip()
    if not full_text:
        raise DocumentError(f"Document contains no usable text: {document.path}")

    def page_for_offset(offset: int) -> int:
        index = max(0, bisect_right(page_starts, offset) - 1)
        return page_numbers[index]

    chunks: list[TextChunk] = []
    start = 0
    while start < len(full_text):
        target_end = min(len(full_text), start + chunk_size_chars)
        end = _choose_boundary(full_text, start, target_end)
        raw = full_text[start:end].strip()

        if raw and (len(raw) >= min_chunk_chars or not chunks or end == len(full_text)):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=raw,
                    char_start=start,
                    char_end=end,
                    page_start=page_for_offset(start),
                    page_end=page_for_offset(max(start, end - 1)),
                )
            )

        if end >= len(full_text):
            break
        next_start = max(start + 1, end - chunk_overlap_chars)
        while next_start < end and full_text[next_start].isspace():
            next_start += 1
        start = next_start

    if not chunks:
        raise DocumentError("Chunking produced no usable chunks")
    return chunks
