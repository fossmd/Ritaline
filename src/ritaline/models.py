"""Core data models used throughout Ritaline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentPage(BaseModel):
    """Text extracted from one source page."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    text: str


class SourceDocument(BaseModel):
    """A normalized PDF or text document."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    pages: tuple[DocumentPage, ...]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def content_sha256(self) -> str:
        """Stable fingerprint of normalized page text and page numbers."""
        payload = "\n\f\n".join(
            f"{page.number}\n{page.text}" for page in self.pages
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class TextChunk(BaseModel):
    """A chunk of source text with provenance metadata."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @property
    def page_range(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


class QAPair(BaseModel):
    """A generated Q&A pair plus enough metadata for auditing and resuming."""

    slot_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    style: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    generation_model: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def training_record(self) -> dict[str, Any]:
        """Return TRL's conversational prompt-completion representation."""
        return {
            "prompt": [{"role": "user", "content": self.question}],
            "completion": [{"role": "assistant", "content": self.answer}],
            "style": self.style,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
