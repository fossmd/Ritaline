"""Prompt construction for grounded Q&A generation."""

from __future__ import annotations

from .config import GenerationConfig, QAStyle
from .models import SourceDocument, TextChunk


def build_messages(
    generation: GenerationConfig,
    style: QAStyle,
    document: SourceDocument,
    chunk: TextChunk,
) -> list[dict[str, str]]:
    """Build system/user messages for one requested Q&A pair."""
    user_prompt = generation.user_prompt_template.format(
        style_name=style.name,
        style_instruction=style.instruction,
        source_name=document.name,
        page_range=chunk.page_range,
        chunk=chunk.text,
    )
    return [
        {"role": "system", "content": generation.system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
