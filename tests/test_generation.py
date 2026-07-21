from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ritaline.config import EndpointConfig, GenerationConfig, QAStyle
from ritaline.generation import QAGenerator
from ritaline.models import DocumentPage, SourceDocument


class FakeClient:
    def __init__(self, _: EndpointConfig):
        self.counter = 0

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.counter += 1
        prompt = messages[-1]["content"]
        style_line = next(line for line in prompt.splitlines() if line.startswith("Q&A style:"))
        style = style_line.split(":", 1)[1].strip()
        return {
            "question": f"Question {self.counter} for {style}?",
            "answer": f"Answer {self.counter}.",
        }


async def test_generation_is_exact_and_round_robin(tmp_path: Path) -> None:
    document = SourceDocument(
        path=Path("source.txt"),
        pages=(DocumentPage(number=1, text="Grounded source text. " * 200),),
    )
    endpoint = EndpointConfig(
        base_url="https://example.invalid",
        model="generator-model",
        api_key_env=None,
        max_concurrency=3,
    )
    generation = GenerationConfig(
        qa_count=7,
        styles=[
            QAStyle(name="factual", instruction="Ask a fact"),
            QAStyle(name="explanatory", instruction="Ask for an explanation"),
            QAStyle(name="applied", instruction="Ask an application"),
        ],
        chunk_size_chars=500,
        chunk_overlap_chars=50,
        min_chunk_chars=50,
        output_path=tmp_path / "raw.jsonl",
        training_dataset_path=tmp_path / "train.jsonl",
    )

    pairs = await QAGenerator(
        endpoint,
        generation,
        client_factory=FakeClient,
    ).generate(document, resume=False)

    assert len(pairs) == 7
    assert [pair.style for pair in pairs] == [
        "factual",
        "explanatory",
        "applied",
        "factual",
        "explanatory",
        "applied",
        "factual",
    ]
    assert (tmp_path / "raw.jsonl").exists()
    assert (tmp_path / "train.jsonl").exists()
