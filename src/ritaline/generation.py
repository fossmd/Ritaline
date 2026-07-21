"""Round-robin, resumable Q&A dataset generation."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .chunking import chunk_document
from .config import EndpointConfig, GenerationConfig
from .dataset import append_qa_pair, export_training_jsonl, load_qa_pairs, rewrite_qa_pairs
from .exceptions import GenerationError, LLMError
from .llm import OpenAICompatibleClient
from .models import QAPair, SourceDocument, TextChunk
from .prompts import build_messages


class JSONClient(Protocol):
    async def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...

    async def __aenter__(self) -> JSONClient: ...

    async def __aexit__(self, *args: object) -> None: ...


ClientFactory = Callable[[EndpointConfig], JSONClient]


def _question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.casefold()).strip()


class QAGenerator:
    """Generate exactly N grounded Q&A pairs from document chunks."""

    def __init__(
        self,
        endpoint: EndpointConfig,
        generation: GenerationConfig,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.generation = generation
        self.client_factory = client_factory or OpenAICompatibleClient

    def chunks(self, document: SourceDocument) -> list[TextChunk]:
        return chunk_document(
            document,
            chunk_size_chars=self.generation.chunk_size_chars,
            chunk_overlap_chars=self.generation.chunk_overlap_chars,
            min_chunk_chars=self.generation.min_chunk_chars,
        )

    def preview_prompts(
        self,
        document: SourceDocument,
        *,
        count: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the first prompts without calling an endpoint."""
        chunks = self.chunks(document)
        preview: list[dict[str, Any]] = []
        limit = min(max(1, count), self.generation.qa_count)
        for slot_index in range(limit):
            style = self.generation.styles[slot_index % len(self.generation.styles)]
            chunk = chunks[slot_index % len(chunks)]
            preview.append(
                {
                    "slot_index": slot_index,
                    "style": style.name,
                    "chunk_index": chunk.index,
                    "page_range": chunk.page_range,
                    "messages": build_messages(self.generation, style, document, chunk),
                }
            )
        return preview

    def _load_resume_pairs(self, document: SourceDocument, output_path: Path) -> dict[int, QAPair]:
        existing: dict[int, QAPair] = {}
        for pair in load_qa_pairs(output_path):
            if pair.slot_index >= self.generation.qa_count:
                continue
            expected_style = self.generation.styles[
                pair.slot_index % len(self.generation.styles)
            ].name
            if (
                pair.source_name == document.name
                and pair.source_sha256 == document.content_sha256
                and pair.generation_model == self.endpoint.model
                and pair.style == expected_style
            ):
                existing[pair.slot_index] = pair
        return existing

    async def generate(
        self,
        document: SourceDocument,
        *,
        output_path: str | Path | None = None,
        training_dataset_path: str | Path | None = None,
        resume: bool = True,
    ) -> list[QAPair]:
        """Generate, persist, sort, and export the configured dataset."""
        chunks = self.chunks(document)
        raw_path = Path(output_path or self.generation.output_path)
        train_path = Path(training_dataset_path or self.generation.training_dataset_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        if resume:
            pairs_by_slot = self._load_resume_pairs(document, raw_path)
        else:
            pairs_by_slot = {}
            raw_path.unlink(missing_ok=True)

        used_questions = {
            _question_key(pair.question)
            for pair in pairs_by_slot.values()
            if _question_key(pair.question)
        }
        state_lock = asyncio.Lock()
        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.endpoint.max_concurrency)
        rng = random.Random(self.generation.seed)
        chunk_offset = rng.randrange(len(chunks)) if chunks else 0

        async with self.client_factory(self.endpoint) as client:

            async def generate_slot(slot_index: int) -> QAPair:
                style = self.generation.styles[slot_index % len(self.generation.styles)]
                last_error: Exception | None = None

                for attempt in range(self.generation.max_attempts_per_pair):
                    chunk_index = (
                        chunk_offset + slot_index + attempt * len(self.generation.styles)
                    ) % len(chunks)
                    chunk = chunks[chunk_index]
                    messages = build_messages(self.generation, style, document, chunk)
                    try:
                        async with semaphore:
                            payload = await client.generate_json(messages)
                    except LLMError as exc:
                        last_error = exc
                        continue

                    question = payload.get("question")
                    answer = payload.get("answer")
                    if not isinstance(question, str) or not isinstance(answer, str):
                        last_error = GenerationError(
                            "Model JSON must contain string fields 'question' and 'answer'"
                        )
                        continue
                    question = question.strip()
                    answer = answer.strip()
                    if not question or not answer:
                        last_error = GenerationError("Model returned an empty Q&A pair")
                        continue

                    key = _question_key(question)
                    async with state_lock:
                        if self.generation.deduplicate_questions and key in used_questions:
                            last_error = GenerationError("Model returned a duplicate question")
                            continue
                        used_questions.add(key)

                    pair = QAPair(
                        slot_index=slot_index,
                        question=question,
                        answer=answer,
                        style=style.name,
                        source_name=document.name,
                        source_sha256=document.content_sha256,
                        chunk_index=chunk.index,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        generation_model=self.endpoint.model,
                        metadata={
                            "attempt": attempt + 1,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                        },
                    )
                    async with write_lock:
                        append_qa_pair(raw_path, pair)
                    return pair

                raise GenerationError(
                    f"Could not generate slot {slot_index} ({style.name!r}) after "
                    f"{self.generation.max_attempts_per_pair} attempts: {last_error}"
                )

            missing_slots = [
                index
                for index in range(self.generation.qa_count)
                if index not in pairs_by_slot
            ]
            outcomes: dict[int, QAPair | Exception] = {}
            queue: asyncio.Queue[int] = asyncio.Queue()
            for slot_index in missing_slots:
                queue.put_nowait(slot_index)

            async def worker() -> None:
                while True:
                    try:
                        slot_index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        outcomes[slot_index] = await generate_slot(slot_index)
                    except Exception as exc:  # Preserve partial output for a later resume.
                        outcomes[slot_index] = exc
                    finally:
                        queue.task_done()

            worker_count = min(self.endpoint.max_concurrency, len(missing_slots))
            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            if workers:
                await queue.join()
                await asyncio.gather(*workers)

        failures: list[Exception] = []
        for slot_index in missing_slots:
            result = outcomes[slot_index]
            if isinstance(result, Exception):
                failures.append(result)
            else:
                pairs_by_slot[slot_index] = result

        complete = [pairs_by_slot[index] for index in sorted(pairs_by_slot)]
        rewrite_qa_pairs(raw_path, complete)
        export_training_jsonl(complete, train_path)

        if failures or len(complete) != self.generation.qa_count:
            detail = "; ".join(str(error) for error in failures[:3])
            raise GenerationError(
                f"Generated {len(complete)} of {self.generation.qa_count} requested pairs. "
                f"Partial data was saved for resume. {detail}"
            )
        return complete
