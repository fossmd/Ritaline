"""JSONL persistence and training-dataset export."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from .exceptions import GenerationError
from .models import QAPair


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    """Load Q&A records from JSONL, keeping the last record for each slot."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []

    by_slot: dict[int, QAPair] = {}
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    pair = QAPair.model_validate_json(line)
                except ValidationError as exc:
                    raise GenerationError(
                        f"Invalid Q&A record in {dataset_path} at line {line_number}: {exc}"
                    ) from exc
                by_slot[pair.slot_index] = pair
    except OSError as exc:
        raise GenerationError(f"Could not read dataset {dataset_path}: {exc}") from exc
    return [by_slot[index] for index in sorted(by_slot)]


def append_qa_pair(path: str | Path, pair: QAPair) -> None:
    """Append one validated record to a resumable JSONL file."""
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write(pair.model_dump_json())
        handle.write("\n")
        handle.flush()


def rewrite_qa_pairs(path: str | Path, pairs: Iterable[QAPair]) -> None:
    """Atomically rewrite JSONL records in deterministic slot order."""
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    sorted_pairs = sorted(pairs, key=lambda pair: pair.slot_index)
    with temporary.open("w", encoding="utf-8") as handle:
        for pair in sorted_pairs:
            handle.write(pair.model_dump_json())
            handle.write("\n")
    temporary.replace(dataset_path)


def export_training_jsonl(
    pairs: Iterable[QAPair],
    path: str | Path,
) -> Path:
    """Export conversational prompt-completion JSONL for TRL or other trainers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for pair in sorted(pairs, key=lambda item: item.slot_index):
            handle.write(json.dumps(pair.training_record(), ensure_ascii=False))
            handle.write("\n")
    temporary.replace(output_path)
    return output_path
