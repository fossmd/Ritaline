"""High-level Python API for the complete Ritaline workflow."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import EndpointConfig, JobConfig
from .documents import load_document
from .generation import QAGenerator
from .models import QAPair
from .training import fine_tune


async def generate_dataset(
    input_path: str | Path,
    endpoint: EndpointConfig,
    job: JobConfig,
    *,
    resume: bool = True,
) -> list[QAPair]:
    """Read a document and generate the configured Q&A dataset."""
    document = load_document(input_path)
    generator = QAGenerator(endpoint, job.generation)
    return await generator.generate(document, resume=resume)


def train_model(job: JobConfig, dataset_path: str | Path | None = None) -> Path:
    """Fine-tune the configured model from a raw Ritaline Q&A JSONL file."""
    source = Path(dataset_path or job.generation.output_path)
    return fine_tune(source, job.training)


def run_pipeline(
    input_path: str | Path,
    endpoint: EndpointConfig,
    job: JobConfig,
    *,
    resume: bool = True,
    skip_training: bool = False,
) -> tuple[list[QAPair], Path | None]:
    """Generate Q&A data and optionally fine-tune the configured model."""
    pairs = asyncio.run(generate_dataset(input_path, endpoint, job, resume=resume))
    model_path = None
    if job.training.enabled and not skip_training:
        model_path = train_model(job)
    return pairs, model_path
