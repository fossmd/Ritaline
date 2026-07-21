"""Command-line interface for Ritaline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_endpoint_config, load_job_config
from .documents import load_document
from .exceptions import RitalineError
from .generation import QAGenerator
from .pipeline import train_model
from .templates import ENDPOINT_YAML, JOB_YAML

app = typer.Typer(
    name="ritaline",
    help="Generate grounded Q&A datasets from PDF/TXT files and fine-tune open-weight LLMs.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


def _abort(exc: Exception) -> None:
    console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(code=1) from exc


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed Ritaline version."),
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command("init")
def init_config(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory in which to create endpoint.yaml and job.yaml."),
    ] = Path("ritaline-config"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing template files."),
    ] = False,
) -> None:
    """Create editable starter configuration files."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        directory / "endpoint.yaml": ENDPOINT_YAML,
        directory / "job.yaml": JOB_YAML,
    }
    for path, content in files.items():
        if path.exists() and not force:
            console.print(f"[yellow]Skipped existing file:[/yellow] {path}")
            continue
        path.write_text(content, encoding="utf-8")
        console.print(f"[green]Created:[/green] {path}")


@app.command()
def validate(
    endpoint_file: Annotated[
        Path,
        typer.Option("--endpoint", "-e", exists=True, readable=True),
    ],
    job_file: Annotated[
        Path,
        typer.Option("--job", "-j", exists=True, readable=True),
    ],
) -> None:
    """Validate both YAML configuration files without making API calls."""
    try:
        endpoint = load_endpoint_config(endpoint_file)
        job = load_job_config(job_file)
    except RitalineError as exc:
        _abort(exc)
        return

    table = Table(title="Ritaline configuration")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Generation endpoint model", endpoint.model)
    table.add_row("Q&A pairs", str(job.generation.qa_count))
    table.add_row("Styles", ", ".join(style.name for style in job.generation.styles))
    table.add_row("Training enabled", str(job.training.enabled))
    table.add_row("Trainable model", job.training.model_name_or_path)
    table.add_row("Training method", job.training.method)
    console.print(table)


@app.command()
def preview(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    endpoint_file: Annotated[
        Path,
        typer.Option("--endpoint", "-e", exists=True, readable=True),
    ],
    job_file: Annotated[
        Path,
        typer.Option("--job", "-j", exists=True, readable=True),
    ],
    count: Annotated[int, typer.Option("--count", "-n", min=1, max=20)] = 3,
) -> None:
    """Print prompts without contacting the LLM endpoint."""
    try:
        endpoint = load_endpoint_config(endpoint_file)
        job = load_job_config(job_file)
        document = load_document(input_file)
        previews = QAGenerator(endpoint, job.generation).preview_prompts(document, count=count)
        console.print_json(json.dumps(previews, ensure_ascii=False))
    except RitalineError as exc:
        _abort(exc)


@app.command()
def generate(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    endpoint_file: Annotated[
        Path,
        typer.Option("--endpoint", "-e", exists=True, readable=True),
    ],
    job_file: Annotated[
        Path,
        typer.Option("--job", "-j", exists=True, readable=True),
    ],
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume compatible partial JSONL output."),
    ] = True,
) -> None:
    """Generate Q&A data but do not fine-tune a model."""
    try:
        endpoint = load_endpoint_config(endpoint_file)
        job = load_job_config(job_file)
        document = load_document(input_file)
        generator = QAGenerator(endpoint, job.generation)
        pairs = asyncio.run(generator.generate(document, resume=resume))
        console.print(
            f"[green]Generated {len(pairs)} pairs.[/green] Raw: {job.generation.output_path}; "
            f"training export: {job.generation.training_dataset_path}"
        )
    except RitalineError as exc:
        _abort(exc)


@app.command()
def train(
    job_file: Annotated[
        Path,
        typer.Option("--job", "-j", exists=True, readable=True),
    ],
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", "-d", exists=True, readable=True),
    ] = None,
) -> None:
    """Fine-tune the configured model from an existing raw Q&A JSONL file."""
    try:
        job = load_job_config(job_file)
        output = train_model(job, dataset_path=dataset)
        console.print(f"[green]Model/adapters saved to:[/green] {output}")
    except RitalineError as exc:
        _abort(exc)


@app.command("run")
def run_all(
    input_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    endpoint_file: Annotated[
        Path,
        typer.Option("--endpoint", "-e", exists=True, readable=True),
    ],
    job_file: Annotated[
        Path,
        typer.Option("--job", "-j", exists=True, readable=True),
    ],
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume"),
    ] = True,
    skip_training: Annotated[
        bool,
        typer.Option("--skip-training", help="Generate data without starting fine-tuning."),
    ] = False,
) -> None:
    """Generate the exact Q&A count, then fine-tune the configured model."""
    try:
        endpoint = load_endpoint_config(endpoint_file)
        job = load_job_config(job_file)
        document = load_document(input_file)
        generator = QAGenerator(endpoint, job.generation)
        pairs = asyncio.run(generator.generate(document, resume=resume))
        console.print(f"[green]Generated {len(pairs)} Q&A pairs.[/green]")
        if job.training.enabled and not skip_training:
            output = train_model(job)
            console.print(f"[green]Model/adapters saved to:[/green] {output}")
        else:
            console.print("[yellow]Training skipped.[/yellow]")
    except RitalineError as exc:
        _abort(exc)


if __name__ == "__main__":
    app()
