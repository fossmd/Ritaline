"""YAML configuration models and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ConfigurationError

T = TypeVar("T", bound=BaseModel)


DEFAULT_SYSTEM_PROMPT = """You generate high-quality training data from supplied source text.
Every answer must be fully supported by the source. Do not use outside knowledge.
Questions must be clear, self-contained, and useful for supervised fine-tuning.
If the source cannot support a valid pair, say so using the requested JSON schema."""

DEFAULT_USER_TEMPLATE = """Create exactly one Q&A pair from the source excerpt.

Q&A style: {style_name}
Style instructions: {style_instruction}
Source file: {source_name}
Source pages: {page_range}

Requirements:
- The answer must be directly supported by the excerpt.
- Do not mention the excerpt, chunk, page number, or these instructions.
- Do not invent names, dates, figures, causes, or conclusions.
- Return only one JSON object with string fields \"question\" and \"answer\".
- If no valid pair can be produced, return {{\"question\": \"\", \"answer\": \"\"}}.

SOURCE EXCERPT
---
{chunk}
---
"""


class EndpointConfig(BaseModel):
    """Configuration for an OpenAI-compatible Chat Completions endpoint."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = "RITALINE_API_KEY"
    chat_completions_path: str = "/chat/completions"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=20)
    retry_backoff_seconds: float = Field(default=1.5, gt=0)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=800, ge=1)
    response_format: Literal["json_object", "none"] = "json_object"
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("chat_completions_path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"

    @model_validator(mode="after")
    def protect_request_fields(self) -> EndpointConfig:
        reserved = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "response_format",
            "stream",
        }
        conflicts = sorted(reserved.intersection(self.extra_body))
        if conflicts:
            raise ValueError(
                "extra_body cannot override managed request fields: " + ", ".join(conflicts)
            )
        return self


class QAStyle(BaseModel):
    """One named Q&A style and its prompt instruction."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    instruction: str = Field(min_length=1)


class GenerationConfig(BaseModel):
    """Dataset-generation settings."""

    model_config = ConfigDict(extra="forbid")

    qa_count: int = Field(gt=0)
    styles: list[QAStyle] = Field(min_length=1)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_TEMPLATE
    chunk_size_chars: int = Field(default=7000, ge=500)
    chunk_overlap_chars: int = Field(default=700, ge=0)
    min_chunk_chars: int = Field(default=300, ge=1)
    deduplicate_questions: bool = True
    max_attempts_per_pair: int = Field(default=6, ge=1, le=50)
    seed: int = 42
    output_path: Path = Path("outputs/qa_pairs.jsonl")
    training_dataset_path: Path = Path("outputs/training_dataset.jsonl")

    @model_validator(mode="after")
    def validate_chunking(self) -> GenerationConfig:
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        names = [style.name.casefold() for style in self.styles]
        if len(names) != len(set(names)):
            raise ValueError("Q&A style names must be unique")
        required_fields = {
            "{style_name}",
            "{style_instruction}",
            "{source_name}",
            "{page_range}",
            "{chunk}",
        }
        missing = [field for field in required_fields if field not in self.user_prompt_template]
        if missing:
            raise ValueError(
                "user_prompt_template is missing required placeholders: " + ", ".join(missing)
            )
        return self


class TrainingConfig(BaseModel):
    """Hugging Face TRL/PEFT supervised fine-tuning settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_name_or_path: str = Field(min_length=1)
    output_dir: Path = Path("outputs/model")
    method: Literal["lora", "qlora", "full"] = "lora"
    trust_remote_code: bool = False
    max_seq_length: int = Field(default=2048, ge=128)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = -1
    learning_rate: float = Field(default=2e-4, gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    weight_decay: float = Field(default=0.0, ge=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    eval_ratio: float = Field(default=0.05, ge=0, lt=0.5)
    gradient_checkpointing: bool = True
    packing: bool = False
    seed: int = 42
    bf16: bool | None = None
    fp16: bool | None = None
    report_to: list[str] = Field(default_factory=list)
    resume_from_checkpoint: str | None = None
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: list[str] | Literal["all-linear"] = "all-linear"
    qlora_compute_dtype: Literal["auto", "bfloat16", "float16"] = "auto"

    @model_validator(mode="after")
    def validate_precision(self) -> TrainingConfig:
        if self.bf16 is True and self.fp16 is True:
            raise ValueError("bf16 and fp16 cannot both be true")
        return self


class JobConfig(BaseModel):
    """Top-level generation and training configuration."""

    model_config = ConfigDict(extra="forbid")

    generation: GenerationConfig
    training: TrainingConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read YAML configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration must contain a YAML mapping: {path}")
    return data


def load_config(path: str | Path, model_type: type[T]) -> T:
    """Load and validate a YAML file as the requested Pydantic model."""
    config_path = Path(path)
    try:
        return model_type.model_validate(_load_yaml(config_path))
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid configuration in {config_path}: {exc}") from exc


def load_endpoint_config(path: str | Path) -> EndpointConfig:
    return load_config(path, EndpointConfig)


def load_job_config(path: str | Path) -> JobConfig:
    return load_config(path, JobConfig)
