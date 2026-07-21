"""Ritaline public package API."""

from .config import EndpointConfig, GenerationConfig, JobConfig, QAStyle, TrainingConfig
from .documents import load_document
from .generation import QAGenerator
from .pipeline import generate_dataset, run_pipeline, train_model

__all__ = [
    "EndpointConfig",
    "GenerationConfig",
    "JobConfig",
    "QAGenerator",
    "QAStyle",
    "TrainingConfig",
    "generate_dataset",
    "load_document",
    "run_pipeline",
    "train_model",
]

__version__ = "0.1.0"
