import pytest
from pydantic import ValidationError

from ritaline.config import GenerationConfig, QAStyle


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(
            qa_count=2,
            styles=[QAStyle(name="factual", instruction="Ask a fact")],
            chunk_size_chars=500,
            chunk_overlap_chars=500,
        )


def test_duplicate_style_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(
            qa_count=2,
            styles=[
                QAStyle(name="Factual", instruction="Ask a fact"),
                QAStyle(name="factual", instruction="Ask another fact"),
            ],
        )


def test_endpoint_extra_body_cannot_replace_messages() -> None:
    from ritaline.config import EndpointConfig

    with pytest.raises(ValidationError):
        EndpointConfig(
            base_url="https://example.invalid",
            model="model",
            api_key_env=None,
            extra_body={"messages": []},
        )
