"""Custom exceptions raised by Ritaline."""


class RitalineError(Exception):
    """Base class for all package-specific errors."""


class ConfigurationError(RitalineError):
    """Raised when a configuration file is missing or invalid."""


class DocumentError(RitalineError):
    """Raised when a document cannot be read or contains no usable text."""


class LLMError(RitalineError):
    """Raised when an LLM endpoint request fails."""


class GenerationError(RitalineError):
    """Raised when the requested Q&A dataset cannot be completed."""


class TrainingError(RitalineError):
    """Raised when local fine-tuning cannot be started or completed."""
