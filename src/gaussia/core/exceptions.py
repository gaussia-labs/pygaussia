"""Custom exceptions for Gaussia."""


class GaussiaError(Exception):
    """Base exception for Gaussia."""


class RetrieverError(GaussiaError):
    """Exception raised when a retriever fails to load data."""


class MetricError(GaussiaError):
    """Exception raised when a metric calculation fails."""


class GuardianError(GaussiaError):
    """Exception raised when a guardian fails to detect bias."""


class LoaderError(GaussiaError):
    """Exception raised when a loader fails to load data."""


class StatisticalModeError(GaussiaError):
    """Exception raised when a statistical mode calculation fails."""


class LogprobsNotSupportedError(GaussiaError):
    """Raised when the configured LLM provider does not expose logprobs."""


class LogprobsExtractionError(GaussiaError):
    """Raised when expected tokens are absent from the model's top_logprobs."""
