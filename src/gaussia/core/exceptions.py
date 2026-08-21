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


class BoundaryNotDeclaredError(GaussiaError):
    """Raised when a rule-based grader is asked for a verdict with no boundary to check.

    ``PrincipleGrade.score`` is a required float, so a grader has no way to answer "not
    evaluable". Returning ``0.0`` would claim the response respected a boundary that was
    never declared, which is the one reading that understates risk. Raising instead lets the
    exchange be recorded ungraded (FR-016), where it moves neither side of the rate.
    """


class UnrecognizedGroundTruthKeysError(GaussiaError):
    """Raised when ``ground_truth_agentic`` carries keys a metric would ignore.

    The dict is untyped and read with ``.get()``, so a misspelled key is indistinguishable
    from an absent one: a mistyped ``allowed_tools`` turns the access-boundary check off and
    reports "no boundary declared" for a dataset that declared one. Failing before the first
    judge call is what makes that visible, and cheap.
    """
