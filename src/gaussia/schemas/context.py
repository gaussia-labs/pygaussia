"""Context metric schemas."""

from pydantic import BaseModel

from .metrics import BaseMetric


class ContextInteraction(BaseModel):
    qa_id: str
    context_awareness: float


class ContextMetric(BaseMetric):
    """
    Session-level context metric aggregating context awareness scores across all interactions.
    """

    n_interactions: int
    context_awareness: float
    context_awareness_ci_low: float | None = None
    context_awareness_ci_high: float | None = None
    interactions: list[ContextInteraction]
