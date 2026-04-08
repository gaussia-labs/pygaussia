"""Conversational metric schemas."""

from pydantic import BaseModel

from .metrics import BaseMetric


class ConversationalScore(BaseModel):
    mean: float
    ci_low: float | None = None
    ci_high: float | None = None


class ConversationalInteraction(BaseModel):
    qa_id: str
    memory: float
    language: float
    quality_maxim: float
    quantity_maxim: float
    relation_maxim: float
    manner_maxim: float
    sensibleness: float


class ConversationalMetric(BaseMetric):
    """
    Session-level conversational metric aggregating Grice's maxim scores across all interactions.
    """

    n_interactions: int
    conversational_memory: ConversationalScore
    conversational_language: ConversationalScore
    conversational_quality_maxim: ConversationalScore
    conversational_quantity_maxim: ConversationalScore
    conversational_relation_maxim: ConversationalScore
    conversational_manner_maxim: ConversationalScore
    conversational_sensibleness: ConversationalScore
    interactions: list[ConversationalInteraction]
