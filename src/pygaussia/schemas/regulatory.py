"""Regulatory compliance metric schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class RegulatoryChunk(BaseModel):
    """
    A chunk of regulatory text with retrieval and reranking scores.

    Attributes:
        text: The chunk text content.
        source: Source document filename.
        chunk_index: Index of this chunk within the source document.
        similarity: Cosine similarity score from embedding retrieval.
        reranker_score: Score from reranker model (higher = supports, lower = contradicts).
        verdict: Whether the chunk SUPPORTS or CONTRADICTS the agent response.
    """

    text: str
    source: str
    chunk_index: int
    similarity: float = Field(ge=0, le=1)
    reranker_score: float = Field(ge=0, le=1)
    verdict: Literal["SUPPORTS", "CONTRADICTS"]


class RegulatoryInteraction(BaseModel):
    """Per-interaction regulatory evaluation result embedded in the session-level metric."""

    qa_id: str
    query: str
    assistant: str
    compliance_score: float = Field(ge=0, le=1)
    verdict: Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"]
    supporting_chunks: int = Field(ge=0)
    contradicting_chunks: int = Field(ge=0)
    retrieved_chunks: list[RegulatoryChunk]
    insight: str


class RegulatoryMetric(BaseMetric):
    """
    Session-level regulatory compliance metric aggregating compliance scores
    across all interactions.

    Attributes:
        n_interactions: Number of interactions evaluated in this session.
        compliance_score: Weighted mean compliance score across all interactions (0.0-1.0).
        compliance_score_ci_low: Lower credible bound — only set in Bayesian mode.
        compliance_score_ci_high: Upper credible bound — only set in Bayesian mode.
        verdict: Overall session verdict derived from the aggregated compliance score.
        total_supporting_chunks: Sum of supporting chunks across all interactions.
        total_contradicting_chunks: Sum of contradicting chunks across all interactions.
        interactions: Per-interaction evaluation details.
    """

    n_interactions: int
    compliance_score: float = Field(ge=0, le=1)
    compliance_score_ci_low: float | None = None
    compliance_score_ci_high: float | None = None
    verdict: Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"]
    total_supporting_chunks: int = Field(ge=0)
    total_contradicting_chunks: int = Field(ge=0)
    interactions: list[RegulatoryInteraction]


__all__ = ["RegulatoryChunk", "RegulatoryInteraction", "RegulatoryMetric"]
