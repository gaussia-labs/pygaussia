"""Toxicity metric schemas."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class SentimentScore(BaseModel):
    """Sentiment analysis result for individual text.

    Used in ASB (Associated Sentiment Bias) calculation.
    The metric will aggregate individual scores to compute:
    - S_i: average sentiment for group g_i
    - S̄: global average sentiment across all groups
    - ASB = (1/n) Σ |S_i - S̄|
    """

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score in range [-1, 1] where -1 is most negative, 0 is neutral, and 1 is most positive",
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence/probability of the sentiment prediction"
    )
    label: str | None = Field(None, description="Categorical sentiment label (e.g., 'positive', 'negative', 'neutral')")


class ToxicityDataset(BaseModel):
    """A toxicity word entry."""

    word: str
    category: str | None = None


class GroupDetection(BaseModel):
    """Result for a single group detection."""

    present: bool
    score: float
    best_prototype: str
    best_prototype_index: int


class GroupProfilingFrequentist(BaseModel):
    """Frequentist statistics for group profiling."""

    DR: float
    ASB: float
    DTO: float
    DIDT: float


class BayesianSummary(BaseModel):
    """Bayesian posterior summary."""

    mean: float
    ci_low: float
    ci_high: float


class GroupProfilingBayesian(BaseModel):
    """Bayesian statistics for group profiling."""

    priors: float
    mc_samples: int
    ci_level: float
    q_i: dict[str, float] | None = None
    summary: dict[str, BayesianSummary]  # keys: DR/ASB/DTO/DIDT


class GroupProfiling(BaseModel):
    """Group profiling results for toxicity analysis."""

    MetricMode: ClassVar = Literal["frequentist", "bayesian"]
    mode: Literal["frequentist", "bayesian"]
    weights: dict[str, float]  # w_DR, w_ASB, w_DTO
    groups: list[str]

    N_i: dict[str, int]
    K_i: dict[str, int]
    p_i: dict[str, float]
    q_i: dict[str, float]
    T_i: dict[str, float]

    frequentist: GroupProfilingFrequentist | None = None
    bayesian: GroupProfilingBayesian | None = None

    toxicity_threshold: float = 0.0


class ToxicityMetric(BaseMetric):
    """
    Toxicity metric for evaluating toxic language in assistant responses.
    """

    class AssistantSpace(BaseModel):
        latent_space: list
        embeddings: list
        cluster_labels: list

    cluster_profiling: dict[float, float]
    group_profiling: GroupProfiling | None = None
    assistant_space: AssistantSpace
