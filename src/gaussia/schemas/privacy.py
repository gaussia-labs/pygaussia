"""Schemas for the Domain-Adjusted Privacy Detection metric.

Field names mirror the sandbox ``chatbot_v3_results.json`` keys so that
``model_dump(mode="json")`` feeds ``panel_chatbot_v3.py`` without changes
(FR-014). The redundant ``*_100`` values and the qualitative ``interpretation``
are recomputed from the stored raw values inside the model validator on every
construction, so they cannot drift out of sync.
"""

from collections.abc import Mapping
from itertools import pairwise
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PlainSerializer, model_validator

from .common import Batch
from .metrics import BaseMetric

FrozenWeights = Annotated[
    Mapping[str, float],
    AfterValidator(lambda weights: MappingProxyType(dict(weights))),
    PlainSerializer(dict, return_type=dict),
]

Interpretation = Literal[
    "Not suitable",
    "Baseline / complement only",
    "Suitable with mitigations",
    "Operationally suitable",
    "Recommended",
    "Recommended with strong local evidence",
]

_TOLERANCE = 1e-9


def interpretation_for(score_100: float) -> Interpretation:
    """Map a 0-100 score to the paper's qualitative band (Table 3)."""
    if score_100 < 40.0:
        return "Not suitable"
    if score_100 < 60.0:
        return "Baseline / complement only"
    if score_100 < 75.0:
        return "Suitable with mitigations"
    if score_100 < 85.0:
        return "Operationally suitable"
    if score_100 < 95.0:
        return "Recommended"
    return "Recommended with strong local evidence"


class Span(BaseModel):
    """A region of text carrying a PII class label and character offsets ``[start, end)``."""

    label: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_offsets(self) -> "Span":
        if self.start >= self.end:
            raise ValueError("span start must be strictly less than end")
        return self


class PrivacyBatch(Batch):
    """A ``Batch`` whose ``query`` is the text shown to the detector, annotated with ground-truth spans."""

    spans: list[Span] = Field(default_factory=list)


class PrivacyDomainConfig(BaseModel):
    """Immutable description of the evaluation domain shared by every detector in a run."""

    model_config = ConfigDict(frozen=True)

    classes: frozenset[str] = Field(min_length=1)
    criticality_weights: FrozenWeights
    fn_severity_weights: FrozenWeights
    iou_threshold: float = Field(default=0.50, gt=0.0, le=1.0)
    regulatory_framework: str | None = None

    @model_validator(mode="after")
    def _check_weights(self) -> "PrivacyDomainConfig":
        for field_name, weights in (
            ("criticality_weights", self.criticality_weights),
            ("fn_severity_weights", self.fn_severity_weights),
        ):
            if set(weights) != set(self.classes):
                raise ValueError(f"{field_name} keys must equal the domain classes exactly")
            if any(not 0.0 <= value <= 1.0 for value in weights.values()):
                raise ValueError(f"{field_name} values must each lie in [0, 1]")
            if abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"{field_name} must sum to 1.0")
        return self


class ClassMetrics(BaseModel):
    """Per-class confusion counts and the rates derived from them."""

    model_config = ConfigDict(frozen=True)

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f2: float = Field(ge=0.0, le=1.0)
    fn_rate: float = Field(ge=0.0, le=1.0)
    criticality_weight: float = Field(ge=0.0, le=1.0)
    fn_severity_weight: float = Field(ge=0.0, le=1.0)


class DetectionScoreContribution(BaseModel):
    """Per-class term ``w_c · f2_c`` of ``DetectionScore``."""

    model_config = ConfigDict(frozen=True)

    weight: float = Field(ge=0.0, le=1.0)
    f2: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_product(self) -> "DetectionScoreContribution":
        if abs(self.contribution - self.weight * self.f2) > _TOLERANCE:
            raise ValueError("contribution must equal weight * f2")
        return self


class CriticalFNContribution(BaseModel):
    """Per-class term ``severity_c * fn_rate_c`` of ``CriticalFN``."""

    model_config = ConfigDict(frozen=True)

    severity_weight: float = Field(ge=0.0, le=1.0)
    fn_rate: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_product(self) -> "CriticalFNContribution":
        if abs(self.contribution - self.severity_weight * self.fn_rate) > _TOLERANCE:
            raise ValueError("contribution must equal severity_weight * fn_rate")
        return self


class PrivacyMetric(BaseMetric):
    """Per-detector evaluation result containing every component of the paper's output contract."""

    name: str
    success: bool = True
    error: str | None = None

    load_time: float = Field(default=0.0, ge=0.0)
    inference_latency: float = Field(default=0.0, ge=0.0)

    corpus_samples: int = Field(default=0, ge=0)
    ground_truth_count: int = Field(default=0, ge=0)
    raw_prediction_count: int = Field(default=0, ge=0)
    prediction_count: int = Field(default=0, ge=0)
    out_of_domain_filtered: int = Field(default=0, ge=0)

    class_metrics: dict[str, ClassMetrics] = Field(default_factory=dict)
    detection_score: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_score_contributions: dict[str, DetectionScoreContribution] = Field(default_factory=dict)

    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    supported_classes: list[str] = Field(default_factory=list)
    covered_domain_classes: list[str] = Field(default_factory=list)
    missing_domain_classes: list[str] = Field(default_factory=list)

    domain_fit: float = Field(default=0.0, ge=0.0, le=1.0)
    regulatory_fit: float = Field(default=0.0, ge=0.0, le=1.0)

    critical_fn: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty_fn: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_fn_contributions: dict[str, CriticalFNContribution] = Field(default_factory=dict)
    r1_weakest_class_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    r1_weakest_class: str | None = None
    r2_systemic_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    r_final: float = Field(default=0.0, ge=0.0, le=1.0)

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_100: float = Field(default=0.0, ge=0.0, le=100.0)

    interpretation: Interpretation = "Not suitable"
    r1_weakest_class_risk_100: float = Field(default=0.0, ge=0.0, le=100.0)
    r2_systemic_risk_100: float = Field(default=0.0, ge=0.0, le=100.0)
    r_final_100: float = Field(default=0.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "PrivacyMetric":
        self.interpretation = interpretation_for(self.score_100)
        self.r1_weakest_class_risk_100 = self.r1_weakest_class_risk * 100.0
        self.r2_systemic_risk_100 = self.r2_systemic_risk * 100.0
        self.r_final_100 = self.r_final * 100.0
        if self.success is False:
            return self
        if abs(self.score_100 - self.score * 100.0) > 1e-6:
            raise ValueError("score_100 must equal score * 100")
        expected = self.detection_score * self.coverage * self.domain_fit * self.regulatory_fit * self.penalty_fn
        if abs(self.score - expected) > _TOLERANCE:
            raise ValueError("score must equal the product of its five components")
        expected_r_final = 1.0 - (1.0 - self.r1_weakest_class_risk) * (1.0 - self.r2_systemic_risk)
        if abs(self.r_final - expected_r_final) > _TOLERANCE:
            raise ValueError("r_final must equal 1 - (1 - r1) * (1 - r2)")
        domain_classes = set(self.covered_domain_classes) | set(self.missing_domain_classes)
        if set(self.class_metrics) != domain_classes:
            raise ValueError("class_metrics keys must equal the domain taxonomy")
        return self


class PrivacyRanking(BaseMetric):
    """Ordered per-detector results plus the winning detector identifier."""

    results: list[PrivacyMetric]
    winning_detector: str | None = None
    iou_threshold: float = Field(gt=0.0, le=1.0)
    domain_classes: list[str] = Field(default_factory=list)
    regulatory_framework: str | None = None

    @model_validator(mode="after")
    def _check_ordering(self) -> "PrivacyRanking":
        successful = [r for r in self.results if r.success]
        seen_failure = False
        for result in self.results:
            if result.success:
                if seen_failure:
                    raise ValueError("successful entries must precede failed entries")
            else:
                seen_failure = True
        for previous, current in pairwise(successful):
            if previous.score_100 < current.score_100:
                raise ValueError("successful results must be ordered by score_100 descending")
        expected_winner = self.results[0].name if self.results and self.results[0].success else None
        if self.winning_detector != expected_winner:
            raise ValueError("winning_detector must match the top-ranked successful detector")
        return self


__all__ = [
    "ClassMetrics",
    "CriticalFNContribution",
    "DetectionScoreContribution",
    "Interpretation",
    "PrivacyBatch",
    "PrivacyDomainConfig",
    "PrivacyMetric",
    "PrivacyRanking",
    "Span",
    "interpretation_for",
]
