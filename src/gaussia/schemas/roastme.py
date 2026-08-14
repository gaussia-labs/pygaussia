"""Schemas for Roast Me — profile-then-exploit adversarial evaluation.

This is the first module in ``schemas/`` that imports from ``core/`` at runtime, and the
direction is deliberate: ``Principle.grader`` is typed ``Grader``, Pydantic resolves that
annotation when it builds the model, so the real class has to be present rather than a
type-checking-only name. There is no cycle, because the ten ``core/`` interfaces reference
these models under ``TYPE_CHECKING`` only.

The framework's ``Batch`` appears here for ``RoastBatch`` alone — the output boundary where
Roast Me hands its records to the metric pipeline (FR-034).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaussia.core.grader import Grader

from .common import Batch

_WEIGHT_TOLERANCE = 1e-9


class Principle(BaseModel):
    """One rule of the behavioral contract, with the grader bound to it.

    ``arbitrary_types_allowed`` is needed because ``grader`` is an ABC instance rather than
    a model. FR-003 needs no validator: ``grader`` is required, so a principle with none
    cannot be constructed, and therefore cannot reach a contract to contribute a silent zero
    to ``v``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    rubric: str = Field(min_length=1)
    grader: Grader


class BehavioralContract(BaseModel):
    """The set ``Pi``: what the assistant is being held to, and how severely.

    Gaussia ships no instance (FR-002). The paper defines ``Pi`` as an input to the method,
    so a default would be the library deciding what counts as a failure.
    """

    principles: list[Principle] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_contract(self) -> "BehavioralContract":
        total = sum(principle.weight for principle in self.principles)
        # The tolerance is explicit so a contract assembled from decimals is not rejected
        # for float noise.
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError("principle weights must sum to 1.0")
        identifiers = [principle.id for principle in self.principles]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("principle identifiers must be unique")
        return self


class GraderConfig(BaseModel):
    """Configuration for the shipped logprob grader. Every value is the user's (FR-006)."""

    positive_tokens: tuple[str, ...] = Field(min_length=1)
    negative_tokens: tuple[str, ...] = Field(min_length=1)
    reasoning_budget: int = Field(ge=1)
    fallback_samples: int = Field(ge=1)
    top_logprobs: int = Field(ge=1)


class ExploiterConfig(BaseModel):
    """The method parameters of the category search.

    Split by what each value *is*, not by whether the paper names it. ``tau`` and ``eta``
    say how badly the assistant has to behave before it counts, which is the user's
    judgement about their own domain. ``lambda_`` and ``queries_per_category`` are
    statistical convention. ``pool_size`` is a knob of an implementation gaussia writes.
    ``kappa`` and ``delta`` are compared against numbers a substitutable component produces,
    so their meaning travels with that implementation and they are resolved from it once, at
    Exploiter construction (FR-040, FR-041).
    """

    tau: float = Field(ge=0.0, le=1.0)
    eta: float = Field(ge=0.0, le=1.0)
    lambda_: float = Field(default=1.0, ge=0.0)
    queries_per_category: int = Field(default=10, ge=2)
    pool_size: int = Field(default=20, ge=1)
    kappa: float | None = None
    delta: float | None = Field(default=None, ge=0.0)


class Document(BaseModel):
    """One unit of the knowledge base handed to the Probe Library."""

    id: str = Field(min_length=1)
    content: str
    structured: bool
    kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginSpec(BaseModel):
    """A risk family, mapping to one principle of the contract."""

    id: str = Field(min_length=1)
    name: str
    description: str
    principle: str = Field(min_length=1)


class StrategySpec(BaseModel):
    """An interaction pattern: which entity it operates on, how it transforms it, and whether the hook is documented.

    ``transform`` stays a string because configuration is data. The registry resolves it to
    a ``Transform`` once, during catalogue validation, and nothing branches on it afterwards;
    membership of the registry is checked there rather than here, since ``schemas/`` may not
    depend on the module that owns the registry.

    ``description`` is load-bearing rather than documentation: its clauses become the probe's
    attributes and from there the prose descriptor of the weakness map, which is the only thing
    about a strategy allowed to cross to the Exploiter (FR-013). An empty one leaves nothing
    sayable, so it is rejected here rather than at the point of profiling — a catalogue that
    validates and then fails mid-run is the failure mode FR-025 exists to prevent.
    """

    id: str = Field(min_length=1)
    name: str
    description: str = Field(min_length=1)
    plugin: str | None = None
    entity_kind: str = Field(min_length=1)
    transform: str = Field(min_length=1)
    doc: int = Field(ge=0, le=1)
    phrasing_hint: str


class Catalogue(BaseModel):
    """The user's plugins and strategies.

    Dangling references, unknown transforms, duplicate identifiers and entity kinds no
    configured engine handles are rejected by the catalogue validator, which is the only
    place that holds the contract and the engines needed to decide them (FR-025).
    """

    plugins: list[PluginSpec] = Field(min_length=1)
    strategies: list[StrategySpec] = Field(min_length=1)


class KnowledgeHook(BaseModel):
    """The structured provenance of a probe against the knowledge base."""

    kind: str = Field(min_length=1)
    references: str = Field(min_length=1)
    doc: int = Field(ge=0, le=1)
    how: str
    base_entity: str | None = None
    principle: str | None = None
    verified: bool | None = None
    absence_reliable: bool = True


class Probe(BaseModel):
    """The unit the Profiler consumes."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    hook: KnowledgeHook | None = None
    plugin: str | None = None
    strategy: str = Field(min_length=1)
    attrs: list[str] = Field(default_factory=list)
    engine: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_control_consistency(self) -> "Probe":
        # Control-ness would otherwise be encoded in two places that can silently disagree:
        # a probe with no plugin but a principle set would be excluded from every rate while
        # still charging that principle (FR-011). A hookless probe leans on no entity at all,
        # so there is no second place to disagree with and `plugin` stands alone (FR-026).
        if self.hook is not None and (self.plugin is None) != (self.hook.principle is None):
            raise ValueError("probe plugin and hook principle must both be set or both be unset")
        return self


class TargetResponse(BaseModel):
    """What the user's target adapter returns. Gaussia consumes nothing else about the assistant."""

    content: str
    failed: bool = False
    failure_reason: str | None = None
    session_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_content(self) -> "TargetResponse":
        # An adapter returning nothing without saying why would otherwise produce a graded
        # outcome over an empty string, the silent failure FR-016 exists to prevent.
        if not self.content and not self.failed:
            raise ValueError("empty content is legitimate only on an exchange marked failed")
        return self


class PrincipleGrade(BaseModel):
    """One grader's estimate for one principle on one response.

    ``grader``, ``method`` and ``model`` are the three things FR-005 requires every grade to
    record, and none of them stands in for another: which implementation produced the verdict, how
    it read it, and which model it read it from. One grader reaches its verdict by two methods —
    the logprob path and the sampling fallback — and a rule-based grader has no model at all, so
    neither field identifies the grader. It is required rather than defaulted for the same reason:
    nothing legitimately produces a grade anonymously. Same convention as ``Probe.engine`` and
    ``FailureReport.components``: recorded for reading, never branched on.
    """

    principle: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    grader: str = Field(min_length=1)
    method: str = Field(min_length=1)
    model: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class GradedOutcome(BaseModel):
    """One probe's exchange, graded.

    ``violation`` is ``None`` and only ``None`` for an ungraded outcome (FR-016): every
    aggregate filters on it, so a failed exchange moves neither numerator nor denominator. A
    ``0.0`` would have meant "graded, no violation".
    """

    probe_id: str = Field(min_length=1)
    response: str
    grades: list[PrincipleGrade] = Field(default_factory=list)
    violation: float | None = Field(ge=0.0, le=1.0)
    evidence_available: bool = False
    scoreable: bool = True


class WeaknessEntry(BaseModel):
    """One ``(principle, descriptor)`` entry of the weakness map ``omega``.

    ``descriptor`` is prose, not the strategy identifier: paper invariant 3 forbids sending
    identifiers across, so aggregation is keyed internally by the strategy ``id`` and only
    the prose lands on the profile. ``source_strategy`` exists for auditing and is stripped
    before the profile crosses (FR-013).
    """

    principle: str = Field(min_length=1)
    descriptor: str = Field(min_length=1)
    rate: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=0)
    standard_error: float = Field(ge=0.0)
    source_strategy: str | None = None


class AssistantProfile(BaseModel):
    """The profile ``theta``: the weakness map and the retained hooks. The only artifact the Exploiter receives."""

    weaknesses: list[WeaknessEntry] = Field(default_factory=list)
    hooks: list[KnowledgeHook] = Field(default_factory=list)


class ProfilerResult(BaseModel):
    """What the Profiler returns. A plain result object: Roast Me does not emit through the metric pipeline."""

    profile: AssistantProfile
    outcomes: list[GradedOutcome]
    overall_rate: float = Field(ge=0.0, le=1.0)
    n_scoreable: int = Field(ge=0)
    n_ungraded: int = Field(ge=0)


class FindingKind(StrEnum):
    """How the interpretation classifies one piece of run evidence."""

    BEHAVIORAL = "behavioral_finding"
    GRADER_FALSE_POSITIVE = "grader_false_positive"
    CONTROL = "control_finding"
    TRANSPORT = "transport_finding"
    NEEDS_REVIEW = "needs_human_review"


class GraderAssessment(StrEnum):
    """Whether the finding supports the grader and its principle attribution."""

    SUPPORTS = "supports_grader"
    FALSE_POSITIVE = "grader_false_positive"
    TAXONOMY_MISMATCH = "taxonomy_mismatch"
    NOT_APPLICABLE = "not_applicable"


class InterpretedFinding(BaseModel):
    """One model interpretation anchored to a concrete probe."""

    model_config = ConfigDict(extra="forbid")

    probe_id: str = Field(min_length=1)
    kind: FindingKind
    grader_assessment: GraderAssessment
    principle: str | None
    title: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    analysis: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class FindingsInterpretation(BaseModel):
    """The model-owned part of a findings report."""

    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(min_length=1)
    findings: list[InterpretedFinding]
    limitations: list[str]


class FindingsReport(BaseModel):
    """Auditable envelope around an LLM interpretation.

    The run score remains the Profiler's result. This artifact records which model interpreted
    that evidence, whether it is the same model as the target, and the structured interpretation
    used to render the human-readable report.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    generated_at: datetime
    reported_overall_rate: float = Field(ge=0.0, le=1.0)
    same_model_as_target: bool
    interpretation: FindingsInterpretation
    usage: dict[str, int] = Field(default_factory=dict)


class Category(BaseModel):
    """An ordered conjunction of natural-language attributes, each traceable to what induced it."""

    attributes: list[str] = Field(min_length=1)
    provenance: list[str]

    @model_validator(mode="after")
    def _check_provenance(self) -> "Category":
        if len(self.provenance) != len(self.attributes):
            raise ValueError("provenance must carry one entry per attribute")
        return self


class CategoryEvaluation(BaseModel):
    """One category, evaluated: its queries, their responses, the grades behind each violation and the score.

    The per-query lists are parallel to ``queries``. A misalignment would corrupt ``S(c)``
    silently, and would leave a violation with no rationale behind it — which FR-036 makes
    inadmissible.
    """

    category: Category
    queries: list[str]
    responses: list[str]
    violations: list[float]
    on_profile: list[bool]
    realism_gap: float
    score: float
    n: int = Field(ge=0)
    rationale: list[list[PrincipleGrade]]
    dropped_attributes: list[str]

    @model_validator(mode="after")
    def _check_per_query_lengths(self) -> "CategoryEvaluation":
        expected = len(self.queries)
        lengths = (
            ("responses", len(self.responses)),
            ("violations", len(self.violations)),
            ("on_profile", len(self.on_profile)),
            ("rationale", len(self.rationale)),
        )
        for name, length in lengths:
            if length != expected:
                raise ValueError(f"{name} must carry one entry per query")
        return self


class RoastDatasetRecord(BaseModel):
    """One record of the Roast Dataset: the last shape Roast Me owns before the output boundary.

    ``evidence_available`` travels with the record rather than being inferred from
    ``evidence``: absent evidence and evidence sought and not found are different findings,
    and collapsing them into one ``None`` would erase the distinction US2 requires.
    """

    query: str
    response: str
    violation: float | None
    principles_charged: list[str]
    rationale: list[PrincipleGrade]
    evidence: str | None
    evidence_available: bool = False


class FailureReport(BaseModel):
    """The ranked categories, the individual queries that reached ``tau``, and who produced the numbers.

    ``components`` records which implementation of each substitutable piece ran — the
    search, the query generator, the on-profile filter, the realism estimator — plus the
    ``kappa`` and ``delta`` actually in force and whether each was supplied or recommended
    (FR-039, FR-041). Two of the shipped three are gaussia's own construction rather than the
    paper's, so a weak report has to be attributable to the piece that can be swapped
    instead of to the method. Recorded for reading, never branched on.
    """

    categories: list[CategoryEvaluation]
    queries_over_threshold: list[RoastDatasetRecord]
    components: dict[str, str]


class RoastBatch(Batch):
    """A ``Batch`` carrying the Roast Dataset record for the turn it came from.

    The framework's ``Batch`` has no free-form slot, so the record's own fields need
    somewhere to live that existing metrics can ignore (FR-034). ``ground_truth_assistant``
    is filled with ``""`` by the conversion: a trap has no correct answer, and inventing one
    would let a metric score against it.
    """

    roast: RoastDatasetRecord


__all__ = [
    "AssistantProfile",
    "BehavioralContract",
    "Catalogue",
    "Category",
    "CategoryEvaluation",
    "Document",
    "ExploiterConfig",
    "FailureReport",
    "FindingKind",
    "FindingsInterpretation",
    "FindingsReport",
    "GradedOutcome",
    "GraderAssessment",
    "GraderConfig",
    "InterpretedFinding",
    "KnowledgeHook",
    "PluginSpec",
    "Principle",
    "PrincipleGrade",
    "Probe",
    "ProfilerResult",
    "RoastBatch",
    "RoastDatasetRecord",
    "StrategySpec",
    "TargetResponse",
    "WeaknessEntry",
]
