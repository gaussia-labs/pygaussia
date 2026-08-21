"""Schemas for Roast Me — profile-then-exploit adversarial evaluation.

This is the first module in ``schemas/`` that imports from ``core/`` at runtime, and the
direction is deliberate: ``Principle.grader`` is typed ``Grader``, Pydantic resolves that
annotation when it builds the model, so the real class has to be present rather than a
type-checking-only name. There is no cycle, because the ten ``core/`` interfaces reference
these models under ``TYPE_CHECKING`` only.

The framework's ``Batch`` appears here for ``RoastBatch`` alone — the output boundary where
Roast Me hands its records to the metric pipeline (FR-034).
"""

from typing import Any

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
    require_logprobs: bool = False
    """Refuse to grade rather than fall back to sampling over ``fallback_samples``.

    Off by default, because degrading is what FR-008 and spec D13 ask for: raising would make the
    violation-rate denominator depend on whether the provider happened to expose logprobs.

    On for a run whose number is going to be compared against another. The two paths are two
    estimators — one reads a continuous probability out of the verdict token's distribution in a
    single call, the other votes across ``k`` samples and can only land on multiples of ``1/k`` —
    so a mean taken across both is a mean over two different measurements. Where that matters more
    than having a number at all, this is how a user says so; there was previously no way to.
    """


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


class EngineDeclaration(BaseModel):
    """Which probe engines ran, and which of them the paper's trade-off tables characterise.

    Both halves are facts the framework holds and the user does not: which engines were composed
    is the library's own execution, and what the paper measured is gaussia's own paper. Neither is
    a judgement about a domain, so neither is the user's to supply — unlike ``tau``, the contract
    or the catalogue, which gaussia ships none of on purpose.

    It exists because the default composition produced no published number. Retrieval, graph and
    multi-hop run by default; the paper's tables cover retrieval, graph and enumeration. The
    default set is therefore one nothing published covers, and it cannot simply be corrected —
    enumeration needs an ``EntityEnumerator``, which is domain knowledge gaussia ships none of by
    decision D14, so the evaluated set is unreachable out of the box by construction. The fix is to
    make the gap visible rather than to close it.

    ``in_paper_tables`` is a fact about the paper and not about the run, so it does not drift with
    a run. It can go stale if the paper changes, which is why ``paper`` records the version it was
    read from. Recorded for reading, never branched on.
    """

    ran: list[str]
    in_paper_tables: list[str]
    outside_paper_tables: list[str]
    paper: str


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

    ``doc`` is the **expected** grounding label and nothing reads it, which is the design rather
    than an omission: invariant 4 has the real label derived by the generating engine from its own
    view of the boundary, never from a declaration and never from the enumeration used for scoring.
    So the two can disagree, and where they do it is the transformation that did not do what the
    catalogue says it does — ``flip_value`` returns an entity carrying no digits unchanged, and the
    engine then labels it documented, quietly turning that strategy into a second control. Worth
    knowing when writing one; ``KnowledgeHook.doc`` is the label a run actually used.
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


class GroundedTwist(BaseModel):
    """One fact of the corpus, the falsehood derived from it, and the query that asserts it.

    What a ``FactTwister`` returns (FR-042). ``entity`` is the thing the fact is *about*, and it
    stays real: the falsehood is the datum, not the name. That is the whole difference from the
    transformations of FR-025, and it is why a twist needs no boundary — nothing here claims an
    absence, so there is nothing to confirm.

    ``pattern`` is echoed back rather than chosen. A twister is asked for one named pattern per
    call, so this field records that the request was honoured, and it lands on
    ``KnowledgeHook.how`` the way a transformation's key does.
    """

    entity: str = Field(min_length=1)
    real_fact: str = Field(min_length=1)
    false_premise: str = Field(min_length=1)
    query: str = Field(min_length=1)
    pattern: str = Field(min_length=1)


class Probe(BaseModel):
    """The unit the Profiler consumes."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    hook: KnowledgeHook | None = None
    plugin: str | None = None
    strategy: str = Field(min_length=1)
    attrs: list[str] = Field(default_factory=list)
    engine: str | None = None
    model: str | None = None
    """The model that participated in producing this probe, where one did (FR-046).

    Same convention as ``engine`` and ``PrincipleGrade.model``: recorded so a model-generated
    probe set is attributable, never branched on. ``None`` is the deterministic path, and it is
    what every probe of the templated engines carries.
    """
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
    ungraded_reason: str | None = None
    """Why this exchange carries no violation, and ``None`` when it carries one.

    ``violation is None`` says a number is missing; it does not say whether the assistant never
    answered or the judge never ruled. Those call for opposite responses — one is the assistant's
    transport, the other is the grading model's — and the run used to discard the distinction along
    with ``TargetResponse.failure_reason``, which nothing read.

    It matters most when the count is small and nobody looks twice. Two ungraded out of sixty reads
    like noise either way; that both were the judge, on consecutive probes, is what says the run
    should be repeated rather than reported.
    """


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
    probes: list[Probe] = Field(default_factory=list)
    """The probes the run sent, so the result stands on its own.

    ``GradedOutcome`` names a ``probe_id`` and carries no query, and the probes used to live only in
    whatever variable the caller happened to keep. So a serialised result could not be turned back
    into a dataset — ``to_dataset`` needs both halves — and reading one meant holding the probe set
    alongside it and trusting that the two came from the same run.
    """

    overall_rate: float = Field(ge=0.0, le=1.0)
    n_scoreable: int = Field(ge=0)
    n_ungraded: int = Field(ge=0)
    grading_methods: dict[str, int] = Field(default_factory=dict)
    """How many grades each judging method produced, so ``overall_rate`` says what estimated it.

    A grader may reach a verdict more than one way — the shipped one reads the verdict token's
    distribution and votes across samples when it cannot — and those are two estimators, not two
    implementations of one. A rate averaged over both is an average over two measurements, and per
    grade the fact was already recorded and never aggregated, so a run that degraded throughout
    looked exactly like a run that did not.
    """


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
    """Whether each recorded query was asked. Invariantly ``True`` since a query the ``kappa`` gate
    stopped is regenerated or discarded rather than scored, so no unasked query reaches this list.
    Kept as the guard readers already apply, because a search is substitutable and nothing but this
    field says a query was sent."""

    realism_gap: float
    score: float
    passed: bool
    """Whether ``score`` reached the ``tau`` in force, decided once where ``tau`` is known.

    Reconstructable from the score and the ``tau`` recorded alongside the components, and
    previously never stated — so every consumer reimplemented the comparison, and one of them
    eventually writes ``>`` where the method says ``>=``. The kind of defect that produces a
    plausible number rather than an error.
    """

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


class RecordProvenance(BaseModel):
    """Where a Roast Dataset record came from, so a number can be walked back to what produced it.

    Every field is optional because the two halves of a run know different things. A record the
    Profiler wrote came from a probe and names it; a record the Exploiter surfaced was a query a
    model invented, with no probe behind it, and names the category that proposed it instead.

    It travels as one object rather than as loose optional fields on the record so that "this came
    from nowhere" stays sayable: an Exploiter record carries a ``category`` and no ``probe_id``, and
    that shape is the fact rather than an accident of which columns happened to be null.
    """

    probe_id: str | None = None
    """The probe this record's query came from. ``None`` for a query the search invented."""

    strategy: str | None = None
    """The interaction pattern that asked for the probe. The user's own identifier, so it reaches a
    record but never the profile — FR-013 forbids it crossing to the Exploiter, not existing."""

    engine: str | None = None
    """Which engine produced the probe, so the absence/breadth trade-off stays measurable after the
    run and not only inside the probe set (FR-022)."""

    hook: KnowledgeHook | None = None
    """The probe's provenance against the knowledge base, carried whole.

    It is what finally gives ``absence_reliable`` and ``verified`` a reader. Both were written and
    neither was ever looked at again: an unreliable absence label was indistinguishable from a
    confirmed one by the time anybody saw a violation rate, which is the distinction FR-023 exists
    to preserve.
    """

    category: list[str] | None = None
    """The attributes of the category a surfaced query belonged to.

    ``queries_over_threshold`` flattens every category into one list, so without this a surfaced
    record has to be matched back to its category by the text of its query.
    """


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
    provenance: RecordProvenance = Field(default_factory=RecordProvenance)
    """What produced this record. A record used to carry a query and a score and nothing that said
    where either came from, so reading a violation meant matching its text back against the probe
    set by hand."""


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
    grading_methods: dict[str, int] = Field(default_factory=dict)
    """How many grades each judging method produced across the search. Same reason as on
    ``ProfilerResult``: the two paths of the shipped grader are two estimators, and every ``S(c)``
    here is a mean over whichever ones answered."""


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
    "EngineDeclaration",
    "ExploiterConfig",
    "FailureReport",
    "GradedOutcome",
    "GraderConfig",
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
