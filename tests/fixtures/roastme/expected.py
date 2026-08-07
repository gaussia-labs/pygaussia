"""Hand-computed Roast Me fixtures (T017): probes, grades, and every expected number.

Every expected value below is worked out from the paper's equations as `data-model.md` and
`spec.md` state them, and written as a literal with its derivation in the comment beside it.
Nothing here is a snapshot of what any implementation produced — SC-001 requires the arithmetic
be reproducible by hand to within `1e-9`, and a snapshot would assert only self-consistency.

The two formulas everything rests on:

    v(x, r) = sum_j w_j * pi_hat_j(x, r)                                    (eq:violation)

    se_n(values) = sqrt( sum_i (x_i - mean)^2 / n ) / sqrt(n)
                 = sqrt( sum_i (x_i - mean)^2 ) / n

The second is one formula serving both places a standard error appears, which is what
`data-model.md` asks for when it calls `WeaknessEntry.standard_error` "the same quantity `S(c)`
subtracts, computed the same way". Two consequences pin it down and no other form satisfies
both:

  * On binary values (k ones out of n) the uncorrected sample variance is exactly p(1-p), so
    the expression collapses to the binomial standard error sqrt(p(1-p)/n) that `data-model.md`
    names for the weakness map — exactly, in real arithmetic: the collapse costs no precision
    and explains no discrepancy between literals.
  * At n = 1 the deviation is identically zero, so se is zero *by construction* and
    `S(c) = mean - lambda*se` degenerates to the raw mean — the fact FR-040 cites to justify
    the `ge=2` floor (SC-013). Bessel's correction would divide by zero there instead of
    yielding zero, so the uncorrected form is the only one that makes SC-013's statement true.

Every literal below is written as the closed form quoted beside it evaluates in double
precision. The two orderings of that closed form are not bit-identical — `sqrt(a/n)` and
`sqrt(a)/sqrt(n)` can land one ULP apart, and `test_expected_fixtures.py` recomputes them the
second way — so the literals are pinned to the first, which is the ordering `scoring.py` uses.
Every assertion against them is made to 1e-9, fifteen orders of magnitude wider than that gap.

Identifiers for plugins, strategies, transforms and entity kinds are deliberately meaningless
strings. SC-003 requires that no library behaviour and no test assertion depend on what a
plugin, strategy or template is called, and an opaque value makes an accidental dependency
visible instead of plausible.
"""

from gaussia.schemas.roastme import (
    BehavioralContract,
    Catalogue,
    Category,
    Document,
    KnowledgeHook,
    PluginSpec,
    Principle,
    Probe,
    StrategySpec,
)

from .doubles import RecordedTarget, StubGrader

PRINCIPLE_A = "no_invention"
PRINCIPLE_B = "no_contradiction"
PRINCIPLE_C = "no_overreach"

WEIGHT_A = 0.5
WEIGHT_B = 0.3
WEIGHT_C = 0.2

PLUGIN_ONE = "pl-4f21"
PLUGIN_TWO = "pl-9c07"
STRATEGY_ONE = "st-1a6b"
STRATEGY_TWO = "st-77de"
STRATEGY_CONTROL = "st-0e33"
TRANSFORM_KEEP = "tf-c410"
TRANSFORM_INVENT = "tf-b8a2"
ENTITY_KIND = "ek-2d55"

# The same real, documented entity behind two probes: one whose strategy names a plugin and one
# whose strategy names none. SC-003 puts them on opposite sides of the scoring line, and the
# entity's existence decides nothing.
SHARED_ENTITY = "entity-shared-7b91"

QUERY_ONE = "probe query one"
QUERY_TWO = "probe query two"
QUERY_THREE = "probe query three"
QUERY_FOUR = "probe query four"
QUERY_FIVE = "probe query five"
QUERY_SIX = "probe query six"
QUERY_CONTROL = "probe query control"
QUERY_FAILED = "probe query failed exchange"

ATTRS_ONE = ["asks for an exception to a stated rule", "phrased as a hypothetical"]
ATTRS_TWO = ["cites a named clause", "asks for a numeric limit"]
ATTRS_CONTROL = ["asks a documented question plainly"]

# pi_hat_j per (query, principle). Binary on purpose: on binary grades the weakness rate is both
# the mean of the scores and the proportion of violations, so the fixture holds under either
# reading of "violations over trials".
GRADES: dict[tuple[str, str], float] = {
    (QUERY_ONE, PRINCIPLE_A): 1.0,
    (QUERY_ONE, PRINCIPLE_B): 1.0,
    (QUERY_ONE, PRINCIPLE_C): 0.0,
    (QUERY_TWO, PRINCIPLE_A): 1.0,
    (QUERY_TWO, PRINCIPLE_B): 0.0,
    (QUERY_TWO, PRINCIPLE_C): 0.0,
    (QUERY_THREE, PRINCIPLE_A): 0.0,
    (QUERY_THREE, PRINCIPLE_B): 0.0,
    (QUERY_THREE, PRINCIPLE_C): 0.0,
    (QUERY_FOUR, PRINCIPLE_A): 1.0,
    (QUERY_FOUR, PRINCIPLE_B): 1.0,
    (QUERY_FOUR, PRINCIPLE_C): 1.0,
    (QUERY_FIVE, PRINCIPLE_A): 0.0,
    (QUERY_FIVE, PRINCIPLE_B): 1.0,
    (QUERY_FIVE, PRINCIPLE_C): 1.0,
    (QUERY_SIX, PRINCIPLE_A): 0.0,
    (QUERY_SIX, PRINCIPLE_B): 0.0,
    (QUERY_SIX, PRINCIPLE_C): 1.0,
    (QUERY_CONTROL, PRINCIPLE_A): 1.0,
    (QUERY_CONTROL, PRINCIPLE_B): 1.0,
    (QUERY_CONTROL, PRINCIPLE_C): 1.0,
}

RESPONSES: dict[str, str] = {
    QUERY_ONE: "response one",
    QUERY_TWO: "response two",
    QUERY_THREE: "response three",
    QUERY_FOUR: "response four",
    QUERY_FIVE: "response five",
    QUERY_SIX: "response six",
    QUERY_CONTROL: "response control",
}

FAILURE_REASON = "stub transport failure"

# v = 0.5*A + 0.3*B + 0.2*C
EXPECTED_VIOLATION: dict[str, float] = {
    QUERY_ONE: 0.8,  # 0.5*1 + 0.3*1 + 0.2*0
    QUERY_TWO: 0.5,  # 0.5*1 + 0.3*0 + 0.2*0
    QUERY_THREE: 0.0,  # 0.5*0 + 0.3*0 + 0.2*0
    QUERY_FOUR: 1.0,  # 0.5*1 + 0.3*1 + 0.2*1
    QUERY_FIVE: 0.5,  # 0.5*0 + 0.3*1 + 0.2*1
    QUERY_SIX: 0.2,  # 0.5*0 + 0.3*0 + 0.2*1
    QUERY_CONTROL: 1.0,  # graded and recorded, but excluded from every rate (FR-011)
}

# SC-002: a response violating only B and C. The weighted sum is 0.5, while a naive count of
# violated principles is 2/3 = 0.666..., so a pass threshold of 0.6 that the naive count clears
# is one the weighted score fails. This is why v is a weighted sum and not a count.
PARTIAL_VIOLATION_GRADES = {PRINCIPLE_A: 0.0, PRINCIPLE_B: 1.0, PRINCIPLE_C: 1.0}
PARTIAL_VIOLATION = 0.5  # 0.3 + 0.2
PARTIAL_NAIVE_COUNT_FRACTION = 2.0 / 3.0
PARTIAL_THRESHOLD = 0.6

# A fractional-grade case, so v is exercised away from the binary corners.
FRACTIONAL_GRADES = {PRINCIPLE_A: 0.5, PRINCIPLE_B: 0.25, PRINCIPLE_C: 1.0}
FRACTIONAL_VIOLATION = 0.525  # 0.5*0.5 + 0.3*0.25 + 0.2*1.0 = 0.25 + 0.075 + 0.2

# The violation rate over scoreable, graded outcomes: the control and the failed exchange are
# both out of it. (0.8 + 0.5 + 0.0 + 1.0 + 0.5 + 0.2) / 6 = 3.0 / 6
EXPECTED_OVERALL_RATE = 0.5
EXPECTED_SCOREABLE = 6
EXPECTED_UNGRADED = 1
EXPECTED_OUTCOMES = 8

# Standard errors of the weakness map, from se = sqrt(sum (x-mean)^2) / n.
SE_RATE_075_N4 = 0.21650635094610965  # sqrt(0.75*0.25/4) = sqrt(0.046875)
SE_RATE_050_N4 = 0.25  # sqrt(0.50*0.50/4) = sqrt(0.0625)
SE_RATE_025_N4 = 0.21650635094610965  # sqrt(0.25*0.75/4) = sqrt(0.046875)
SE_RATE_050_N2 = 0.3535533905932738  # sqrt(0.50*0.50/2) = sqrt(0.125)

# Per-principle grade vectors per strategy group, and the rate, sample size and standard error
# each one produces. Descriptor group one is queries one to four; group two is five and six. The
# failed exchange contributes no trial, so group one is n = 4 and not n = 5.
WEAKNESS_GROUP_ONE: dict[str, list[float]] = {
    PRINCIPLE_A: [1.0, 1.0, 0.0, 1.0],
    PRINCIPLE_B: [1.0, 0.0, 0.0, 1.0],
    PRINCIPLE_C: [0.0, 0.0, 0.0, 1.0],
}
WEAKNESS_GROUP_TWO: dict[str, list[float]] = {
    PRINCIPLE_A: [0.0, 0.0],
    PRINCIPLE_B: [1.0, 0.0],
    PRINCIPLE_C: [1.0, 1.0],
}

# (principle, rate, n, standard error), sorted by (principle, n) — the descriptor text is
# asserted separately, because invariant 3 forbids the profile carrying anything a test could
# key on here.
EXPECTED_WEAKNESSES: list[tuple[str, float, int, float]] = [
    (PRINCIPLE_B, 0.5, 2, SE_RATE_050_N2),  # [1,0]:      rate 1/2
    (PRINCIPLE_B, 0.5, 4, SE_RATE_050_N4),  # [1,0,0,1]:  rate 2/4
    (PRINCIPLE_C, 1.0, 2, 0.0),  # [1,1]:      rate 2/2, no dispersion
    (PRINCIPLE_C, 0.25, 4, SE_RATE_025_N4),  # [0,0,0,1]:  rate 1/4
    (PRINCIPLE_A, 0.0, 2, 0.0),  # [0,0]:      rate 0/2, no dispersion
    (PRINCIPLE_A, 0.75, 4, SE_RATE_075_N4),  # [1,1,0,1]:  rate 3/4
]

# Every scoreable, non-control descriptor pairs with every principle of the contract: each
# response is graded against all m principles (FR-004), so each contributes a trial to each pair.
EXPECTED_WEAKNESS_COUNT = 6

# S(c) fixtures. Two categories with the same mean and different dispersion: the consistent one
# must rank above the spiky one, which is the whole point of scoring a lower-confidence bound.
CONSISTENT_VIOLATIONS = [0.6, 0.6, 0.6, 0.6]
CONSISTENT_MEAN = 0.6
CONSISTENT_SE = 0.0  # every deviation is zero
CONSISTENT_SCORE_LAMBDA_1 = 0.6  # 0.6 - 1.0*0.0

SPIKY_VIOLATIONS = [1.0, 0.2, 1.0, 0.2]
SPIKY_MEAN = 0.6  # (1.0 + 0.2 + 1.0 + 0.2) / 4
SPIKY_SE = 0.2  # sqrt(4 * 0.4^2) / 4 = 0.8 / 4
SPIKY_SCORE_LAMBDA_1 = 0.4  # 0.6 - 1.0*0.2
SPIKY_SCORE_LAMBDA_2 = 0.2  # 0.6 - 2.0*0.2

# The kappa gate. Both queries broke the assistant outright, but only one is on profile, so the
# category is scored over a gated vector — which costs it far more than halving the mean, because
# the zero also creates the dispersion the penalty reads.
GATE_RAW_VIOLATIONS = [1.0, 1.0]
GATE_ON_PROFILE_SCORES = [0.9, 0.1]
GATE_KAPPA = 0.5
GATED_VIOLATIONS = [1.0, 0.0]
GATED_MEAN = 0.5
GATED_SE = 0.3535533905932738  # sqrt(2 * 0.5^2) / 2 = sqrt(0.5) / 2
GATED_SCORE_LAMBDA_1 = 0.1464466094067262  # 0.5 - sqrt(0.5)/2
UNGATED_SCORE_LAMBDA_1 = 1.0  # 1.0 - 1.0*0.0, had neither query been gated

# SC-013: at n = 1 the penalty vanishes for every lambda, which is why the floor is 2.
SINGLE_VIOLATIONS = [0.7]
SINGLE_MEAN = 0.7
SINGLE_SE = 0.0
SINGLE_SCORE = 0.7

REALISM_GAP_WITHIN = 0.3  # equal to delta: a category "over delta" is discarded, not one at it
REALISM_GAP_OVER = 0.30000001
REALISM_DELTA = 0.3

# Refinement. The evaluation of every sub-conjunction is prescribed, so the minimal one is a fact
# about the fixture and not about a search. Three subsets satisfy both S(c') >= tau = 0.5 and
# D <= delta = 0.3 — the whole conjunction, one pair and one singleton — and the smallest of them
# is unique, so no tie-break is being asserted.
REFINE_ATTRIBUTES = ["mentions an unpublished exception", "asks in the second person", "quotes a figure"]
REFINE_PROVENANCE = ["weakness one", "weakness two", "hook one"]
REFINE_TAU = 0.5
REFINE_DELTA = 0.3
REFINE_TABLE: dict[tuple[str, ...], tuple[float, float]] = {
    (REFINE_ATTRIBUTES[0], REFINE_ATTRIBUTES[1], REFINE_ATTRIBUTES[2]): (0.70, 0.20),  # passes both
    (REFINE_ATTRIBUTES[0], REFINE_ATTRIBUTES[1]): (0.65, 0.20),  # passes both
    (REFINE_ATTRIBUTES[0], REFINE_ATTRIBUTES[2]): (0.40, 0.10),  # below tau
    (REFINE_ATTRIBUTES[1], REFINE_ATTRIBUTES[2]): (0.60, 0.50),  # over delta
    (REFINE_ATTRIBUTES[0],): (0.55, 0.90),  # over delta
    (REFINE_ATTRIBUTES[1],): (0.52, 0.25),  # passes both, and is the unique minimum
    (REFINE_ATTRIBUTES[2],): (0.30, 0.10),  # below tau
}
REFINE_EXPECTED_ATTRIBUTES = [REFINE_ATTRIBUTES[1]]
REFINE_EXPECTED_PROVENANCE = [REFINE_PROVENANCE[1]]
REFINE_EXPECTED_DROPPED = [REFINE_ATTRIBUTES[0], REFINE_ATTRIBUTES[2]]


def refine_category() -> Category:
    return Category(attributes=list(REFINE_ATTRIBUTES), provenance=list(REFINE_PROVENANCE))


def stub_grader() -> StubGrader:
    return StubGrader(GRADES)


def contract(grader: StubGrader) -> BehavioralContract:
    """The hand-built contract: three principles, weights summing to exactly 1.0."""
    return BehavioralContract(
        principles=[
            Principle(id=PRINCIPLE_A, weight=WEIGHT_A, rubric="rubric for A", grader=grader),
            Principle(id=PRINCIPLE_B, weight=WEIGHT_B, rubric="rubric for B", grader=grader),
            Principle(id=PRINCIPLE_C, weight=WEIGHT_C, rubric="rubric for C", grader=grader),
        ]
    )


# A contract FR-001 accepts and whose weights nonetheless sum to more than 1.0: the tolerance
# exists so a contract assembled from decimals is not rejected for float noise, and 5e-10 is
# inside it. Every principle violated outright is then the worst case v can reach, which is
# where the `le=1.0` bounds of the schemas meet the `1 +- 1e-9` the validator grants.
EDGE_WEIGHT_C = 0.2000000005
EDGE_WEIGHT_SUM = 1.0000000005  # 0.5 + 0.3 + 0.2000000005


def tolerance_edge_contract(grader: StubGrader) -> BehavioralContract:
    """The same three principles, weighted to the far edge of what FR-001 tolerates."""
    return BehavioralContract(
        principles=[
            Principle(id=PRINCIPLE_A, weight=WEIGHT_A, rubric="rubric for A", grader=grader),
            Principle(id=PRINCIPLE_B, weight=WEIGHT_B, rubric="rubric for B", grader=grader),
            Principle(id=PRINCIPLE_C, weight=EDGE_WEIGHT_C, rubric="rubric for C", grader=grader),
        ]
    )


def _hook(references: str, doc: int, principle: str | None, how: str) -> KnowledgeHook:
    return KnowledgeHook(kind=ENTITY_KIND, references=references, doc=doc, how=how, principle=principle)


def probes() -> list[Probe]:
    """Eight probes: four on descriptor one, two on descriptor two, one control, one failing.

    The control and the third probe reference the same real, documented entity: the control is
    excluded because its strategy names no plugin, not because of anything about the entity
    (FR-011, SC-003).
    """
    return [
        Probe(
            id="pb-1",
            query=QUERY_ONE,
            hook=_hook("entity-one", 0, PRINCIPLE_A, TRANSFORM_INVENT),
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
            meta={"real_value": "one", "false_value": "one-prime"},
        ),
        Probe(
            id="pb-2",
            query=QUERY_TWO,
            hook=_hook("entity-two", 0, PRINCIPLE_A, TRANSFORM_INVENT),
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
            meta={"real_value": "two", "false_value": "two-prime"},
        ),
        Probe(
            id="pb-3",
            query=QUERY_THREE,
            hook=_hook(SHARED_ENTITY, 1, PRINCIPLE_A, TRANSFORM_KEEP),
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
            meta={"real_value": SHARED_ENTITY},
        ),
        Probe(
            id="pb-4",
            query=QUERY_FOUR,
            hook=_hook("entity-four", 0, PRINCIPLE_A, TRANSFORM_INVENT),
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
            meta={"real_value": "four", "false_value": "four-prime"},
        ),
        Probe(
            id="pb-5",
            query=QUERY_FIVE,
            hook=_hook("entity-five", 1, PRINCIPLE_B, TRANSFORM_KEEP),
            plugin=PLUGIN_TWO,
            strategy=STRATEGY_TWO,
            attrs=list(ATTRS_TWO),
            meta={"real_value": "five"},
        ),
        Probe(
            id="pb-6",
            query=QUERY_SIX,
            hook=_hook("entity-six", 1, PRINCIPLE_B, TRANSFORM_KEEP),
            plugin=PLUGIN_TWO,
            strategy=STRATEGY_TWO,
            attrs=list(ATTRS_TWO),
            meta={"real_value": "six"},
        ),
        Probe(
            id="pb-control",
            query=QUERY_CONTROL,
            hook=_hook(SHARED_ENTITY, 1, None, TRANSFORM_KEEP),
            plugin=None,
            strategy=STRATEGY_CONTROL,
            attrs=list(ATTRS_CONTROL),
            meta={"real_value": SHARED_ENTITY},
        ),
        Probe(
            id="pb-failed",
            query=QUERY_FAILED,
            hook=_hook("entity-failed", 0, PRINCIPLE_A, TRANSFORM_INVENT),
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
            meta={"real_value": "failed", "false_value": "failed-prime"},
        ),
    ]


SCOREABLE_QUERIES = [QUERY_ONE, QUERY_TWO, QUERY_THREE, QUERY_FOUR, QUERY_FIVE, QUERY_SIX]
VIOLATING_QUERIES = [QUERY_ONE, QUERY_TWO, QUERY_FOUR, QUERY_FIVE, QUERY_SIX]
OPAQUE_IDENTIFIERS = [
    PLUGIN_ONE,
    PLUGIN_TWO,
    STRATEGY_ONE,
    STRATEGY_TWO,
    STRATEGY_CONTROL,
    TRANSFORM_KEEP,
    TRANSFORM_INVENT,
]


def recorded_target() -> RecordedTarget:
    """The recorded response set, with the one exchange the adapter reports as failed."""
    return RecordedTarget(responses=RESPONSES, failures={QUERY_FAILED: FAILURE_REASON})


# FR-015: a probe leaning on nothing gives the grader no knowledge-base evidence to check the
# response against, and the record has to say so. Hook and meta are both empty here, so the
# expectation holds whether the implementation reads one or the other.
BLACKBOX_QUERY = "domain agnostic query"
BLACKBOX_RESPONSES = {BLACKBOX_QUERY: "domain agnostic response"}
BLACKBOX_GRADES = {
    (BLACKBOX_QUERY, PRINCIPLE_A): 1.0,
    (BLACKBOX_QUERY, PRINCIPLE_B): 0.0,
    (BLACKBOX_QUERY, PRINCIPLE_C): 0.0,
}
BLACKBOX_VIOLATION = 0.5  # 0.5*1 + 0.3*0 + 0.2*0


def blackbox_probes() -> list[Probe]:
    return [
        Probe(
            id="pb-blackbox",
            query=BLACKBOX_QUERY,
            hook=None,
            plugin=PLUGIN_ONE,
            strategy=STRATEGY_ONE,
            attrs=list(ATTRS_ONE),
        )
    ]


# A knowledge base whose entity set is known exactly, so an absence label can be scored against
# an enumeration the engine under test never sees (FR-021, SC-004). One document is enumerable
# and one is not, which is the field the enumeration engine reads to know whether a boundary can
# be established at all.
DOCUMENT_STRUCTURED_ID = "doc-structured"
DOCUMENT_PROSE_ID = "doc-prose"

KNOWN_ENTITIES = frozenset({SHARED_ENTITY, "entity-alpha", "entity-beta"})


def documents() -> list[Document]:
    return [
        Document(
            id=DOCUMENT_STRUCTURED_ID,
            content=f"Clause one concerns {SHARED_ENTITY}. Clause two concerns entity-alpha. "
            f"Clause three concerns entity-beta.",
            structured=True,
            kind="clause-list",
        ),
        Document(
            id=DOCUMENT_PROSE_ID,
            content=f"A discursive account mentioning {SHARED_ENTITY} and entity-alpha in passing.",
            structured=False,
        ),
    ]


def plugin_specs() -> list[PluginSpec]:
    return [
        PluginSpec(id=PLUGIN_ONE, name="family one", description="documentation only", principle=PRINCIPLE_A),
        PluginSpec(id=PLUGIN_TWO, name="family two", description="documentation only", principle=PRINCIPLE_B),
    ]


def strategy_specs(transform_key: str, control_transform_key: str | None = None) -> list[StrategySpec]:
    """Two scoring strategies and one control.

    The control is a strategy with no plugin — the only mechanism by which a control is
    recognised (FR-026). The transform keys are passed in rather than written here, because the
    registry that decides which strings are valid belongs to the implementation.
    """
    return [
        StrategySpec(
            id=STRATEGY_ONE,
            name="pattern one",
            description="asks for an exception to a stated rule, phrased as a hypothetical",
            plugin=PLUGIN_ONE,
            entity_kind=ENTITY_KIND,
            transform=transform_key,
            doc=0,
            phrasing_hint="phrase it as a hypothetical",
        ),
        StrategySpec(
            id=STRATEGY_TWO,
            name="pattern two",
            description="cites a named clause and asks for a numeric limit",
            plugin=PLUGIN_TWO,
            entity_kind=ENTITY_KIND,
            transform=control_transform_key or transform_key,
            doc=1,
            phrasing_hint="cite the clause by name",
        ),
        StrategySpec(
            id=STRATEGY_CONTROL,
            name="control pattern",
            description="asks a documented question plainly",
            plugin=None,
            entity_kind=ENTITY_KIND,
            transform=control_transform_key or transform_key,
            doc=1,
            phrasing_hint="ask it plainly",
        ),
    ]


def catalogue(transform_key: str, control_transform_key: str | None = None) -> Catalogue:
    return Catalogue(
        plugins=plugin_specs(),
        strategies=strategy_specs(transform_key, control_transform_key),
    )
