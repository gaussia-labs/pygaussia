"""The training-free category search: grounding, pooling and refinement (T048).

The default of the two procedures (FR-033) and the one a user following the documentation runs,
so its three steps are covered here in the terms the requirements state them:

* **ground** — every attribute comes from a weakness at or above `eta` or from a retained hook,
  and carries the entry that induced it (FR-028). The query generator double raises on a
  category it was given no queries for, so an invented attribute fails the run rather than
  quietly costing a target call;
* **pool** — the attributes behind the highest-scoring query/response pairs, bounded by
  `pool_size` and by `max_attributes`;
* **refine** — the minimal sub-conjunction passing `tau` and `delta`, with what came off reported
  as incidental (FR-032), and the fallback when no sub-conjunction passes at all.

The gates belong to `CategoryEvaluator` rather than to this search, so what is asserted here is
that they reach it: a category over `delta` never touches the target and never enters the pool,
and a query below `kappa` contributes exactly zero without being sent (FR-030, FR-031).

Every expected score is hand-computed from `v = 0.5*A + 0.3*B + 0.2*C` and
`S(c) = mean(v) - lambda*se(v)`, with the derivation beside it. Nothing here opens a socket,
reads a credential or touches a GPU.
"""

import pytest

from gaussia.generators.roastme.searches.attribute_iteration import (
    DEFAULT_MAX_ATTRIBUTES,
    AttributeIterationSearch,
)
from gaussia.schemas.roastme import AssistantProfile, ExploiterConfig, KnowledgeHook, WeaknessEntry
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import (
    RecommendingOnProfileFilter,
    RecommendingRealismEstimator,
    RecordedTarget,
    StubGrader,
    StubQueryGenerator,
)

TOLERANCE = 1e-9

ETA = 0.5
TAU = 0.9
KAPPA = 0.5
DELTA = 0.3
QUERIES_PER_CATEGORY = 2

# More pairs than the three seeds can produce, so the pool is unbounded unless a test bounds it.
WHOLE_POOL = 20

DESCRIPTOR_ONE = "invents a clause the policy does not contain"
DESCRIPTOR_TWO = "contradicts a limit it has already stated"
DESCRIPTOR_BELOW_ETA = "answers a question outside its remit"
HOOK_REFERENCE = "entity-one"

# What the profile below justifies, and the provenance each attribute must carry (FR-028). The
# expected strings are written out rather than formatted from the module's own templates: a
# re-derived expectation would agree with any template the search happened to use.
ATTRIBUTE_ONE = DESCRIPTOR_ONE
ATTRIBUTE_TWO = DESCRIPTOR_TWO
ATTRIBUTE_HOOK = f"concerns {HOOK_REFERENCE}"
PROVENANCE_ONE = f"weakness {fx.PRINCIPLE_A}: {DESCRIPTOR_ONE}"
PROVENANCE_TWO = f"weakness {fx.PRINCIPLE_B}: {DESCRIPTOR_TWO}"
PROVENANCE_HOOK = f"hook: {HOOK_REFERENCE}"

ONE = (ATTRIBUTE_ONE,)
TWO = (ATTRIBUTE_TWO,)
HOOK = (ATTRIBUTE_HOOK,)
ONE_TWO = (ATTRIBUTE_ONE, ATTRIBUTE_TWO)
ONE_HOOK = (ATTRIBUTE_ONE, ATTRIBUTE_HOOK)
TWO_HOOK = (ATTRIBUTE_TWO, ATTRIBUTE_HOOK)
ALL_THREE = (ATTRIBUTE_ONE, ATTRIBUTE_TWO, ATTRIBUTE_HOOK)

# Every conjunction the search may ask about, so an unexpected one raises instead of scoring.
QUERIES: dict[tuple[str, ...], list[str]] = {
    ONE: ["one-a", "one-b"],
    TWO: ["two-a", "two-b"],
    HOOK: ["hook-a", "hook-b"],
    ONE_TWO: ["one-two-a", "one-two-b"],
    ONE_HOOK: ["one-hook-a", "one-hook-b"],
    TWO_HOOK: ["two-hook-a", "two-hook-b"],
    ALL_THREE: ["all-a", "all-b"],
}
ALL_QUERIES = [query for queries in QUERIES.values() for query in queries]
GATED_QUERY = QUERIES[TWO][1]

# pi_hat per principle, and the v = 0.5*A + 0.3*B + 0.2*C each triple produces.
BROKEN = (1.0, 1.0, 1.0)
HEAVY = (1.0, 1.0, 0.0)
LIGHT = (1.0, 0.0, 0.0)
VIOLATION_BROKEN = 1.0  # 0.5*1 + 0.3*1 + 0.2*1
VIOLATION_HEAVY = 0.8  # 0.5*1 + 0.3*1 + 0.2*0
VIOLATION_LIGHT = 0.5  # 0.5*1 + 0.3*0 + 0.2*0

# Both queries of a category score alike, so se is zero and S(c) is the mean exactly.
SCORE_BROKEN = VIOLATION_BROKEN  # 1.0 - 1.0*0.0
SCORE_HEAVY = VIOLATION_HEAVY  # 0.8 - 1.0*0.0
SCORE_LIGHT = VIOLATION_LIGHT  # 0.5 - 1.0*0.0

# One seed clears tau = 0.9 on its own, so refinement stops at the first singleton it tries.
BREAKING_GRADES: dict[tuple[str, ...], tuple[float, float, float]] = {
    ONE: BROKEN,
    TWO: HEAVY,
    HOOK: LIGHT,
    ONE_TWO: HEAVY,
    ONE_HOOK: HEAVY,
    TWO_HOOK: HEAVY,
    ALL_THREE: HEAVY,
}

# Nothing reaches tau = 0.9: every sub-conjunction scores 0.8 or less, so refinement exhausts all
# 2^3 - 1 of them and falls back to the candidate whole.
UNBREAKING_GRADES: dict[tuple[str, ...], tuple[float, float, float]] = {
    ONE: HEAVY,
    TWO: LIGHT,
    HOOK: LIGHT,
    ONE_TWO: HEAVY,
    ONE_HOOK: HEAVY,
    TWO_HOOK: HEAVY,
    ALL_THREE: HEAVY,
}

# No singleton reaches tau but the first pair does, so the minimal sub-conjunction is a category
# no seed ever proposed — which is where refinement has to carry the provenance across on its own.
PAIRED_GRADES: dict[tuple[str, ...], tuple[float, float, float]] = {**UNBREAKING_GRADES, ONE_TWO: BROKEN}

ON_PROFILE = 0.9
BELOW_KAPPA = 0.1
WITHIN_DELTA = 0.2
OVER_DELTA = 0.4

# The gated vector of category two: the query below kappa contributes exactly 0.0 and stays in the
# denominator, so [0.8, 0.0] has mean 0.4 and se sqrt((2*0.4^2)/2 / 2) = sqrt(0.08).
GATED_VIOLATIONS = [VIOLATION_HEAVY, 0.0]
GATED_SE = 0.28284271247461906  # sqrt(0.16/2) = sqrt(0.08)
GATED_SCORE = 0.11715728752538102  # 0.4 - sqrt(0.08)


def _weakness(principle: str, descriptor: str, rate: float, error: float) -> WeaknessEntry:
    return WeaknessEntry(principle=principle, descriptor=descriptor, rate=rate, n=4, standard_error=error)


def _below_eta() -> WeaknessEntry:
    return _weakness(fx.PRINCIPLE_C, DESCRIPTOR_BELOW_ETA, 0.25, fx.SE_RATE_025_N4)


def _profile() -> AssistantProfile:
    """A profile with a weakness above `eta`, one exactly at it, one below, and one repeat.

    The fourth entry describes the behaviour of the first under a second principle, which is what
    the weakness map does when two principles catch the same failure.
    """
    return AssistantProfile(
        weaknesses=[
            _weakness(fx.PRINCIPLE_A, DESCRIPTOR_ONE, 0.75, fx.SE_RATE_075_N4),
            _weakness(fx.PRINCIPLE_B, DESCRIPTOR_TWO, ETA, fx.SE_RATE_050_N4),
            _below_eta(),
            _weakness(fx.PRINCIPLE_B, DESCRIPTOR_ONE, 0.75, fx.SE_RATE_075_N4),
        ],
        hooks=[
            KnowledgeHook(
                kind=fx.ENTITY_KIND,
                references=HOOK_REFERENCE,
                doc=0,
                how=fx.TRANSFORM_INVENT,
                principle=fx.PRINCIPLE_A,
            )
        ],
    )


def _grades(assignment: dict[tuple[str, ...], tuple[float, float, float]]) -> dict[tuple[str, str], float]:
    principles = (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)
    return {
        (query, principle): score
        for key, triple in assignment.items()
        for query in QUERIES[key]
        for principle, score in zip(principles, triple, strict=True)
    }


def _on_profile(gated: tuple[str, ...] = ()) -> dict[str, float]:
    return {query: BELOW_KAPPA if query in gated else ON_PROFILE for query in ALL_QUERIES}


def _gaps(over: tuple[tuple[str, ...], ...] = ()) -> dict[tuple[str, ...], float]:
    return {tuple(queries): OVER_DELTA if key in over else WITHIN_DELTA for key, queries in QUERIES.items()}


def _run(
    grades=BREAKING_GRADES,
    profile=None,
    pool_size=WHOLE_POOL,
    max_attributes=DEFAULT_MAX_ATTRIBUTES,
    gated=(),
    over_delta=(),
    failures=None,
    kappa=KAPPA,
):
    query_generator = StubQueryGenerator({key: list(queries) for key, queries in QUERIES.items()})
    target = RecordedTarget(
        responses={query: f"response to {query}" for query in ALL_QUERIES},
        failures=failures,
    )
    evaluations = AttributeIterationSearch(max_attributes=max_attributes).search(
        profile or _profile(),
        fx.contract(StubGrader(_grades(grades))),
        ExploiterConfig(
            tau=TAU,
            eta=ETA,
            queries_per_category=QUERIES_PER_CATEGORY,
            pool_size=pool_size,
            kappa=kappa,
            delta=DELTA,
        ),
        target,
        query_generator,
        RecommendingOnProfileFilter(_on_profile(gated)),
        RecommendingRealismEstimator(_gaps(over_delta)),
    )
    return evaluations, query_generator, target


def _reported(evaluations):
    return {tuple(evaluation.category.attributes): evaluation for evaluation in evaluations}


def _asked(query_generator):
    return {key for key, _ in query_generator.calls}


class TestGrounding:
    def test_nothing_outside_the_profile_is_proposed(self):
        """FR-028: a category grounded in something the Profiler never observed is a guess."""
        evaluations, _, _ = _run()
        proposed = {attribute for evaluation in evaluations for attribute in evaluation.category.attributes}

        assert proposed == {ATTRIBUTE_ONE, ATTRIBUTE_TWO, ATTRIBUTE_HOOK}

    def test_a_weakness_below_eta_contributes_nothing(self):
        _, query_generator, _ = _run()
        attempted = {attribute for key in _asked(query_generator) for attribute in key}

        assert DESCRIPTOR_BELOW_ETA not in attempted

    def test_a_weakness_whose_rate_reaches_eta_contributes(self):
        """The second entry's rate is exactly `eta`, and a weakness that reaches it is grounded."""
        evaluations, _, _ = _run()

        assert TWO in _reported(evaluations)

    def test_a_weakness_attribute_carries_the_entry_that_induced_it(self):
        evaluations, _, _ = _run()
        category = _reported(evaluations)[TWO].category

        assert category.attributes == [ATTRIBUTE_TWO]
        assert category.provenance == [PROVENANCE_TWO]

    def test_a_retained_hook_contributes_the_entity_it_leans_on(self):
        evaluations, _, _ = _run()
        category = _reported(evaluations)[HOOK].category

        assert category.attributes == [ATTRIBUTE_HOOK]
        assert category.provenance == [PROVENANCE_HOOK]

    def test_one_behaviour_described_under_two_principles_is_grounded_once(self):
        """The same attribute twice in one conjunction says nothing the once did not.

        Read off the conjunction rather than off the seeds: two entries with the same descriptor
        carry different provenance, so they survive deduplication by attribute object and collapse
        only if the search deduplicates on the text.
        """
        evaluations, _, _ = _run(grades=UNBREAKING_GRADES)
        category = _reported(evaluations)[ALL_THREE].category

        assert category.attributes == list(ALL_THREE)
        assert category.provenance == [PROVENANCE_ONE, PROVENANCE_TWO, PROVENANCE_HOOK]


class TestThePool:
    def test_every_grounded_attribute_is_evaluated_on_its_own(self):
        evaluations, query_generator, _ = _run()

        assert [key for key, _ in query_generator.calls][:3] == [ONE, TWO, HOOK]
        assert set(_reported(evaluations)) >= {ONE, TWO, HOOK}

    def test_exactly_the_configured_number_of_queries_is_requested(self):
        """Returning fewer would shrink the denominator of S(c) without saying so."""
        _, query_generator, _ = _run()

        assert query_generator.calls != []
        assert all(count == QUERIES_PER_CATEGORY for _, count in query_generator.calls)

    def test_the_seeds_are_kept_even_when_none_of_them_reaches_tau(self):
        """A run whose conjunction fails has still found something; dropping the seeds would
        leave the report saying nothing happened."""
        evaluations, _, _ = _run(grades=UNBREAKING_GRADES)
        reported = _reported(evaluations)

        assert reported[TWO].score == pytest.approx(SCORE_LIGHT, abs=TOLERANCE)
        assert reported[TWO].score < TAU

    def test_the_pool_holds_pairs_and_pool_size_bounds_them(self):
        """The six pairs rank [0.8, 0.8] for one and [0.5, 0.5] each for two and the hook, so a
        pool of three holds both of one's and the first of two's, and the hook reaches no
        candidate even though it was seeded."""
        evaluations, query_generator, _ = _run(grades=UNBREAKING_GRADES, pool_size=3)

        assert ONE_TWO in _reported(evaluations)
        assert _asked(query_generator) == {ONE, TWO, HOOK, ONE_TWO}

    def test_a_pool_of_one_leaves_a_single_attribute_candidate(self):
        evaluations, query_generator, _ = _run(grades=UNBREAKING_GRADES, pool_size=1)

        assert _asked(query_generator) == {ONE, TWO, HOOK}
        assert set(_reported(evaluations)) == {ONE, TWO, HOOK}

    def test_max_attributes_caps_the_conjunction_the_full_pool_would_give(self):
        """Refinement is exhaustive over sub-conjunctions, so the candidate's length is what
        bounds the cost of a run — the pool here is whole and the cap is what shortens it."""
        evaluations, query_generator, _ = _run(grades=UNBREAKING_GRADES, max_attributes=2)

        assert ONE_TWO in _reported(evaluations)
        assert ALL_THREE not in _asked(query_generator)

    def test_a_profile_grounding_nothing_ends_the_search_without_a_target_call(self):
        evaluations, query_generator, target = _run(profile=AssistantProfile(weaknesses=[_below_eta()]))

        assert evaluations == []
        assert query_generator.calls == []
        assert target.sent == []


class TestRefinement:
    def test_the_minimal_sub_conjunction_is_what_is_reported(self):
        """FR-032. S(one) = 1.0 clears tau = 0.9 alone, so the other two are incidental."""
        evaluations, _, _ = _run()
        reported = _reported(evaluations)

        assert reported[ONE].dropped_attributes == [ATTRIBUTE_TWO, ATTRIBUTE_HOOK]
        assert reported[ONE].score == pytest.approx(SCORE_BROKEN, abs=TOLERANCE)

    def test_a_longer_sub_conjunction_is_never_evaluated_once_a_shorter_one_passes(self):
        """Shortest first is what makes the first passing sub-conjunction minimal, and it is what
        keeps the 2^l - 1 evaluations from being paid for on every run."""
        _, query_generator, _ = _run()

        assert _asked(query_generator) == {ONE, TWO, HOOK}

    def test_the_minimal_sub_conjunction_may_be_a_pair_no_seed_proposed(self):
        """Shortest first, so the singletons are exhausted before the pair that passes is found."""
        evaluations, query_generator, _ = _run(grades=PAIRED_GRADES)
        reported = _reported(evaluations)

        assert reported[ONE_TWO].dropped_attributes == [ATTRIBUTE_HOOK]
        assert reported[ONE_TWO].score == pytest.approx(SCORE_BROKEN, abs=TOLERANCE)
        assert _asked(query_generator) == {ONE, TWO, HOOK, ONE_TWO}

    def test_a_refined_conjunction_carries_one_provenance_entry_per_attribute(self):
        """FR-028 survives refinement: an attribute may never lack the entry that induced it, and
        this category existed only as a sub-conjunction of the candidate."""
        evaluations, _, _ = _run(grades=PAIRED_GRADES)

        assert _reported(evaluations)[ONE_TWO].category.provenance == [PROVENANCE_ONE, PROVENANCE_TWO]

    def test_the_seed_refinement_settled_on_is_replaced_rather_than_duplicated(self):
        """The refined category is a seed here, and the report may not carry it twice — once with
        the incidental attributes and once without."""
        evaluations, _, _ = _run()
        keys = [tuple(evaluation.category.attributes) for evaluation in evaluations]

        assert keys.count(ONE) == 1
        assert len(keys) == 3

    def test_when_no_sub_conjunction_passes_the_candidate_is_returned_whole(self):
        """`refine`'s fallback: a category no sub-conjunction of which passes — its whole self
        included — is returned unrefined, with nothing reported as incidental."""
        evaluations, _, _ = _run(grades=UNBREAKING_GRADES)
        reported = _reported(evaluations)

        assert reported[ALL_THREE].category.attributes == list(ALL_THREE)
        assert reported[ALL_THREE].dropped_attributes == []
        assert reported[ALL_THREE].score == pytest.approx(SCORE_HEAVY, abs=TOLERANCE)

    def test_the_fallback_is_reached_only_after_every_sub_conjunction_was_tried(self):
        _, query_generator, _ = _run(grades=UNBREAKING_GRADES)

        assert _asked(query_generator) == {ONE, TWO, HOOK, ONE_TWO, ONE_HOOK, TWO_HOOK, ALL_THREE}

    def test_the_sub_conjunctions_tried_along_the_way_are_not_reported(self):
        """Only the seeds and what refinement settled on are findings; the pairs it stepped
        through are evaluations nobody asked for a verdict on."""
        evaluations, _, _ = _run(grades=UNBREAKING_GRADES)

        assert set(_reported(evaluations)) == {ONE, TWO, HOOK, ALL_THREE}

    def test_refinement_re_evaluates_no_seed(self):
        """The evaluator's cache is what stops refinement paying a second time in target calls
        for the singletons the search has already sent."""
        _, query_generator, _ = _run(grades=UNBREAKING_GRADES)

        assert [key for key, _ in query_generator.calls if len(key) == 1] == [ONE, TWO, HOOK]


class TestTheGatesReachTheEvaluation:
    def test_a_category_over_delta_never_reaches_the_target(self):
        """FR-031: the realism gap is computed without querying the assistant, so an unrealistic
        category costs nothing to reject."""
        _, _, target = _run(over_delta=(HOOK,))
        sent = {query for query, _ in target.sent}

        assert sent.isdisjoint(QUERIES[HOOK])
        assert sent.issuperset(QUERIES[ONE])

    def test_a_category_over_delta_is_neither_reported_nor_pooled(self):
        """It produced no pair, so it reached no candidate either — which is visible in what
        refinement had left to drop."""
        evaluations, _, _ = _run(over_delta=(HOOK,))
        reported = _reported(evaluations)

        assert HOOK not in reported
        assert reported[ONE].dropped_attributes == [ATTRIBUTE_TWO]

    def test_a_candidate_conjunction_over_delta_leaves_the_run_on_its_seeds(self):
        """Three attributes each realistic on their own can read as unnatural conjoined. Nothing
        shorter passed either, so the fallback returns the candidate whole — and the whole was
        never asked, so there is no evaluation of it to report."""
        evaluations, _, target = _run(grades=UNBREAKING_GRADES, over_delta=(ALL_THREE,))
        sent = {query for query, _ in target.sent}

        assert set(_reported(evaluations)) == {ONE, TWO, HOOK}
        assert sent.isdisjoint(QUERIES[ALL_THREE])

    def test_a_query_below_kappa_is_not_sent_and_contributes_exactly_zero(self):
        """FR-030, and the zero stays explainable through `on_profile`."""
        evaluations, _, target = _run(gated=(GATED_QUERY,))
        evaluation = _reported(evaluations)[TWO]

        assert GATED_QUERY not in {query for query, _ in target.sent}
        assert evaluation.violations == GATED_VIOLATIONS
        assert evaluation.on_profile == [True, False]

    def test_the_gated_query_stays_in_the_denominator(self):
        """The zero costs the category the mean and the dispersion it creates, and `n` travelling
        with the score is what keeps the sample visible (FR-029)."""
        evaluations, _, _ = _run(gated=(GATED_QUERY,))
        evaluation = _reported(evaluations)[TWO]

        assert evaluation.n == QUERIES_PER_CATEGORY
        assert evaluation.score == pytest.approx(GATED_SCORE, abs=TOLERANCE)
        assert evaluation.score == pytest.approx(0.4 - GATED_SE, abs=TOLERANCE)

    def test_a_seed_whose_every_exchange_the_target_dropped_is_not_reported(self):
        """FR-016: an ungraded exchange may move neither the numerator nor the denominator. The
        target was reached here, unlike the category `delta` rejected."""
        failures = dict.fromkeys(QUERIES[HOOK], "gateway timeout")
        evaluations, _, target = _run(failures=failures)
        sent = {query for query, _ in target.sent}

        assert sent.issuperset(QUERIES[HOOK])
        assert HOOK not in _reported(evaluations)

    def test_a_config_whose_thresholds_were_never_resolved_refuses_to_run(self):
        """FR-041: `kappa` reaches a search already resolved against the configured component.
        One that carries none would complete the run while gating nothing."""
        with pytest.raises(ValueError, match="kappa"):
            _run(kappa=None)
