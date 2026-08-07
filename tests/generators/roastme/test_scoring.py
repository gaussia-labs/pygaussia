"""Arithmetic of the Roast Me scoring module (T018).

Every expectation is a hand-computed literal from `tests/fixtures/roastme/expected.py`, with
the derivation beside it there. The tolerance is SC-001's `1e-9` throughout.

Covers `v` with its per-principle grades retained (FR-004), the weakness-map rate with its
sample size and standard error (FR-012), `S(c)` (FR-029), the `kappa` gate zeroing a query
(FR-030), the `delta` budget (FR-031) and refinement to the minimal sub-conjunction (FR-032)
— so SC-001, SC-002 and the arithmetic half of SC-007.

The literals themselves are re-derived from their closed forms in `test_expected_fixtures.py`,
so a slip there is caught independently of whether the library is right.
"""

import math

import pytest

from gaussia.generators.roastme.searches.scoring import (
    GATED_CONTRIBUTION,
    category_score,
    is_on_profile,
    refine,
    standard_error,
    violation_score,
    weakness_entry,
    within_realism_budget,
)
from gaussia.schemas.roastme import Category, GradedOutcome, PrincipleGrade
from tests.fixtures.roastme import expected as fx

TOLERANCE = 1e-9


def _grades(scores: dict[str, float]) -> list[PrincipleGrade]:
    return [
        PrincipleGrade(principle=principle, score=score, grader="StubGrader", method="stub", model=None)
        for principle, score in scores.items()
    ]


def _contract():
    return fx.contract(fx.stub_grader())


class TestViolationScore:
    def test_weighted_sum_over_every_principle(self):
        contract = _contract()
        for query, violation in fx.EXPECTED_VIOLATION.items():
            grades = _grades({principle: fx.GRADES[(query, principle)] for principle in _principle_ids(contract)})
            assert violation_score(grades, contract) == pytest.approx(violation, abs=TOLERANCE)

    def test_fractional_grades(self):
        contract = _contract()
        grades = _grades(fx.FRACTIONAL_GRADES)
        assert violation_score(grades, contract) == pytest.approx(fx.FRACTIONAL_VIOLATION, abs=TOLERANCE)

    def test_partial_violation_falls_below_a_threshold_a_naive_count_clears(self):
        """SC-002: two of three principles violated, but not the two that carry the weight."""
        contract = _contract()
        score = violation_score(_grades(fx.PARTIAL_VIOLATION_GRADES), contract)

        assert score == pytest.approx(fx.PARTIAL_VIOLATION, abs=TOLERANCE)
        assert score < fx.PARTIAL_THRESHOLD
        assert fx.PARTIAL_NAIVE_COUNT_FRACTION >= fx.PARTIAL_THRESHOLD

    def test_grades_are_retained_alongside_the_aggregate(self):
        """FR-004: the aggregate never replaces the per-principle grades it came from."""
        contract = _contract()
        grades = _grades(fx.PARTIAL_VIOLATION_GRADES)
        violation_score(grades, contract)

        assert [grade.principle for grade in grades] == list(fx.PARTIAL_VIOLATION_GRADES)
        assert [grade.score for grade in grades] == list(fx.PARTIAL_VIOLATION_GRADES.values())

    def test_a_contract_principle_with_no_grade_raises(self):
        """FR-003, FR-004: a missing grade would contribute a silent zero to `v` instead."""
        contract = _contract()
        grades = _grades({fx.PRINCIPLE_A: 1.0, fx.PRINCIPLE_B: 1.0})

        with pytest.raises(ValueError, match=fx.PRINCIPLE_C):
            violation_score(grades, contract)

    def test_a_principle_graded_twice_raises(self):
        """The other half of the same rule: `v` may not depend on the order of the list.

        A silent last-wins would make one response score differently depending on which of two
        disagreeing grades happened to be appended last — the same silent-zero failure mode as a
        missing grade, from the opposite direction.
        """
        contract = _contract()
        grades = [
            *_grades(dict.fromkeys(_principle_ids(contract), 1.0)),
            PrincipleGrade(principle=fx.PRINCIPLE_A, score=0.0, grader="StubGrader", method="stub", model=None),
        ]

        with pytest.raises(ValueError, match=fx.PRINCIPLE_A):
            violation_score(grades, contract)

    def test_a_contract_at_the_edge_of_the_weight_tolerance_cannot_score_above_one(self):
        """FR-001 accepts weights summing to `1 +- 1e-9`; every score field is bounded by 1.0.

        The tolerance exists so a contract assembled from decimals is not rejected for float
        noise, not so `v` may leave `[0, 1]`. Clamped where it is produced, so the outcome that
        carries it is constructible rather than raising on a legitimate contract.
        """
        contract = fx.tolerance_edge_contract(fx.stub_grader())
        grades = _grades(dict.fromkeys(_principle_ids(contract), 1.0))

        assert fx.EDGE_WEIGHT_SUM > 1.0
        score = violation_score(grades, contract)

        assert score == 1.0
        assert GradedOutcome(probe_id="pb-edge", response="r", grades=grades, violation=score).violation == 1.0


def _principle_ids(contract) -> list[str]:
    return [principle.id for principle in contract.principles]


class TestStandardError:
    def test_no_values_at_all_raises(self):
        """`S(c)` over nothing is not zero, and a zero penalty on an empty category would read
        as a category that failed consistently."""
        with pytest.raises(ValueError, match="at least one value"):
            standard_error([])

    def test_zero_dispersion_gives_zero(self):
        assert standard_error([0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.0, abs=TOLERANCE)

    def test_single_value_gives_zero_by_construction(self):
        """The fact FR-040 cites for the `ge=2` floor: no deviation, so no penalty."""
        assert standard_error(fx.SINGLE_VIOLATIONS) == pytest.approx(fx.SINGLE_SE, abs=TOLERANCE)

    def test_matches_the_binomial_form_on_binary_values(self):
        """The coincidence, pinned so it is not mistaken for the definition.

        On binary values the uncorrected standard error of the mean equals `sqrt(p(1-p)/n)`, which
        is why the weakness map's binary fixtures can be read either way. It is a coincidence and
        not the specification: `v = sum(w_j * pi_j)` is continuous — a logistic grade, a vote
        fraction, or a weighted sum over several principles — and the two forms part company as
        soon as it is. `test_zero_dispersion_gives_zero` above is what holds that line.
        """
        for values in (
            fx.WEAKNESS_GROUP_ONE[fx.PRINCIPLE_A],
            fx.WEAKNESS_GROUP_ONE[fx.PRINCIPLE_B],
            fx.WEAKNESS_GROUP_ONE[fx.PRINCIPLE_C],
            fx.WEAKNESS_GROUP_TWO[fx.PRINCIPLE_B],
        ):
            rate = sum(values) / len(values)
            binomial = math.sqrt(rate * (1.0 - rate) / len(values))
            assert standard_error(values) == pytest.approx(binomial, abs=TOLERANCE)


class TestWeaknessEntry:
    @pytest.mark.parametrize(
        ("principle", "rate", "standard_error_value"),
        [
            (fx.PRINCIPLE_A, 0.75, fx.SE_RATE_075_N4),
            (fx.PRINCIPLE_B, 0.5, fx.SE_RATE_050_N4),
            (fx.PRINCIPLE_C, 0.25, fx.SE_RATE_025_N4),
        ],
    )
    def test_group_one(self, principle, rate, standard_error_value):
        entry = weakness_entry(principle, "descriptor prose", fx.WEAKNESS_GROUP_ONE[principle])

        assert entry.principle == principle
        assert entry.rate == pytest.approx(rate, abs=TOLERANCE)
        assert entry.n == 4
        assert entry.standard_error == pytest.approx(standard_error_value, abs=TOLERANCE)

    @pytest.mark.parametrize(
        ("principle", "rate", "standard_error_value"),
        [
            (fx.PRINCIPLE_A, 0.0, 0.0),
            (fx.PRINCIPLE_B, 0.5, fx.SE_RATE_050_N2),
            (fx.PRINCIPLE_C, 1.0, 0.0),
        ],
    )
    def test_group_two(self, principle, rate, standard_error_value):
        entry = weakness_entry(principle, "descriptor prose", fx.WEAKNESS_GROUP_TWO[principle])

        assert entry.rate == pytest.approx(rate, abs=TOLERANCE)
        assert entry.n == 2
        assert entry.standard_error == pytest.approx(standard_error_value, abs=TOLERANCE)

    def test_a_rate_over_tolerance_edge_violations_stays_in_range(self):
        """The `WeaknessEntry.rate` half of the same bound as `GradedOutcome.violation`.

        A rate is a mean of violation scores, so it inherits whatever `v` is allowed to be: clamp
        `v` at production and the mean of a hundred of them is still in `[0, 1]`.
        """
        contract = fx.tolerance_edge_contract(fx.stub_grader())
        grades = _grades(dict.fromkeys(_principle_ids(contract), 1.0))
        violations = [violation_score(grades, contract)] * 4

        assert weakness_entry(fx.PRINCIPLE_A, "descriptor prose", violations).rate == 1.0

    @pytest.mark.parametrize(
        ("grades", "rate", "standard_error_value"),
        [
            (fx.WEAKNESS_FLAT_GRADES, fx.WEAKNESS_FLAT_RATE, fx.WEAKNESS_FLAT_SE),
            (fx.WEAKNESS_SPREAD_GRADES, fx.WEAKNESS_SPREAD_RATE, fx.WEAKNESS_SPREAD_SE),
            (fx.WEAKNESS_SINGLE_GRADES, fx.WEAKNESS_SINGLE_RATE, fx.WEAKNESS_SINGLE_SE),
        ],
    )
    def test_fractional_grades_pin_the_general_form_of_the_rate_and_its_error(self, grades, rate, standard_error_value):
        """The vectors above are binary, where a proportion's error coincides with the mean's.

        Graders return `pi_j in [0,1]` — a logistic probability or a vote fraction — so the binary
        vectors cannot tell the two forms apart. These three sit at one rate and one `n` so that
        `sqrt(rate(1-rate)/n)` is a constant across them and only the mean's form tracks the
        dispersion. `omega` is a mean of grades, not a count of them.
        """
        entry = weakness_entry(fx.PRINCIPLE_A, "descriptor prose", grades)

        assert entry.rate == pytest.approx(rate, abs=TOLERANCE)
        assert entry.standard_error == pytest.approx(standard_error_value, abs=TOLERANCE)

    def test_sample_size_travels_with_the_rate(self):
        """A descriptor resting on a handful of probes must not read as settled (FR-012)."""
        few = weakness_entry(fx.PRINCIPLE_A, "descriptor prose", [1.0, 1.0])
        many = weakness_entry(fx.PRINCIPLE_A, "descriptor prose", [1.0] * 40)

        assert few.rate == pytest.approx(many.rate, abs=TOLERANCE)
        assert few.n == 2
        assert many.n == 40


class TestCategoryScore:
    def test_consistent_category(self):
        assert category_score(fx.CONSISTENT_VIOLATIONS, 1.0) == pytest.approx(
            fx.CONSISTENT_SCORE_LAMBDA_1, abs=TOLERANCE
        )

    def test_spiky_category(self):
        assert category_score(fx.SPIKY_VIOLATIONS, 1.0) == pytest.approx(fx.SPIKY_SCORE_LAMBDA_1, abs=TOLERANCE)

    def test_equal_mean_lower_variance_ranks_higher(self):
        """US4 scenario 1: consistent failure outranks a lucky spike."""
        consistent_mean = sum(fx.CONSISTENT_VIOLATIONS) / len(fx.CONSISTENT_VIOLATIONS)
        spiky_mean = sum(fx.SPIKY_VIOLATIONS) / len(fx.SPIKY_VIOLATIONS)
        assert consistent_mean == pytest.approx(spiky_mean, abs=TOLERANCE)

        assert category_score(fx.CONSISTENT_VIOLATIONS, 1.0) > category_score(fx.SPIKY_VIOLATIONS, 1.0)

    def test_lambda_scales_the_penalty(self):
        assert category_score(fx.SPIKY_VIOLATIONS, 2.0) == pytest.approx(fx.SPIKY_SCORE_LAMBDA_2, abs=TOLERANCE)
        assert category_score(fx.CONSISTENT_VIOLATIONS, 2.0) == pytest.approx(fx.CONSISTENT_MEAN, abs=TOLERANCE)

    def test_single_evaluation_degenerates_to_the_unpenalised_mean(self):
        """SC-013: the fixture behind the `ge=2` floor. The penalty stops existing at n = 1."""
        mean = sum(fx.SINGLE_VIOLATIONS) / len(fx.SINGLE_VIOLATIONS)

        assert mean == pytest.approx(fx.SINGLE_MEAN, abs=TOLERANCE)
        for lambda_ in (0.0, 1.0, 5.0, 50.0):
            assert category_score(fx.SINGLE_VIOLATIONS, lambda_) == pytest.approx(mean, abs=TOLERANCE)


class TestOnProfileGate:
    def test_a_query_below_kappa_contributes_exactly_zero(self):
        """The two pieces `evaluation.py` composes, over the hand-computed gate fixture (FR-030).

        The rule has one home: the gate is `is_on_profile` and the contribution is
        `GATED_CONTRIBUTION`, and the live composition of the two is pinned end-to-end in
        `test_policy_gradient.py`.
        """
        gated = [
            violation if is_on_profile(on_profile, fx.GATE_KAPPA) else GATED_CONTRIBUTION
            for violation, on_profile in zip(fx.GATE_RAW_VIOLATIONS, fx.GATE_ON_PROFILE_SCORES, strict=True)
        ]

        assert gated == fx.GATED_VIOLATIONS
        assert gated[1] == 0.0
        assert GATED_CONTRIBUTION == 0.0

    def test_the_gate_changes_the_category_score(self):
        assert category_score(fx.GATE_RAW_VIOLATIONS, 1.0) == pytest.approx(fx.UNGATED_SCORE_LAMBDA_1, abs=TOLERANCE)
        assert category_score(fx.GATED_VIOLATIONS, 1.0) == pytest.approx(fx.GATED_SCORE_LAMBDA_1, abs=TOLERANCE)

    def test_a_query_at_kappa_is_on_profile(self):
        """FR-030 gates a query *below* kappa, so equality is inside the gate."""
        assert is_on_profile(fx.GATE_KAPPA, fx.GATE_KAPPA) is True
        assert is_on_profile(fx.GATE_KAPPA - 1e-9, fx.GATE_KAPPA) is False

    def test_the_gate_is_read_on_the_filter_s_own_scale(self):
        """A kappa calibrated for one scale must not be applied to another; the comparison
        itself is scale-free, which is why the threshold travels with the component."""
        assert is_on_profile(60.0, 50.0) is True
        assert is_on_profile(0.6, 50.0) is False


class TestRealismBudget:
    def test_a_category_at_delta_survives(self):
        assert within_realism_budget(fx.REALISM_GAP_WITHIN, fx.REALISM_DELTA) is True

    def test_a_category_over_delta_is_out(self):
        assert within_realism_budget(fx.REALISM_GAP_OVER, fx.REALISM_DELTA) is False

    def test_the_budget_does_not_depend_on_the_score(self):
        """FR-031: over delta is discarded regardless of `S(c)`, so the two are independent."""
        assert within_realism_budget(fx.REALISM_GAP_OVER, fx.REALISM_DELTA) is False
        assert category_score(fx.CONSISTENT_VIOLATIONS, 1.0) > fx.REFINE_TAU


class TestRefinement:
    def test_returns_the_minimal_sub_conjunction(self):
        refined, dropped = refine(
            fx.refine_category(),
            lambda candidate: fx.REFINE_TABLE[tuple(candidate.attributes)],
            fx.REFINE_TAU,
            fx.REFINE_DELTA,
        )

        assert refined.attributes == fx.REFINE_EXPECTED_ATTRIBUTES
        assert sorted(dropped) == sorted(fx.REFINE_EXPECTED_DROPPED)

    def test_carries_the_provenance_of_what_it_kept(self):
        """FR-028: an attribute can never lack the entry that induced it, refinement included."""
        refined, _ = refine(
            fx.refine_category(),
            lambda candidate: fx.REFINE_TABLE[tuple(candidate.attributes)],
            fx.REFINE_TAU,
            fx.REFINE_DELTA,
        )

        assert refined.provenance == fx.REFINE_EXPECTED_PROVENANCE

    def test_the_kept_sub_conjunction_satisfies_both_thresholds(self):
        refined, _ = refine(
            fx.refine_category(),
            lambda candidate: fx.REFINE_TABLE[tuple(candidate.attributes)],
            fx.REFINE_TAU,
            fx.REFINE_DELTA,
        )
        score, gap = fx.REFINE_TABLE[tuple(refined.attributes)]

        assert score >= fx.REFINE_TAU
        assert gap <= fx.REFINE_DELTA

    def test_a_category_no_sub_conjunction_of_which_passes_is_left_whole(self):
        """FR-032: nothing came off, so nothing may be reported as incidental.

        Reachable in production: `attribute_iteration.py` refines its candidate against the same
        thresholds the search proposed it under, and a candidate the realism budget rejects fails
        every sub-conjunction including itself.
        """
        refined, dropped = refine(
            fx.refine_category(),
            lambda candidate: (0.0, 1.0),
            fx.REFINE_TAU,
            fx.REFINE_DELTA,
        )

        assert refined.attributes == fx.REFINE_ATTRIBUTES
        assert refined.provenance == fx.REFINE_PROVENANCE
        assert dropped == []

    def test_a_category_whose_only_passing_form_is_itself_is_left_whole(self):
        category = Category(attributes=["only attribute"], provenance=["weakness one"])
        refined, dropped = refine(
            category,
            lambda candidate: (0.9, 0.1) if candidate.attributes == ["only attribute"] else (0.0, 1.0),
            fx.REFINE_TAU,
            fx.REFINE_DELTA,
        )

        assert refined.attributes == ["only attribute"]
        assert dropped == []
