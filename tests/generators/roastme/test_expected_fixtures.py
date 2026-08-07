"""The hand-computed fixtures are re-derived from the formulas (T017).

Everything the rest of the suite asserts against lives in `tests/fixtures/roastme/expected.py` as
a literal. A slip in one of those literals would make a wrong implementation look right, so each
one is recomputed here from the formula quoted in its comment — using arithmetic written out in
this module rather than anything imported from the library, which is what keeps the check from
being circular.

It also pins the two facts the fixtures are *for*: that the expected weakness triples follow from
the graded probe set rather than being asserted alongside it, and that the minimal sub-conjunction
the refinement fixture expects is the unique minimum of its own evaluation table.

This module needs no implementation, so it runs from the red phase onward. That is the point: the
numbers have to be known good before anything is written against them.
"""

import math
from itertools import combinations

import pytest

from tests.fixtures.roastme import expected as fx

TOLERANCE = 1e-9

WEIGHTS = {fx.PRINCIPLE_A: fx.WEIGHT_A, fx.PRINCIPLE_B: fx.WEIGHT_B, fx.PRINCIPLE_C: fx.WEIGHT_C}

GROUP_ONE_PROBES = ("pb-1", "pb-2", "pb-3", "pb-4")
GROUP_TWO_PROBES = ("pb-5", "pb-6")


def _violation(grades: dict[str, float]) -> float:
    return sum(WEIGHTS[principle] * score for principle, score in grades.items())


def _standard_error(values) -> float:
    """se = sqrt( sum (x - mean)^2 / n ) / sqrt(n), on the uncorrected variance."""
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((value - mean) ** 2 for value in values) / n) / math.sqrt(n)


def _score(values, lambda_: float) -> float:
    return sum(values) / len(values) - lambda_ * _standard_error(values)


def _grades_for(query: str) -> dict[str, float]:
    return {principle: fx.GRADES[(query, principle)] for principle in WEIGHTS}


class TestTheContract:
    def test_the_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)

    def test_the_weights_are_distinct_so_a_count_and_a_sum_disagree(self):
        assert len(set(WEIGHTS.values())) == len(WEIGHTS)


class TestViolationLiterals:
    @pytest.mark.parametrize("query", sorted(fx.EXPECTED_VIOLATION))
    def test_each_expected_violation_follows_from_the_grades(self, query):
        assert _violation(_grades_for(query)) == pytest.approx(fx.EXPECTED_VIOLATION[query], abs=TOLERANCE)

    def test_the_partial_violation_case(self):
        assert _violation(fx.PARTIAL_VIOLATION_GRADES) == pytest.approx(fx.PARTIAL_VIOLATION, abs=TOLERANCE)

    def test_the_partial_case_really_splits_a_count_from_a_sum(self):
        """SC-002 only means something if the naive count clears the threshold the sum fails."""
        violated = sum(1 for score in fx.PARTIAL_VIOLATION_GRADES.values() if score > 0.0)
        assert violated / len(fx.PARTIAL_VIOLATION_GRADES) == pytest.approx(
            fx.PARTIAL_NAIVE_COUNT_FRACTION, abs=TOLERANCE
        )
        assert fx.PARTIAL_NAIVE_COUNT_FRACTION >= fx.PARTIAL_THRESHOLD
        assert fx.PARTIAL_VIOLATION < fx.PARTIAL_THRESHOLD

    def test_the_fractional_case(self):
        assert _violation(fx.FRACTIONAL_GRADES) == pytest.approx(fx.FRACTIONAL_VIOLATION, abs=TOLERANCE)

    def test_the_blackbox_case(self):
        grades = {principle: fx.BLACKBOX_GRADES[(fx.BLACKBOX_QUERY, principle)] for principle in WEIGHTS}
        assert _violation(grades) == pytest.approx(fx.BLACKBOX_VIOLATION, abs=TOLERANCE)


class TestTheProbeSet:
    def test_the_counts_follow_from_the_probes(self):
        probes = fx.probes()
        scoreable = [probe for probe in probes if probe.plugin is not None]
        graded = [probe for probe in scoreable if probe.query in fx.RESPONSES]

        assert len(probes) == fx.EXPECTED_OUTCOMES
        assert len(graded) == fx.EXPECTED_SCOREABLE
        assert len(scoreable) - len(graded) == fx.EXPECTED_UNGRADED

    def test_the_overall_rate_is_the_mean_over_the_scoreable_graded_outcomes(self):
        violations = [fx.EXPECTED_VIOLATION[query] for query in fx.SCOREABLE_QUERIES]

        assert len(violations) == fx.EXPECTED_SCOREABLE
        assert sum(violations) / len(violations) == pytest.approx(fx.EXPECTED_OVERALL_RATE, abs=TOLERANCE)

    def test_the_violating_queries_are_exactly_those_with_a_nonzero_violation(self):
        violating = [query for query in fx.SCOREABLE_QUERIES if fx.EXPECTED_VIOLATION[query] > 0.0]
        assert violating == fx.VIOLATING_QUERIES

    def test_the_control_and_one_scored_probe_share_a_documented_entity(self):
        probes = {probe.id: probe for probe in fx.probes()}
        scored, control = probes["pb-3"], probes["pb-control"]

        assert scored.hook.references == control.hook.references == fx.SHARED_ENTITY
        assert scored.hook.doc == control.hook.doc == 1
        assert scored.plugin is not None
        assert control.plugin is None

    def test_no_opaque_identifier_leaks_into_the_prose(self):
        """The descriptor assertions in the Profiler tests only bite if the identifiers could not
        have appeared in the prose by accident."""
        prose = " ".join([*fx.ATTRS_ONE, *fx.ATTRS_TWO, *fx.ATTRS_CONTROL, *fx.REFINE_ATTRIBUTES])

        for identifier in fx.OPAQUE_IDENTIFIERS:
            assert identifier not in prose


class TestWeaknessLiterals:
    def test_the_group_vectors_follow_from_the_graded_probes(self):
        probes = {probe.id: probe for probe in fx.probes()}

        for group, expected in ((GROUP_ONE_PROBES, fx.WEAKNESS_GROUP_ONE), (GROUP_TWO_PROBES, fx.WEAKNESS_GROUP_TWO)):
            for principle, vector in expected.items():
                actual = [fx.GRADES[(probes[probe_id].query, principle)] for probe_id in group]
                assert actual == vector

    def test_the_failed_exchange_is_not_in_a_group(self):
        assert "pb-failed" not in GROUP_ONE_PROBES
        assert len(GROUP_ONE_PROBES) == 4

    @pytest.mark.parametrize(
        ("literal", "rate", "n"),
        [
            (fx.SE_RATE_075_N4, 0.75, 4),
            (fx.SE_RATE_050_N4, 0.50, 4),
            (fx.SE_RATE_025_N4, 0.25, 4),
            (fx.SE_RATE_050_N2, 0.50, 2),
        ],
    )
    def test_each_standard_error_literal_matches_the_binomial_closed_form(self, literal, rate, n):
        assert literal == pytest.approx(math.sqrt(rate * (1.0 - rate) / n), abs=1e-15)

    def test_the_expected_triples_follow_from_the_group_vectors(self):
        derived = []
        for group in (fx.WEAKNESS_GROUP_ONE, fx.WEAKNESS_GROUP_TWO):
            for principle, vector in group.items():
                derived.append((principle, sum(vector) / len(vector), len(vector), _standard_error(vector)))

        derived.sort(key=lambda entry: (entry[0], entry[2]))
        expected = sorted(fx.EXPECTED_WEAKNESSES, key=lambda entry: (entry[0], entry[2]))

        assert len(derived) == fx.EXPECTED_WEAKNESS_COUNT
        for got, want in zip(derived, expected, strict=True):
            assert got[0] == want[0]
            assert got[1] == pytest.approx(want[1], abs=TOLERANCE)
            assert got[2] == want[2]
            assert got[3] == pytest.approx(want[3], abs=TOLERANCE)

    def test_the_uncorrected_variance_and_the_binomial_form_agree_on_binary_values(self):
        """This is why one formula can serve both the weakness map and `S(c)`."""
        for group in (fx.WEAKNESS_GROUP_ONE, fx.WEAKNESS_GROUP_TWO):
            for vector in group.values():
                rate = sum(vector) / len(vector)
                assert _standard_error(vector) == pytest.approx(
                    math.sqrt(rate * (1.0 - rate) / len(vector)), abs=TOLERANCE
                )


class TestCategoryScoreLiterals:
    def test_the_consistent_category(self):
        assert sum(fx.CONSISTENT_VIOLATIONS) / len(fx.CONSISTENT_VIOLATIONS) == pytest.approx(
            fx.CONSISTENT_MEAN, abs=TOLERANCE
        )
        assert _standard_error(fx.CONSISTENT_VIOLATIONS) == pytest.approx(fx.CONSISTENT_SE, abs=TOLERANCE)
        assert _score(fx.CONSISTENT_VIOLATIONS, 1.0) == pytest.approx(fx.CONSISTENT_SCORE_LAMBDA_1, abs=TOLERANCE)

    def test_the_spiky_category(self):
        assert sum(fx.SPIKY_VIOLATIONS) / len(fx.SPIKY_VIOLATIONS) == pytest.approx(fx.SPIKY_MEAN, abs=TOLERANCE)
        assert _standard_error(fx.SPIKY_VIOLATIONS) == pytest.approx(fx.SPIKY_SE, abs=TOLERANCE)
        assert _score(fx.SPIKY_VIOLATIONS, 1.0) == pytest.approx(fx.SPIKY_SCORE_LAMBDA_1, abs=TOLERANCE)
        assert _score(fx.SPIKY_VIOLATIONS, 2.0) == pytest.approx(fx.SPIKY_SCORE_LAMBDA_2, abs=TOLERANCE)

    def test_the_two_categories_share_a_mean_and_differ_in_dispersion(self):
        """Asserted over the two violation vectors, not over the literals they were reduced to.

        `SPIKY_MEAN` and `CONSISTENT_MEAN` are both written `0.6`, so comparing them to each
        other says nothing about whether the vectors below them still share a mean — which is the
        property US4 scenario 1 rests on.
        """
        consistent = sum(fx.CONSISTENT_VIOLATIONS) / len(fx.CONSISTENT_VIOLATIONS)
        spiky = sum(fx.SPIKY_VIOLATIONS) / len(fx.SPIKY_VIOLATIONS)

        assert spiky == pytest.approx(consistent, abs=TOLERANCE)
        assert _standard_error(fx.CONSISTENT_VIOLATIONS) < _standard_error(fx.SPIKY_VIOLATIONS)
        assert _score(fx.CONSISTENT_VIOLATIONS, 1.0) > _score(fx.SPIKY_VIOLATIONS, 1.0)

    def test_the_gated_vector_is_the_raw_one_with_the_off_profile_query_zeroed(self):
        gated = [
            violation if on_profile >= fx.GATE_KAPPA else 0.0
            for violation, on_profile in zip(fx.GATE_RAW_VIOLATIONS, fx.GATE_ON_PROFILE_SCORES, strict=True)
        ]
        assert gated == fx.GATED_VIOLATIONS

    def test_the_gate_costs_more_than_it_looks(self):
        assert _standard_error(fx.GATED_VIOLATIONS) == pytest.approx(fx.GATED_SE, abs=TOLERANCE)
        assert _score(fx.GATED_VIOLATIONS, 1.0) == pytest.approx(fx.GATED_SCORE_LAMBDA_1, abs=TOLERANCE)
        assert _score(fx.GATE_RAW_VIOLATIONS, 1.0) == pytest.approx(fx.UNGATED_SCORE_LAMBDA_1, abs=TOLERANCE)
        assert fx.GATED_SCORE_LAMBDA_1 < fx.GATED_MEAN

    def test_at_one_evaluation_the_penalty_is_zero_for_every_lambda(self):
        assert _standard_error(fx.SINGLE_VIOLATIONS) == pytest.approx(fx.SINGLE_SE, abs=TOLERANCE)
        for lambda_ in (0.0, 1.0, 5.0, 50.0):
            assert _score(fx.SINGLE_VIOLATIONS, lambda_) == pytest.approx(fx.SINGLE_SCORE, abs=TOLERANCE)
        assert pytest.approx(fx.SINGLE_MEAN, abs=TOLERANCE) == fx.SINGLE_SCORE


class TestRefinementFixture:
    def test_the_table_covers_every_sub_conjunction(self):
        subsets = {
            combination
            for size in range(1, len(fx.REFINE_ATTRIBUTES) + 1)
            for combination in combinations(fx.REFINE_ATTRIBUTES, size)
        }
        assert set(fx.REFINE_TABLE) == subsets

    def test_the_expected_result_is_the_unique_minimum_of_the_table(self):
        passing = [
            attributes
            for attributes, (score, gap) in fx.REFINE_TABLE.items()
            if score >= fx.REFINE_TAU and gap <= fx.REFINE_DELTA
        ]
        smallest = min(len(attributes) for attributes in passing)
        minimal = [attributes for attributes in passing if len(attributes) == smallest]

        assert len(minimal) == 1
        assert list(minimal[0]) == fx.REFINE_EXPECTED_ATTRIBUTES

    def test_the_dropped_attributes_are_the_complement(self):
        assert sorted(fx.REFINE_EXPECTED_ATTRIBUTES + fx.REFINE_EXPECTED_DROPPED) == sorted(fx.REFINE_ATTRIBUTES)

    def test_the_provenance_follows_the_attributes(self):
        index = fx.REFINE_ATTRIBUTES.index(fx.REFINE_EXPECTED_ATTRIBUTES[0])
        assert [fx.REFINE_PROVENANCE[index]] == fx.REFINE_EXPECTED_PROVENANCE

    def test_the_whole_conjunction_would_also_have_passed(self):
        """Refinement has to be doing work: the unrefined category is not the answer."""
        score, gap = fx.REFINE_TABLE[tuple(fx.REFINE_ATTRIBUTES)]

        assert score >= fx.REFINE_TAU
        assert gap <= fx.REFINE_DELTA
        assert fx.REFINE_EXPECTED_ATTRIBUTES != fx.REFINE_ATTRIBUTES


class TestTheRealismFixture:
    def test_the_boundary_case_is_at_delta_and_the_failing_case_is_over_it(self):
        assert pytest.approx(fx.REALISM_DELTA, abs=TOLERANCE) == fx.REALISM_GAP_WITHIN
        assert fx.REALISM_GAP_OVER > fx.REALISM_DELTA
