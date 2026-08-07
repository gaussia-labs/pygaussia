"""The Profiler over a recorded response set (T019).

Control exclusion driven purely by a strategy with no plugin, including the pair of probes over
the same documented entity that land on opposite sides of that line (FR-011, SC-003); a failed
exchange recorded ungraded and moving neither numerator nor denominator (FR-016, SC-008); the
profile crossing with prose and no strategy identifiers (FR-013); and the evidence flag every
graded outcome has to carry (FR-015).

The whole module runs offline against a target that replays recorded responses, which is the
credential-free path of FR-014 rather than a separate mode.
"""

import inspect

import pytest

from gaussia.generators.roastme.profiler import Profiler
from gaussia.schemas.roastme import Document
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import StubGrader

TOLERANCE = 1e-9


def _run(probes=None):
    grader = fx.stub_grader()
    contract = fx.contract(grader)
    target = fx.recorded_target()
    profiler = Profiler(contract=contract, target=target)
    return profiler.profile(fx.probes() if probes is None else probes), target


def _outcome(result, probe_id: str):
    matches = [outcome for outcome in result.outcomes if outcome.probe_id == probe_id]
    assert len(matches) == 1
    return matches[0]


class TestGradedRecord:
    def test_every_probe_produces_an_outcome(self):
        result, _ = _run()
        assert len(result.outcomes) == fx.EXPECTED_OUTCOMES

    def test_violation_matches_hand_computation(self):
        result, _ = _run()
        probes = {probe.id: probe for probe in fx.probes()}

        for outcome in result.outcomes:
            query = probes[outcome.probe_id].query
            if query not in fx.EXPECTED_VIOLATION:
                continue
            assert outcome.violation == pytest.approx(fx.EXPECTED_VIOLATION[query], abs=TOLERANCE)

    def test_per_principle_grades_are_retained(self):
        """FR-004: a failure has to trace to the principle it breaks."""
        result, _ = _run()
        outcome = _outcome(result, "pb-1")

        assert [grade.principle for grade in outcome.grades] == [
            fx.PRINCIPLE_A,
            fx.PRINCIPLE_B,
            fx.PRINCIPLE_C,
        ]
        assert [grade.score for grade in outcome.grades] == [1.0, 1.0, 0.0]

    def test_the_target_is_the_only_path_to_the_assistant(self):
        result, target = _run()
        assert len(target.sent) == fx.EXPECTED_OUTCOMES
        assert len(result.outcomes) == len(target.sent)


class TestControlExclusion:
    def test_a_control_is_recorded_but_not_scoreable(self):
        result, _ = _run()
        control = _outcome(result, "pb-control")

        assert control.scoreable is False
        assert control.violation == pytest.approx(fx.EXPECTED_VIOLATION[fx.QUERY_CONTROL], abs=TOLERANCE)
        assert control.grades != []

    def test_the_control_is_out_of_every_rate(self):
        result, _ = _run()

        assert result.n_scoreable == fx.EXPECTED_SCOREABLE
        assert result.overall_rate == pytest.approx(fx.EXPECTED_OVERALL_RATE, abs=TOLERANCE)

    def test_two_probes_over_one_documented_entity_land_on_opposite_sides(self):
        """SC-003: the entity is real and documented in both cases. Only the plugin decides."""
        result, _ = _run()
        scored = _outcome(result, "pb-3")
        control = _outcome(result, "pb-control")
        probes = {probe.id: probe for probe in fx.probes()}

        assert probes["pb-3"].hook.references == probes["pb-control"].hook.references
        assert probes["pb-3"].hook.doc == probes["pb-control"].hook.doc == 1
        assert probes["pb-3"].plugin is not None
        assert probes["pb-control"].plugin is None
        assert scored.scoreable is True
        assert control.scoreable is False

    def test_a_control_contributes_no_weakness_entry(self):
        result, _ = _run()
        assert len(result.profile.weaknesses) == fx.EXPECTED_WEAKNESS_COUNT


class TestFailedExchange:
    def test_a_failed_exchange_is_recorded_ungraded(self):
        """FR-016: `None` is the only representation of ungraded, so no aggregate sees it."""
        result, _ = _run()
        failed = _outcome(result, "pb-failed")

        assert failed.violation is None
        assert failed.grades == []
        assert result.n_ungraded == fx.EXPECTED_UNGRADED

    def test_it_moves_neither_numerator_nor_denominator(self):
        """SC-008: dropping the failing probe changes nothing about the rates."""
        with_failure, _ = _run()
        without_failure, _ = _run([probe for probe in fx.probes() if probe.id != "pb-failed"])

        assert with_failure.overall_rate == pytest.approx(without_failure.overall_rate, abs=TOLERANCE)
        assert with_failure.n_scoreable == without_failure.n_scoreable
        assert without_failure.n_ungraded == 0

    def test_it_contributes_no_trial_to_its_descriptor(self):
        result, _ = _run()
        group_one = [entry for entry in result.profile.weaknesses if entry.n == 4]

        assert len(group_one) == 3


class TestWeaknessMap:
    def test_rates_sample_sizes_and_standard_errors(self):
        """FR-012 and SC-001: keyed by (principle, descriptor), each with its own `n` and `se`."""
        result, _ = _run()
        actual = sorted(
            ((entry.principle, entry.rate, entry.n, entry.standard_error) for entry in result.profile.weaknesses),
            key=lambda entry: (entry[0], entry[2]),
        )
        want = sorted(fx.EXPECTED_WEAKNESSES, key=lambda entry: (entry[0], entry[2]))

        assert len(actual) == len(want)
        for got, expected in zip(actual, want, strict=True):
            assert got[0] == expected[0]
            assert got[1] == pytest.approx(expected[1], abs=TOLERANCE)
            assert got[2] == expected[2]
            assert got[3] == pytest.approx(expected[3], abs=TOLERANCE)

    def test_every_contract_principle_is_keyed(self):
        result, _ = _run()
        assert {entry.principle for entry in result.profile.weaknesses} == {
            fx.PRINCIPLE_A,
            fx.PRINCIPLE_B,
            fx.PRINCIPLE_C,
        }


class TestWhatCrossesToTheExploiter:
    def test_descriptors_carry_no_identifier(self):
        """Paper invariant 3: the profile carries readable prose, never internal identifiers."""
        result, _ = _run()

        for entry in result.profile.weaknesses:
            assert entry.descriptor.strip() != ""
            for identifier in fx.OPAQUE_IDENTIFIERS:
                assert identifier not in entry.descriptor

    def test_the_source_strategy_is_stripped(self):
        """FR-013: it exists for auditing and must not travel on what crosses."""
        result, _ = _run()
        assert all(entry.source_strategy is None for entry in result.profile.weaknesses)

    def test_the_retained_hooks_keep_their_doc_label(self):
        result, _ = _run()
        assert result.profile.hooks != []
        assert all(hook.doc in (0, 1) for hook in result.profile.hooks)

    def test_the_hooks_of_the_probes_that_broke_it_are_retained(self):
        """US1: an evaluator needs to know which specific entities broke the assistant."""
        result, _ = _run()
        probes = {probe.id: probe for probe in fx.probes()}
        retained = {hook.references for hook in result.profile.hooks}

        for probe in probes.values():
            if probe.query in fx.VIOLATING_QUERIES:
                assert probe.hook.references in retained

    def test_no_control_hook_crosses(self):
        """A control puts no principle under test, so it must not ground a category either."""
        result, _ = _run()
        assert all(hook.principle is not None for hook in result.profile.hooks)


class TestEvidenceFlag:
    def test_a_grounded_probe_records_that_evidence_existed(self):
        result, _ = _run()
        assert _outcome(result, "pb-1").evidence_available is True

    def test_a_probe_leaning_on_nothing_records_that_it_did_not(self):
        """FR-015: where the grader had nothing to check against, the score reflects the
        model's own knowledge and the record has to say so."""
        grader = StubGrader(fx.BLACKBOX_GRADES)
        contract = fx.contract(grader)
        target = fx.recorded_target()
        target.responses = fx.BLACKBOX_RESPONSES
        result = Profiler(contract=contract, target=target).profile(fx.blackbox_probes())

        outcome = _outcome(result, "pb-blackbox")
        assert outcome.evidence_available is False
        assert outcome.violation == pytest.approx(fx.BLACKBOX_VIOLATION, abs=TOLERANCE)


class TestNoKnowledgeBaseAccess:
    def test_no_profiler_entry_point_accepts_a_document(self):
        """FR-010 and paper invariant 2: the Profiler accepts probes and tags only. Nothing on
        its surface takes the corpus, so the constraint is structural rather than documented."""
        for member in (Profiler.__init__, Profiler.profile):
            annotations = [str(parameter.annotation) for parameter in inspect.signature(member).parameters.values()]
            assert not any(Document.__name__ in annotation for annotation in annotations)
