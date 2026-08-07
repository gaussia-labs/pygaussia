"""The output boundary: Roast Dataset records into the framework's dataset shapes (T021).

`data-model.md` states what every required framework field is filled with, so the conversion is
asserted field by field rather than by shape. Two of those fills carry a requirement:
`ground_truth_assistant` is `""` because a trap has no correct answer, and `evidence_available`
survives the crossing because "no evidence existed" and "evidence was sought and not found" are
different findings that a single `None` would collapse (FR-034, US2).

SC-006 names `Toxicity` as the consuming metric for a reason: it reads the assistant's answer
alone, where `Humanity` and `Vision` score against the expected answer this conversion leaves
empty.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaussia.core.extractor import BaseGroupExtractor
from gaussia.core.loader import ToxicityLoader
from gaussia.core.retriever import Retriever
from gaussia.generators.roastme.dataset import to_dataset, to_record
from gaussia.generators.roastme.profiler import Profiler
from gaussia.schemas.common import Dataset
from gaussia.schemas.roastme import RoastBatch
from gaussia.schemas.toxicity import GroupDetection, ToxicityDataset, ToxicityMetric
from tests.fixtures.roastme import expected as fx

TOLERANCE = 1e-9

SESSION_ID = "roast-run-1"
ASSISTANT_ID = "assistant-under-test"
CONTEXT = "adversarial probe run over the fixture knowledge base"


def _profile():
    grader = fx.stub_grader()
    contract = fx.contract(grader)
    return Profiler(contract=contract, target=fx.recorded_target()).profile(fx.probes())


def _dataset(language: str | None = None) -> Dataset:
    result = _profile()
    kwargs = {} if language is None else {"language": language}
    return to_dataset(
        fx.probes(),
        result.outcomes,
        session_id=SESSION_ID,
        assistant_id=ASSISTANT_ID,
        context=CONTEXT,
        **kwargs,
    )


def _batch(dataset: Dataset, qa_id: str) -> RoastBatch:
    matches = [turn for turn in dataset.conversation if turn.qa_id == qa_id]
    assert len(matches) == 1
    assert isinstance(matches[0], RoastBatch)
    return matches[0]


class TestRecordConversion:
    def test_a_record_carries_the_score_and_the_grades_behind_it(self):
        result = _profile()
        probes = {probe.id: probe for probe in fx.probes()}
        outcome = next(item for item in result.outcomes if item.probe_id == "pb-4")
        record = to_record(probes["pb-4"], outcome)

        assert record.query == probes["pb-4"].query
        assert record.response == outcome.response
        assert record.violation == pytest.approx(fx.EXPECTED_VIOLATION[fx.QUERY_FOUR], abs=TOLERANCE)
        assert record.rationale == outcome.grades

    def test_every_charged_principle_has_a_grade_behind_it(self):
        result = _profile()
        probes = {probe.id: probe for probe in fx.probes()}

        for outcome in result.outcomes:
            record = to_record(probes[outcome.probe_id], outcome)
            graded = {grade.principle for grade in record.rationale}
            assert set(record.principles_charged) <= graded

    def test_a_response_violating_every_principle_charges_every_principle(self):
        result = _profile()
        probes = {probe.id: probe for probe in fx.probes()}
        outcome = next(item for item in result.outcomes if item.probe_id == "pb-4")
        record = to_record(probes["pb-4"], outcome)

        assert set(record.principles_charged) == {fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C}

    def test_an_ungraded_exchange_is_still_a_record(self):
        """An ungraded exchange is part of the audit trail; what it must not do is enter a rate."""
        result = _profile()
        probes = {probe.id: probe for probe in fx.probes()}
        outcome = next(item for item in result.outcomes if item.probe_id == "pb-failed")
        record = to_record(probes["pb-failed"], outcome)

        assert record.violation is None


class TestDatasetFields:
    def test_one_dataset_per_run_with_one_turn_per_probe(self):
        dataset = _dataset()

        assert isinstance(dataset, Dataset)
        assert dataset.session_id == SESSION_ID
        assert dataset.assistant_id == ASSISTANT_ID
        assert dataset.context == CONTEXT
        assert len(dataset.conversation) == fx.EXPECTED_OUTCOMES

    def test_the_probe_identifier_becomes_the_turn_identifier(self):
        dataset = _dataset()
        assert {turn.qa_id for turn in dataset.conversation} == {probe.id for probe in fx.probes()}

    def test_query_and_answer_are_filled_from_the_exchange(self):
        dataset = _dataset()
        turn = _batch(dataset, "pb-1")

        assert turn.query == fx.QUERY_ONE
        assert turn.assistant == fx.RESPONSES[fx.QUERY_ONE]

    def test_the_expected_answer_is_left_empty(self):
        """A trap has no correct answer, and inventing one would let a metric score against it."""
        dataset = _dataset()
        assert all(turn.ground_truth_assistant == "" for turn in dataset.conversation)

    def test_the_framework_s_own_turn_weight_is_left_unset(self):
        """`Batch.weight` is the framework's aggregation weight, unrelated to principle weights."""
        dataset = _dataset()
        assert all(turn.weight is None for turn in dataset.conversation)

    def test_the_chatbot_role_is_left_unset(self):
        dataset = _dataset()
        assert dataset.chatbot_role is None

    def test_the_language_defaults_to_english_and_is_overridable(self):
        """Leaving it unset would label a Spanish corpus as English in the session metadata."""
        assert _dataset().language == "english"
        assert _dataset(language="spanish").language == "spanish"


class TestEvidenceSurvivesTheBoundary:
    def test_the_record_travels_on_the_turn(self):
        dataset = _dataset()
        turn = _batch(dataset, "pb-1")

        assert turn.roast.query == fx.QUERY_ONE
        assert turn.roast.violation == pytest.approx(fx.EXPECTED_VIOLATION[fx.QUERY_ONE], abs=TOLERANCE)
        assert turn.roast.rationale != []

    def test_evidence_available_is_carried_rather_than_inferred(self):
        """US2: absent evidence must stay distinguishable from evidence sought and not found."""
        dataset = _dataset()
        assert _batch(dataset, "pb-1").roast.evidence_available is True

    def test_a_blackbox_run_says_so_on_the_record(self):
        grader = fx.stub_grader()
        grader.scores = fx.BLACKBOX_GRADES
        contract = fx.contract(grader)
        target = fx.recorded_target()
        target.responses = fx.BLACKBOX_RESPONSES
        result = Profiler(contract=contract, target=target).profile(fx.blackbox_probes())

        dataset = to_dataset(
            fx.blackbox_probes(),
            result.outcomes,
            session_id=SESSION_ID,
            assistant_id=ASSISTANT_ID,
            context=CONTEXT,
        )
        turn = _batch(dataset, "pb-blackbox")

        assert turn.roast.evidence_available is False
        assert turn.roast.evidence is None


class _StubToxicityLoader(ToxicityLoader):
    def load(self, language: str) -> list[ToxicityDataset]:
        return [
            ToxicityDataset(word="hate", category="offensive"),
            ToxicityDataset(word="idiot", category="offensive"),
        ]


class _StubGroupExtractor(BaseGroupExtractor):
    def detect_one(self, text: str) -> dict[str, GroupDetection]:
        return {
            "male": GroupDetection(present=True, score=0.9, best_prototype="male", best_prototype_index=0),
            "female": GroupDetection(present=False, score=0.1, best_prototype="female", best_prototype_index=1),
        }

    def detect_batch(self, texts: list[str]) -> list[dict[str, GroupDetection]]:
        return [self.detect_one(text) for text in texts]


class TestAnExistingMetricConsumesIt:
    def test_toxicity_consumes_the_emitted_dataset_unchanged(self):
        """SC-006: end to end through a metric that reads the assistant's answer alone, with no
        change to that metric and no subclass of it."""
        dataset = _dataset()
        turns = len(dataset.conversation)

        class _RoastRetriever(Retriever):
            def load_dataset(self) -> list[Dataset]:
                return [dataset]

        embeddings = np.array([[0.1, 0.2]] * turns)
        embedder = MagicMock()
        embedder.encode.return_value = embeddings
        embedder.encode_query.return_value = embeddings

        umap_instance = MagicMock()
        umap_instance.fit_transform.return_value = embeddings
        hdbscan_instance = MagicMock()
        hdbscan_instance.fit_predict.return_value = np.zeros(turns, dtype=int)

        with (
            patch("gaussia.metrics.toxicity.umap") as mock_umap,
            patch("gaussia.metrics.toxicity.hdbscan") as mock_hdbscan,
        ):
            mock_umap.UMAP.return_value = umap_instance
            mock_hdbscan.HDBSCAN.return_value = hdbscan_instance

            from gaussia.metrics.toxicity import Toxicity

            metrics = Toxicity.run(
                _RoastRetriever,
                embedder=embedder,
                toxicity_loader=_StubToxicityLoader,
                group_extractor=_StubGroupExtractor(),
            )

        assert len(metrics) == 1
        assert isinstance(metrics[0], ToxicityMetric)
        assert metrics[0].session_id == SESSION_ID
