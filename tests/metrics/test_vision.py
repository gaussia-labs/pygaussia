"""Unit tests for vision metrics: VisionSimilarity, VisionHallucination."""

from unittest.mock import MagicMock, patch

import pytest

from pygaussia.metrics.vision import VisionHallucination, VisionSimilarity
from pygaussia.schemas.vision import VisionHallucinationMetric, VisionSimilarityMetric
from tests.fixtures.mock_data import create_sample_batch, create_sample_dataset
from tests.fixtures.mock_retriever import MockRetriever, VisionDatasetRetriever


def _mock_scorer(similarities: list[float]):
    scorer = MagicMock()
    scorer.calculate.side_effect = similarities
    return scorer


def _vision_retriever(batches):
    dataset = create_sample_dataset(
        session_id="vision_session",
        assistant_id="argos_vlm",
        context="Argos security camera",
        conversation=batches,
    )
    return type("VisionRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})


def _batch(qa_id: str):
    return create_sample_batch(
        qa_id=qa_id,
        assistant="VLM description",
        ground_truth_assistant="Ground truth description",
    )


def _run_metric(metric_class, batches, scorer, threshold=0.75):
    retriever = _vision_retriever(batches)
    metric = metric_class(retriever, scorer=scorer, threshold=threshold)
    metric.batch(session_id="vision_session", context="camera", assistant_id="vlm", batch=batches)
    metric.on_process_complete()
    return metric


class TestVisionSimilarity:
    def test_initialization(self, vision_dataset_retriever):
        scorer = _mock_scorer([])
        metric = VisionSimilarity(vision_dataset_retriever, scorer=scorer)
        assert metric._session_data == {}
        assert metric._threshold == 0.75

    def test_perfect_similarity(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionSimilarity, batches, _mock_scorer([1.0, 1.0]))

        result = metric.metrics[0]
        assert isinstance(result, VisionSimilarityMetric)
        assert result.mean_similarity == pytest.approx(1.0)
        assert result.min_similarity == pytest.approx(1.0)
        assert result.max_similarity == pytest.approx(1.0)

    def test_mixed_similarity(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionSimilarity, batches, _mock_scorer([1.0, 0.0]))

        result = metric.metrics[0]
        assert result.mean_similarity == pytest.approx(0.5, abs=0.01)
        assert result.min_similarity == pytest.approx(0.0, abs=0.01)
        assert result.max_similarity == pytest.approx(1.0, abs=0.01)

    def test_summary_contains_key_info(self):
        batches = [_batch("qa_001")]
        metric = _run_metric(VisionSimilarity, batches, _mock_scorer([1.0]))

        summary = metric.metrics[0].summary
        assert "100%" in summary
        assert "1 frames" in summary

    def test_interactions_stored(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionSimilarity, batches, _mock_scorer([1.0, 0.0]))

        interactions = metric.metrics[0].interactions
        assert len(interactions) == 2
        assert interactions[0].qa_id == "qa_001"
        assert interactions[0].similarity_score is not None

    def test_multiple_sessions(self):
        batches = [_batch("qa_001")]
        dataset_a = create_sample_dataset(session_id="session_a", conversation=batches)
        dataset_b = create_sample_dataset(session_id="session_b", conversation=batches)
        retriever = type("R", (MockRetriever,), {"load_dataset": lambda self: [dataset_a, dataset_b]})

        scorer = _mock_scorer([1.0, 1.0])
        metric = VisionSimilarity(retriever, scorer=scorer)
        metric.batch(session_id="session_a", context="cam", assistant_id="vlm", batch=batches)
        metric.batch(session_id="session_b", context="cam", assistant_id="vlm", batch=batches)
        metric.on_process_complete()

        assert len(metric.metrics) == 2

    def test_run_method(self, vision_dataset_retriever):
        scorer = _mock_scorer([1.0, 1.0, 1.0, 1.0])
        results = VisionSimilarity.run(vision_dataset_retriever, scorer=scorer, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], VisionSimilarityMetric)


class TestVisionHallucination:
    def test_initialization(self, vision_dataset_retriever):
        scorer = _mock_scorer([])
        metric = VisionHallucination(vision_dataset_retriever, scorer=scorer)
        assert metric._session_data == {}
        assert metric._threshold == 0.75

    def test_no_hallucinations(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([1.0, 1.0]))

        result = metric.metrics[0]
        assert isinstance(result, VisionHallucinationMetric)
        assert result.n_hallucinations == 0
        assert result.hallucination_rate == pytest.approx(0.0)

    def test_all_hallucinations(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([0.0, 0.0]))

        result = metric.metrics[0]
        assert result.n_hallucinations == 2
        assert result.hallucination_rate == pytest.approx(1.0)

    def test_partial_hallucinations(self):
        batches = [_batch("qa_001"), _batch("qa_002"), _batch("qa_003")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([1.0, 1.0, 0.0]))

        result = metric.metrics[0]
        assert result.n_hallucinations == 1
        assert result.n_frames == 3
        assert result.hallucination_rate == pytest.approx(1 / 3, abs=0.01)

    def test_configurable_threshold(self):
        batches = [_batch("qa_001")]
        # similarity = 0.0, threshold = 0.0 → 0.0 < 0.0 is False → not a hallucination
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([0.0]), threshold=0.0)

        assert metric.metrics[0].n_hallucinations == 0

    def test_threshold_stored_in_result(self):
        batches = [_batch("qa_001")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([1.0]), threshold=0.8)

        assert metric.metrics[0].threshold == 0.8

    def test_summary_contains_key_info(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([1.0, 0.0]))

        summary = metric.metrics[0].summary
        assert "1 of 2" in summary
        assert "0.75" in summary

    def test_interactions_classified_correctly(self):
        batches = [_batch("qa_001"), _batch("qa_002")]
        metric = _run_metric(VisionHallucination, batches, _mock_scorer([1.0, 0.0]))

        interactions = metric.metrics[0].interactions
        assert interactions[0].is_hallucination is False
        assert interactions[1].is_hallucination is True

    def test_run_method(self, vision_dataset_retriever):
        scorer = _mock_scorer([1.0, 1.0, 1.0, 1.0])
        results = VisionHallucination.run(vision_dataset_retriever, scorer=scorer, verbose=False)

        assert isinstance(results, list)
        assert isinstance(results[0], VisionHallucinationMetric)
