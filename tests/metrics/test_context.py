"""Tests for Context metric."""

from unittest.mock import MagicMock, patch

import pytest

from gaussia.metrics.context import Context
from gaussia.schemas.context import ContextMetric
from gaussia.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import ContextDatasetRetriever, MockRetriever


class TestContextMetric:
    """Test suite for Context metric."""

    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    def test_initialization_default(self, mock_model):
        context = Context(retriever=MockRetriever, model=mock_model)

        assert context.model == mock_model
        assert isinstance(context.statistical_mode, FrequentistMode)
        assert context.use_structured_output is False
        assert context.bos_json_clause == "```json"
        assert context.eos_json_clause == "```"

    def test_initialization_custom(self, mock_model):
        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
        )

        assert context.use_structured_output is True
        assert context.bos_json_clause == "<json>"
        assert context.eos_json_clause == "</json>"

    def test_initialization_bayesian(self, mock_model):
        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            statistical_mode=BayesianMode(mc_samples=100),
        )
        assert isinstance(context.statistical_mode, BayesianMode)

    @patch("gaussia.metrics.context.Judge")
    def test_batch_processing(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "I analyzed the context...",
            {"insight": "Good context awareness", "score": 0.85},
        )

        context = Context(retriever=MockRetriever, model=mock_model)
        batches = [create_sample_batch(qa_id="qa_001"), create_sample_batch(qa_id="qa_002")]
        context.batch(
            session_id="test_session", context="Healthcare AI context", assistant_id="test_assistant", batch=batches
        )
        context.on_process_complete()

        assert len(context.metrics) == 1
        metric = context.metrics[0]
        assert isinstance(metric, ContextMetric)
        assert metric.session_id == "test_session"
        assert metric.n_interactions == 2
        assert metric.context_awareness == pytest.approx(0.85)
        assert metric.context_awareness_ci_low is None
        assert metric.context_awareness_ci_high is None
        assert len(metric.interactions) == 2
        assert metric.interactions[0].qa_id == "qa_001"
        assert metric.interactions[0].context_awareness == pytest.approx(0.85)

    @patch("gaussia.metrics.context.Judge")
    def test_batch_session_accumulation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "ok", "score": 0.8})

        context = Context(retriever=MockRetriever, model=mock_model)
        context.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        context.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q2")])
        context.on_process_complete()

        assert len(context.metrics) == 1
        assert context.metrics[0].n_interactions == 2

    @patch("gaussia.metrics.context.Judge")
    def test_batch_multiple_sessions(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "ok", "score": 0.7})

        context = Context(retriever=MockRetriever, model=mock_model)
        context.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        context.batch(session_id="s2", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q2")])
        context.on_process_complete()

        assert len(context.metrics) == 2

    @patch("gaussia.metrics.context.Judge")
    def test_batch_with_observation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Observation analysis...", {"insight": "With observation", "score": 0.9})

        context = Context(retriever=MockRetriever, model=mock_model)
        batch = create_sample_batch(qa_id="qa_001", observation="The user seems confused")
        context.batch(session_id="test_session", context="Test context", assistant_id="test_assistant", batch=[batch])

        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch("gaussia.metrics.context.Judge")
    def test_batch_without_observation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analysis without observation...",
            {"insight": "Without observation", "score": 0.8},
        )

        context = Context(retriever=MockRetriever, model=mock_model)
        batch = create_sample_batch(qa_id="qa_001", observation=None)
        context.batch(session_id="test_session", context="Test context", assistant_id="test_assistant", batch=[batch])

        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch("gaussia.metrics.context.Judge")
    def test_batch_raises_on_no_result(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        context = Context(retriever=MockRetriever, model=mock_model)
        with pytest.raises(ValueError, match="No valid response"):
            context.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[create_sample_batch(qa_id="qa_001")],
            )

    @patch("gaussia.metrics.context.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Thinking...", {"insight": "Test insight", "score": 0.75})

        metrics = Context.run(ContextDatasetRetriever, model=mock_model, verbose=False)

        assert len(metrics) > 0
        assert all(isinstance(m, ContextMetric) for m in metrics)

    @patch("gaussia.metrics.context.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "test", "score": 0.5})

        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )
        context.batch(session_id="s", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="qa_001")])

        mock_judge_class.assert_called_once_with(
            model=mock_model,
            use_structured_output=False,
            strict=True,
            bos_json_clause="[",
            eos_json_clause="]",
            verbose=False,
        )

    @patch("gaussia.metrics.context.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Thinking...", {"insight": "Test", "score": 0.8})

        context = Context(retriever=MockRetriever, model=mock_model, verbose=True)
        context.batch(session_id="s", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="qa_001")])
        context.on_process_complete()

        assert len(context.metrics) == 1

    @patch("gaussia.metrics.context.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        from gaussia.llm.schemas import ContextJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        expected_result = ContextJudgeOutput(score=0.85, insight="Good context")
        mock_judge.check.return_value = ("", expected_result)

        context = Context(retriever=MockRetriever, model=mock_model, use_structured_output=True)
        context.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[create_sample_batch(qa_id="qa_001")],
        )
        context.on_process_complete()

        assert len(context.metrics) == 1
        assert context.metrics[0].context_awareness == pytest.approx(0.85)

    @patch("gaussia.metrics.context.Judge")
    def test_bayesian_mode_produces_ci(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"insight": "ok", "score": 0.8})

        context = Context(
            retriever=MockRetriever,
            model=mock_model,
            statistical_mode=BayesianMode(mc_samples=200, rng_seed=42),
        )
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        context.batch(session_id="s", context="c", assistant_id="a", batch=batches)
        context.on_process_complete()

        metric = context.metrics[0]
        assert metric.context_awareness_ci_low is not None
        assert metric.context_awareness_ci_high is not None
        assert metric.context_awareness_ci_low <= metric.context_awareness <= metric.context_awareness_ci_high
