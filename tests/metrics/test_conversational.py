"""Tests for Conversational metric."""

from unittest.mock import MagicMock, patch

import pytest

from pygaussia.metrics.conversational import Conversational
from pygaussia.schemas.conversational import ConversationalMetric, ConversationalScore
from pygaussia.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import ConversationalDatasetRetriever, MockRetriever

MOCK_JUDGE_RESULT = {
    "insight": "Good conversation quality",
    "memory": 0.9,
    "language": 0.85,
    "quality_maxim": 0.8,
    "quantity_maxim": 0.75,
    "relation_maxim": 0.9,
    "manner_maxim": 0.85,
    "sensibleness": 0.88,
}


class TestConversationalMetric:
    """Test suite for Conversational metric."""

    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    def test_initialization_default(self, mock_model):
        conv = Conversational(retriever=MockRetriever, model=mock_model)

        assert conv.model == mock_model
        assert isinstance(conv.statistical_mode, FrequentistMode)
        assert conv.use_structured_output is False
        assert conv.bos_json_clause == "```json"
        assert conv.eos_json_clause == "```"

    def test_initialization_custom(self, mock_model):
        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
        )

        assert conv.use_structured_output is True
        assert conv.bos_json_clause == "<json>"
        assert conv.eos_json_clause == "</json>"

    def test_initialization_bayesian(self, mock_model):
        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            statistical_mode=BayesianMode(mc_samples=100),
        )
        assert isinstance(conv.statistical_mode, BayesianMode)

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_processing(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("I analyzed the conversation...", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)

        batches = [create_sample_batch(qa_id="qa_001"), create_sample_batch(qa_id="qa_002")]
        conv.batch(session_id="test_session", context="Test context", assistant_id="test_assistant", batch=batches)
        conv.on_process_complete()

        assert len(conv.metrics) == 1
        metric = conv.metrics[0]
        assert isinstance(metric, ConversationalMetric)
        assert metric.session_id == "test_session"
        assert metric.n_interactions == 2
        assert isinstance(metric.conversational_memory, ConversationalScore)
        assert metric.conversational_memory.mean == pytest.approx(0.9)
        assert metric.conversational_memory.ci_low is None
        assert metric.conversational_language.mean == pytest.approx(0.85)
        assert metric.conversational_quality_maxim.mean == pytest.approx(0.8)
        assert metric.conversational_quantity_maxim.mean == pytest.approx(0.75)
        assert metric.conversational_relation_maxim.mean == pytest.approx(0.9)
        assert metric.conversational_manner_maxim.mean == pytest.approx(0.85)
        assert metric.conversational_sensibleness.mean == pytest.approx(0.88)
        assert len(metric.interactions) == 2
        assert metric.interactions[0].qa_id == "qa_001"
        assert metric.interactions[0].memory == pytest.approx(0.9)

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_session_accumulation(self, mock_judge_class, mock_model):
        """Two separate batch() calls for the same session_id → one metric."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        conv.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        conv.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q2")])
        conv.on_process_complete()

        assert len(conv.metrics) == 1
        assert conv.metrics[0].n_interactions == 2

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_multiple_sessions(self, mock_judge_class, mock_model):
        """Different session_ids → separate metrics."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        conv.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        conv.batch(session_id="s2", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q2")])
        conv.on_process_complete()

        assert len(conv.metrics) == 2

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_with_observation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Observation analysis...", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        batch = create_sample_batch(qa_id="qa_001", observation="The user seems satisfied")
        conv.batch(session_id="test_session", context="Test context", assistant_id="test_assistant", batch=[batch])

        call_args = mock_judge.check.call_args
        assert "observation" in call_args[0][2]

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_without_observation(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Analysis without observation...", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        batch = create_sample_batch(qa_id="qa_001", observation=None)
        conv.batch(session_id="test_session", context="Test context", assistant_id="test_assistant", batch=[batch])

        call_args = mock_judge.check.call_args
        assert "ground_truth_assistant" in call_args[0][2]

    @patch("pygaussia.metrics.conversational.Judge")
    def test_batch_raises_on_no_result(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        with pytest.raises(ValueError, match="No valid response"):
            conv.batch(
                session_id="test_session",
                context="Test context",
                assistant_id="test_assistant",
                batch=[create_sample_batch(qa_id="qa_001")],
            )

    @patch("pygaussia.metrics.conversational.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Thinking...", MOCK_JUDGE_RESULT)

        metrics = Conversational.run(ConversationalDatasetRetriever, model=mock_model, verbose=False)

        assert len(metrics) > 0
        assert all(isinstance(m, ConversationalMetric) for m in metrics)

    @patch("pygaussia.metrics.conversational.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", MOCK_JUDGE_RESULT)

        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=False,
            bos_json_clause="[",
            eos_json_clause="]",
        )
        conv.batch(session_id="s", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="qa_001")])

        mock_judge_class.assert_called_once_with(
            model=mock_model,
            use_structured_output=False,
            strict=True,
            bos_json_clause="[",
            eos_json_clause="]",
        )

    @patch("pygaussia.metrics.conversational.Judge")
    def test_language_parameter(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model)
        conv.batch(
            session_id="s",
            context="c",
            assistant_id="a",
            batch=[create_sample_batch(qa_id="qa_001")],
            language="spanish",
        )

        call_args = mock_judge.check.call_args
        assert call_args[0][2]["preferred_language"] == "spanish"

    @patch("pygaussia.metrics.conversational.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Thinking...", MOCK_JUDGE_RESULT)

        conv = Conversational(retriever=MockRetriever, model=mock_model, verbose=True)
        conv.batch(session_id="s", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="qa_001")])
        conv.on_process_complete()

        assert len(conv.metrics) == 1

    @patch("pygaussia.metrics.conversational.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        from pygaussia.llm.schemas import ConversationalJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        expected_result = ConversationalJudgeOutput(
            memory=8.5,
            language=9.0,
            insight="Good conversation",
            quality_maxim=8.0,
            quantity_maxim=7.5,
            relation_maxim=9.0,
            manner_maxim=8.5,
            sensibleness=9.0,
        )
        mock_judge.check.return_value = ("", expected_result)

        conv = Conversational(retriever=MockRetriever, model=mock_model, use_structured_output=True)
        conv.batch(
            session_id="test_session",
            context="Test context",
            assistant_id="test_assistant",
            batch=[create_sample_batch(qa_id="qa_001")],
        )
        conv.on_process_complete()

        assert len(conv.metrics) == 1
        assert conv.metrics[0].conversational_memory.mean == pytest.approx(8.5)
        assert conv.metrics[0].conversational_language.mean == pytest.approx(9.0)

    @patch("pygaussia.metrics.conversational.Judge")
    def test_bayesian_mode_produces_ci(self, mock_judge_class, mock_model):
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        varying_results = [
            {**MOCK_JUDGE_RESULT, "memory": 0.6},
            {**MOCK_JUDGE_RESULT, "memory": 0.8},
            {**MOCK_JUDGE_RESULT, "memory": 1.0},
        ]
        mock_judge.check.side_effect = [("", r) for r in varying_results]

        conv = Conversational(
            retriever=MockRetriever,
            model=mock_model,
            statistical_mode=BayesianMode(mc_samples=500, rng_seed=42),
        )
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        conv.batch(session_id="s", context="c", assistant_id="a", batch=batches)
        conv.on_process_complete()

        metric = conv.metrics[0]
        assert metric.conversational_memory.ci_low is not None
        assert metric.conversational_memory.ci_high is not None
        assert metric.conversational_memory.ci_low <= metric.conversational_memory.mean
        assert metric.conversational_memory.mean <= metric.conversational_memory.ci_high
