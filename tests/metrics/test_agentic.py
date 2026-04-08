"""Tests for Agentic metric."""

from unittest.mock import MagicMock, patch

from pygaussia.metrics.agentic import Agentic
from pygaussia.schemas.agentic import AgenticMetric, ToolCorrectnessScore
from tests.fixtures.mock_retriever import AgenticDatasetRetriever, MockRetriever


class TestAgenticMetric:
    """Test suite for Agentic metric."""

    def mock_model(self):
        return MagicMock()

    def test_initialization_stores_k(self):
        """k is stored and accessible after initialization."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=5)
        assert agentic.k == 5
        assert agentic.threshold == 0.7
        assert agentic.tool_threshold == 1.0

    def test_evaluate_tool_correctness_perfect_match(self):
        """Perfect tool match scores 1.0 on all aspects."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=3)
        result = agentic._evaluate_tool_correctness(
            {"tools_used": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "result": 12, "step": 1}], "final_answer_uses_tools": True},
            {"expected_tools": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "step": 1}], "tool_sequence_matters": True},
        )
        assert isinstance(result, ToolCorrectnessScore)
        assert result.overall_correctness == 1.0
        assert result.is_correct is True

    def test_evaluate_tool_correctness_wrong_params(self):
        """Wrong parameters lower the overall score below 1.0."""
        agentic = Agentic(retriever=MockRetriever, model=MagicMock(), k=3)
        result = agentic._evaluate_tool_correctness(
            {"tools_used": [{"tool_name": "calc", "parameters": {"a": 5, "b": 8}, "step": 1}], "final_answer_uses_tools": True},
            {"expected_tools": [{"tool_name": "calc", "parameters": {"a": 5, "b": 7}, "step": 1}], "tool_sequence_matters": True},
        )
        assert result.parameter_accuracy < 1.0
        assert result.overall_correctness < 1.0

    @patch("pygaussia.metrics.agentic.Judge")
    def test_process_per_conversation_pass_at_k(self, mock_judge_class):
        """pass_at_k and pass_pow_k are computed per conversation using its interactions."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            # Conversation 1: 3/3 correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.85, "reasoning": ""}),
            ("", {"correctness_score": 0.95, "reasoning": ""}),
            # Conversation 2: 2/2 correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.85, "reasoning": ""}),
            # Conversation 3: 1/3 correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
            ("", {"correctness_score": 0.3, "reasoning": ""}),
            ("", {"correctness_score": 0.2, "reasoning": ""}),
            # Conversation 4: 1/1 correct
            ("", {"correctness_score": 0.9, "reasoning": ""}),
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics) == 4
        assert all(m.k == 3 for m in metrics)

        # Conversations 1, 2, 4: fully correct → pass_at_k = 1.0
        assert metrics[0].pass_at_k == 1.0 and metrics[0].pass_pow_k == 1.0
        assert metrics[1].pass_at_k == 1.0 and metrics[1].pass_pow_k == 1.0
        assert metrics[3].pass_at_k == 1.0 and metrics[3].pass_pow_k == 1.0

        # Conversation 3: 1/3 correct → p=1/3, pass@3 = 1-(2/3)^3 ≈ 0.704, pass^3 = (1/3)^3 ≈ 0.037
        assert 0.70 < metrics[2].pass_at_k < 0.71
        assert 0.037 < metrics[2].pass_pow_k < 0.038

    @patch("pygaussia.metrics.agentic.Judge")
    def test_process_all_correct(self, mock_judge_class):
        """All correct conversations → pass_at_k = pass_pow_k = 1.0."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.95, "reasoning": ""})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        metrics = agentic._process()

        assert all(m.pass_at_k == 1.0 for m in metrics)
        assert all(m.pass_pow_k == 1.0 for m in metrics)

    @patch("pygaussia.metrics.agentic.Judge")
    def test_process_none_correct(self, mock_judge_class):
        """All incorrect interactions → pass_at_k = pass_pow_k = 0.0."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.2, "reasoning": ""})

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        metrics = agentic._process()

        assert all(m.pass_at_k == 0.0 for m in metrics)
        assert all(m.pass_pow_k == 0.0 for m in metrics)

    @patch("pygaussia.metrics.agentic.Judge")
    def test_run_returns_metrics_with_pass_fields(self, mock_judge_class):
        """run() returns AgenticMetric instances with k, pass_at_k, pass_pow_k set."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("", {"correctness_score": 0.9, "reasoning": ""})

        metrics = Agentic.run(AgenticDatasetRetriever, k=3, model=MagicMock(), threshold=0.7)

        assert len(metrics) > 0
        assert isinstance(metrics[0], AgenticMetric)
        assert metrics[0].k == 3
        assert 0.0 <= metrics[0].pass_at_k <= 1.0
        assert 0.0 <= metrics[0].pass_pow_k <= 1.0

    @patch("pygaussia.metrics.agentic.Judge")
    def test_threshold_boundary(self, mock_judge_class):
        """Score at threshold counts as correct; just below does not."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.side_effect = [
            ("", {"correctness_score": 0.7, "reasoning": ""}),   # exactly at threshold → correct
            ("", {"correctness_score": 0.699, "reasoning": ""}),  # just below → incorrect
            ("", {"correctness_score": 0.701, "reasoning": ""}),  # just above → correct
        ]

        agentic = Agentic(retriever=AgenticDatasetRetriever, model=MagicMock(), k=3, threshold=0.7)
        metrics = agentic._process()

        assert len(metrics[0].correct_indices) == 2

    def test_pass_at_k_formula(self):
        """Bernoulli model: 1 - (1 - c/n)^k."""
        from pygaussia.metrics.agentic import pass_at_k

        assert 0.96 < pass_at_k(n=3, c=2, k=3) < 0.97   # 1 - (1/3)^3 ≈ 0.963
        assert pass_at_k(n=3, c=0, k=3) == 0.0
        assert pass_at_k(n=3, c=3, k=3) == 1.0
        assert 0.70 < pass_at_k(n=9, c=3, k=3) < 0.71   # 1 - (2/3)^3 ≈ 0.704

    def test_pass_pow_k_formula(self):
        """Formula: (c/n)^k."""
        from pygaussia.metrics.agentic import pass_pow_k

        assert 0.29 < pass_pow_k(n=3, c=2, k=3) < 0.30  # (2/3)^3 ≈ 0.296
        assert pass_pow_k(n=3, c=0, k=3) == 0.0
        assert pass_pow_k(n=3, c=3, k=3) == 1.0

    def test_pass_at_k_k_exceeds_n(self):
        """k > n is valid with the Bernoulli model."""
        from pygaussia.metrics.agentic import pass_at_k, pass_pow_k

        assert 0.99 < pass_at_k(n=3, c=2, k=5) < 1.0   # 1-(1/3)^5 ≈ 0.9959
        assert 0.13 < pass_pow_k(n=3, c=2, k=5) < 0.14  # (2/3)^5 ≈ 0.132
