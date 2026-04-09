"""Tests for BestOf metric."""

from unittest.mock import MagicMock, patch

import pytest

from gaussia.metrics.best_of import BestOf
from gaussia.schemas.best_of import BestOfMetric
from tests.fixtures.mock_retriever import BestOfDatasetRetriever, MockRetriever


class TestBestOfMetric:
    """Test suite for BestOf metric."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test BestOf initialization with default parameters."""
        best_of = BestOf(retriever=MockRetriever, model=mock_model)

        assert best_of.model == mock_model
        assert best_of.use_structured_output is False
        assert best_of.bos_json_clause == "```json"
        assert best_of.eos_json_clause == "```"
        assert best_of.criteria == "BestOf"

    def test_initialization_custom(self, mock_model):
        """Test BestOf initialization with custom parameters."""
        best_of = BestOf(
            retriever=MockRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="<json>",
            eos_json_clause="</json>",
            criteria="Quality comparison",
        )

        assert best_of.model == mock_model
        assert best_of.use_structured_output is True
        assert best_of.bos_json_clause == "<json>"
        assert best_of.eos_json_clause == "</json>"
        assert best_of.criteria == "Quality comparison"

    @patch("gaussia.metrics.best_of.Judge")
    def test_process_tournament(self, mock_judge_class, mock_model):
        """Test tournament processing with multiple contestants."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Analyzing contestants...",
            {
                "winner": "assistant_a",
                "verdict": "Better overall performance",
                "confidence": 0.85,
                "reasoning": {"assistant_a": {"strengths": ["accurate"]}},
            },
        )

        best_of = BestOf(retriever=BestOfDatasetRetriever, model=mock_model)
        metrics = best_of._process()

        assert len(metrics) == 1
        assert isinstance(metrics[0], BestOfMetric)
        assert metrics[0].bestof_winner_id is not None

    @patch("gaussia.metrics.best_of.Judge")
    def test_run_method(self, mock_judge_class, mock_model):
        """Test the run class method."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "winner": "assistant_a",
                "verdict": "Better quality",
                "confidence": 0.9,
                "reasoning": {},
            },
        )

        metrics = BestOf.run(
            BestOfDatasetRetriever,
            model=mock_model,
            verbose=False,
        )

        assert len(metrics) > 0
        assert isinstance(metrics[0], BestOfMetric)

    @patch("gaussia.metrics.best_of.Judge")
    def test_judge_initialization_params(self, mock_judge_class, mock_model):
        """Test that Judge is initialized with correct parameters."""
        from gaussia.llm.schemas import BestOfJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "",
            BestOfJudgeOutput(
                winner="assistant_a",
                verdict="test",
                confidence=0.5,
                reasoning={},
            ),
        )

        best_of = BestOf(
            retriever=BestOfDatasetRetriever,
            model=mock_model,
            use_structured_output=True,
            bos_json_clause="[",
            eos_json_clause="]",
        )

        best_of._process()

        mock_judge_class.assert_called_once_with(
            model=mock_model,
            use_structured_output=True,
            strict=True,
            bos_json_clause="[",
            eos_json_clause="]",
        )

    @patch("gaussia.metrics.best_of.Judge")
    def test_tie_handling(self, mock_judge_class, mock_model):
        """Test handling of tie results."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        # First call returns tie, second call picks a winner to end tournament
        mock_judge.check.side_effect = [
            (
                "Both are equal...",
                {
                    "winner": "tie",
                    "verdict": "Both responses are equally good",
                    "confidence": 0.5,
                    "reasoning": {},
                },
            ),
            (
                "Deciding...",
                {
                    "winner": "assistant_a",
                    "verdict": "Slightly better",
                    "confidence": 0.6,
                    "reasoning": {},
                },
            ),
        ]

        best_of = BestOf(retriever=BestOfDatasetRetriever, model=mock_model)
        metrics = best_of._process()

        assert len(metrics) == 1
        # First round was a tie, second round picked a winner
        assert metrics[0].bestof_winner_id is not None

    @patch("gaussia.metrics.best_of.Judge")
    def test_structured_output_mode(self, mock_judge_class, mock_model):
        """Test BestOf with structured output enabled."""
        from gaussia.llm.schemas import BestOfJudgeOutput

        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        expected_result = BestOfJudgeOutput(
            winner="assistant_a",
            verdict="Superior performance",
            confidence=0.9,
            reasoning={"assistant_a": {"strengths": ["accurate", "complete"]}},
        )
        mock_judge.check.return_value = ("", expected_result)

        best_of = BestOf(
            retriever=BestOfDatasetRetriever,
            model=mock_model,
            use_structured_output=True,
        )

        metrics = best_of._process()

        assert len(metrics) == 1
        assert metrics[0].bestof_winner_id == "assistant_a"

    @patch("gaussia.metrics.best_of.Judge")
    def test_raises_on_no_result(self, mock_judge_class, mock_model):
        """Test that ValueError is raised when judge returns None."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = ("Some thinking", None)

        best_of = BestOf(retriever=BestOfDatasetRetriever, model=mock_model)

        with pytest.raises(ValueError, match="No valid response"):
            best_of._process()

    @patch("gaussia.metrics.best_of.Judge")
    def test_verbose_mode(self, mock_judge_class, mock_model):
        """Test that verbose mode doesn't break processing."""
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge
        mock_judge.check.return_value = (
            "Thinking...",
            {
                "winner": "assistant_a",
                "verdict": "Test",
                "confidence": 0.8,
                "reasoning": {},
            },
        )

        best_of = BestOf(retriever=BestOfDatasetRetriever, model=mock_model, verbose=True)
        metrics = best_of._process()

        assert len(metrics) == 1
