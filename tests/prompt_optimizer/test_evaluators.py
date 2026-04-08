import pytest
from unittest.mock import MagicMock

from pygaussia.prompt_optimizer.evaluators import LLMEvaluator


class TestLLMEvaluatorParseScore:
    def test_valid_json_score(self):
        assert LLMEvaluator._parse_score('{"score": 0.85}') == pytest.approx(0.85)

    def test_score_with_spaces(self):
        assert LLMEvaluator._parse_score('{"score" : 0.5}') == pytest.approx(0.5)

    def test_integer_score(self):
        assert LLMEvaluator._parse_score('{"score": 1}') == pytest.approx(1.0)

    def test_clamps_above_one(self):
        assert LLMEvaluator._parse_score('{"score": 1.5}') == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert LLMEvaluator._parse_score('{"score": -0.1}') == pytest.approx(0.0)

    def test_invalid_returns_zero(self):
        assert LLMEvaluator._parse_score("no score here") == pytest.approx(0.0)

    def test_empty_string_returns_zero(self):
        assert LLMEvaluator._parse_score("") == pytest.approx(0.0)

    def test_score_embedded_in_text(self):
        assert LLMEvaluator._parse_score('Here is my eval: {"score": 0.72}') == pytest.approx(0.72)


class TestLLMEvaluatorCall:
    def test_returns_parsed_score(self, mock_model):
        mock_model.invoke.return_value = MagicMock(content='{"score": 0.75}')
        evaluator = LLMEvaluator(model=mock_model, criteria="Test criteria.")
        score = evaluator("actual response", "expected response", "query", "context")
        assert score == pytest.approx(0.75)

    def test_calls_model_once(self, mock_model):
        mock_model.invoke.return_value = MagicMock(content='{"score": 0.5}')
        evaluator = LLMEvaluator(model=mock_model, criteria="criteria")
        evaluator("actual", "expected", "query", "context")
        mock_model.invoke.assert_called_once()

    def test_invalid_model_response_returns_zero(self, mock_model):
        mock_model.invoke.return_value = MagicMock(content="I think this looks good.")
        evaluator = LLMEvaluator(model=mock_model, criteria="criteria")
        score = evaluator("actual", "expected", "query", "context")
        assert score == pytest.approx(0.0)

    def test_boundary_score_zero(self, mock_model):
        mock_model.invoke.return_value = MagicMock(content='{"score": 0.0}')
        evaluator = LLMEvaluator(model=mock_model, criteria="criteria")
        assert evaluator("a", "e", "q", "c") == pytest.approx(0.0)

    def test_boundary_score_one(self, mock_model):
        mock_model.invoke.return_value = MagicMock(content='{"score": 1.0}')
        evaluator = LLMEvaluator(model=mock_model, criteria="criteria")
        assert evaluator("a", "e", "q", "c") == pytest.approx(1.0)
