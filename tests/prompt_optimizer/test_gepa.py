from unittest.mock import MagicMock, patch

import pytest

from gaussia.prompt_optimizer.evaluators import LLMEvaluator
from gaussia.prompt_optimizer.gepa import GEPAOptimizer
from gaussia.prompt_optimizer.schemas import OptimizationResult
from tests.prompt_optimizer.conftest import PromptOptimizerRetriever


class TestGEPAInitialization:
    def test_default_evaluator_is_llm_evaluator(self, mock_model, mock_executor):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective", executor=mock_executor,
        )
        assert isinstance(optimizer.evaluator, LLMEvaluator)

    def test_custom_evaluator_is_used(self, mock_model, mock_executor, mock_evaluator):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=mock_evaluator,
        )
        assert optimizer.evaluator is mock_evaluator

    def test_custom_executor_is_used(self, mock_model, mock_executor):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective", executor=mock_executor,
        )
        assert optimizer.executor is mock_executor

    def test_default_iterations(self, mock_model, mock_executor):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective", executor=mock_executor,
        )
        assert optimizer.iterations == 5

    def test_custom_iterations(self, mock_model, mock_executor):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, iterations=3,
        )
        assert optimizer.iterations == 3

    def test_dataset_loaded_from_retriever(self, mock_model, mock_executor):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective", executor=mock_executor,
        )
        assert len(optimizer.dataset) == 2


class TestGEPAEvaluatePrompt:
    def test_returns_average_score(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.6)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        score, _ = optimizer._evaluate_prompt("test prompt")
        assert score == pytest.approx(0.6)

    def test_failing_examples_collected_below_threshold(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.3)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator, failure_threshold=0.6,
        )
        _, failing = optimizer._evaluate_prompt("prompt")
        assert len(failing) == 4

    def test_no_failing_when_above_threshold(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.9)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        _, failing = optimizer._evaluate_prompt("prompt")
        assert len(failing) == 0

    def test_executor_called_for_each_example(self, mock_model, mock_executor, mock_evaluator):
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=mock_evaluator,
        )
        optimizer._evaluate_prompt("prompt")
        assert mock_executor.call_count == 4

    def test_failing_example_fields_populated(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.2)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        _, failing = optimizer._evaluate_prompt("prompt")
        assert all(ex.query and ex.expected and ex.actual for ex in failing)
        assert all(ex.score == pytest.approx(0.2) for ex in failing)


class TestGEPAOptimize:
    def test_stops_immediately_when_no_failing_examples(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=1.0)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert result.iterations_run == 0
        assert result.optimized_prompt == "seed"

    def test_result_type(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=1.0)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert isinstance(result, OptimizationResult)

    def test_n_examples_in_result(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=1.0)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert result.n_examples == 4

    def test_initial_score_reflects_seed_prompt(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=1.0)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert result.initial_score == pytest.approx(1.0)

    @patch("gaussia.prompt_optimizer.gepa.gepa.GEPAOptimizer._generate_candidates")
    def test_adopts_better_candidate(self, mock_generate, mock_model, mock_executor):
        mock_generate.return_value = ["improved prompt"]
        call_count = 0

        def evaluator(actual, expected, query, context):
            nonlocal call_count
            call_count += 1
            # seed evaluation (4 calls) → low, candidate evaluation (4 calls) → high, next eval (4 calls) → perfect
            if call_count <= 4:
                return 0.3
            if call_count <= 8:
                return 0.9
            return 1.0

        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert result.optimized_prompt == "improved prompt"
        assert result.final_score > result.initial_score

    @patch("gaussia.prompt_optimizer.gepa.gepa.GEPAOptimizer._generate_candidates")
    def test_stops_when_candidate_does_not_improve(self, mock_generate, mock_model, mock_executor):
        mock_generate.return_value = ["same quality prompt"]
        evaluator = MagicMock(return_value=0.4)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator, iterations=5,
        )
        result = optimizer._optimize()
        assert result.iterations_run == 1

    @patch("gaussia.prompt_optimizer.gepa.gepa.GEPAOptimizer._generate_candidates")
    def test_history_has_one_entry_per_completed_iteration(self, mock_generate, mock_model, mock_executor):
        mock_generate.return_value = ["candidate"]
        evaluator = MagicMock(return_value=0.4)
        optimizer = GEPAOptimizer(
            PromptOptimizerRetriever, model=mock_model,
            seed_prompt="seed", objective="objective",
            executor=mock_executor, evaluator=evaluator,
        )
        result = optimizer._optimize()
        assert len(result.history) == result.iterations_run


class TestGEPARun:
    def test_run_returns_optimization_result(self, mock_model, mock_executor, mock_evaluator):
        mock_evaluator.return_value = 1.0
        result = GEPAOptimizer.run(
            PromptOptimizerRetriever,
            model=mock_model,
            seed_prompt="seed",
            objective="objective",
            executor=mock_executor,
            evaluator=mock_evaluator,
        )
        assert isinstance(result, OptimizationResult)

    def test_run_exposes_optimized_prompt(self, mock_model, mock_executor, mock_evaluator):
        mock_evaluator.return_value = 1.0
        result = GEPAOptimizer.run(
            PromptOptimizerRetriever,
            model=mock_model,
            seed_prompt="my seed",
            objective="objective",
            executor=mock_executor,
            evaluator=mock_evaluator,
        )
        assert isinstance(result.optimized_prompt, str)
        assert len(result.optimized_prompt) > 0
