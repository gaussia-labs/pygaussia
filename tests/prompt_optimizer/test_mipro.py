from unittest.mock import MagicMock, patch

import pytest

from gaussia.prompt_optimizer.mipro import MIPROv2Optimizer
from gaussia.prompt_optimizer.mipro.proposer import DemoBootstrapper, InstructionProposer, _InstructionVariants
from gaussia.prompt_optimizer.schemas import Demo, MIPROv2Result
from tests.prompt_optimizer.conftest import MOCK_DATASETS, PromptOptimizerRetriever


def _make_optimizer(mock_model, mock_executor, mock_evaluator, **kwargs) -> MIPROv2Optimizer:
    defaults = {
        "num_candidates": 2,
        "num_trials": 2,
        "num_demo_sets": 2,
        "max_demos_per_set": 2,
    }
    defaults.update(kwargs)
    return MIPROv2Optimizer(
        PromptOptimizerRetriever, model=mock_model,
        seed_prompt="seed", objective="objective",
        executor=mock_executor, evaluator=mock_evaluator,
        **defaults,
    )


class TestDemoBootstrapper:
    def test_returns_correct_number_of_demo_sets(self):
        bootstrapper = DemoBootstrapper(MOCK_DATASETS, num_demo_sets=5, max_demos_per_set=2, random_seed=42)
        assert len(bootstrapper.bootstrap()) == 5

    def test_demos_respect_max_per_set(self):
        bootstrapper = DemoBootstrapper(MOCK_DATASETS, num_demo_sets=3, max_demos_per_set=2, random_seed=42)
        for demo_set in bootstrapper.bootstrap():
            assert len(demo_set) <= 2

    def test_demos_are_demo_instances(self):
        bootstrapper = DemoBootstrapper(MOCK_DATASETS, num_demo_sets=3, max_demos_per_set=2, random_seed=42)
        for demo_set in bootstrapper.bootstrap():
            assert all(isinstance(d, Demo) for d in demo_set)

    def test_demo_queries_come_from_dataset(self):
        all_queries = {batch.query for dataset in MOCK_DATASETS for batch in dataset.conversation}
        bootstrapper = DemoBootstrapper(MOCK_DATASETS, num_demo_sets=3, max_demos_per_set=2, random_seed=42)
        for demo_set in bootstrapper.bootstrap():
            assert all(d.query in all_queries for d in demo_set)

    def test_reproducible_with_same_seed(self):
        b1 = DemoBootstrapper(MOCK_DATASETS, 3, 2, 42).bootstrap()
        b2 = DemoBootstrapper(MOCK_DATASETS, 3, 2, 42).bootstrap()
        assert [[d.query for d in ds] for ds in b1] == [[d.query for d in ds] for ds in b2]

    def test_max_demos_capped_by_dataset_size(self):
        bootstrapper = DemoBootstrapper(MOCK_DATASETS, num_demo_sets=2, max_demos_per_set=100, random_seed=42)
        for demo_set in bootstrapper.bootstrap():
            assert len(demo_set) <= 4


class TestInstructionProposer:
    def test_returns_correct_number_of_instructions(self, mock_model):
        mock_model.with_structured_output.return_value.invoke.return_value = _InstructionVariants(
            instructions=["inst1", "inst2", "inst3"]
        )
        proposer = InstructionProposer(mock_model, "seed", "objective", num_candidates=3)
        assert len(proposer.propose()) == 3

    def test_calls_model_with_structured_output(self, mock_model):
        mock_model.with_structured_output.return_value.invoke.return_value = _InstructionVariants(
            instructions=["inst1"]
        )
        proposer = InstructionProposer(mock_model, "seed", "objective", num_candidates=1)
        proposer.propose()
        mock_model.with_structured_output.assert_called_once()

    def test_truncates_to_num_candidates(self, mock_model):
        mock_model.with_structured_output.return_value.invoke.return_value = _InstructionVariants(
            instructions=["a", "b", "c", "d", "e"]
        )
        proposer = InstructionProposer(mock_model, "seed", "objective", num_candidates=3)
        assert len(proposer.propose()) == 3

    def test_returns_strings(self, mock_model):
        mock_model.with_structured_output.return_value.invoke.return_value = _InstructionVariants(
            instructions=["instruction one", "instruction two"]
        )
        proposer = InstructionProposer(mock_model, "seed", "objective", num_candidates=2)
        instructions = proposer.propose()
        assert all(isinstance(i, str) for i in instructions)


class TestMIPROv2BuildPrompt:
    def test_no_demos_returns_instruction_only(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        assert optimizer._build_prompt("the instruction", []) == "the instruction"

    def test_with_demos_includes_instruction(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        demos = [Demo(query="q1", response="r1")]
        result = optimizer._build_prompt("instruction", demos)
        assert "instruction" in result

    def test_with_demos_includes_query_and_response(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        demos = [Demo(query="q1", response="r1"), Demo(query="q2", response="r2")]
        result = optimizer._build_prompt("instruction", demos)
        assert "q1" in result
        assert "r1" in result
        assert "q2" in result
        assert "r2" in result


class TestMIPROv2CollectExamples:
    def test_returns_all_examples(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        assert len(optimizer._collect_examples()) == 4

    def test_each_example_is_context_batch_tuple(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        for context, batch in optimizer._collect_examples():
            assert isinstance(context, str)
            assert hasattr(batch, "query")


class TestMIPROv2ScoreExamples:
    def test_returns_average_score(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.8)
        optimizer = _make_optimizer(mock_model, mock_executor, evaluator)
        examples = optimizer._collect_examples()
        assert optimizer._score_examples("prompt", examples) == pytest.approx(0.8)

    def test_empty_examples_returns_zero(self, mock_model, mock_executor, mock_evaluator):
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        assert optimizer._score_examples("prompt", []) == pytest.approx(0.0)

    def test_evaluator_called_for_each_example(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.5)
        optimizer = _make_optimizer(mock_model, mock_executor, evaluator)
        examples = optimizer._collect_examples()
        optimizer._score_examples("prompt", examples)
        assert evaluator.call_count == 4


class TestMIPROv2EvaluatePrompt:
    def test_uses_all_examples_without_minibatch(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.5)
        optimizer = _make_optimizer(mock_model, mock_executor, evaluator, minibatch_size=2)
        optimizer._evaluate_prompt("prompt", minibatch=False)
        assert evaluator.call_count == 4

    def test_uses_minibatch_when_dataset_exceeds_size(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.5)
        optimizer = _make_optimizer(mock_model, mock_executor, evaluator, minibatch_size=2)
        optimizer._evaluate_prompt("prompt", minibatch=True)
        assert evaluator.call_count == 2

    def test_uses_all_when_dataset_smaller_than_minibatch(self, mock_model, mock_executor):
        evaluator = MagicMock(return_value=0.5)
        optimizer = _make_optimizer(mock_model, mock_executor, evaluator, minibatch_size=100)
        optimizer._evaluate_prompt("prompt", minibatch=True)
        assert evaluator.call_count == 4


class TestMIPROv2Optimize:
    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_result_type(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        result = optimizer._optimize()
        assert isinstance(result, MIPROv2Result)

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_n_examples_in_result(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        result = optimizer._optimize()
        assert result.n_examples == 4

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_demos_are_demo_instances(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        result = optimizer._optimize()
        assert all(isinstance(d, Demo) for d in result.demos)

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_trials_run_matches_num_trials(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator, num_trials=2)
        result = optimizer._optimize()
        assert result.trials_run == 2

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_optimized_instruction_is_string(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        result = optimizer._optimize()
        assert isinstance(result.optimized_instruction, str)
        assert len(result.optimized_instruction) > 0

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_optimized_prompt_contains_instruction(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        optimizer = _make_optimizer(mock_model, mock_executor, mock_evaluator)
        result = optimizer._optimize()
        assert result.optimized_instruction in result.optimized_prompt


class TestMIPROv2Run:
    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_run_returns_miprov2_result(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        result = MIPROv2Optimizer.run(
            PromptOptimizerRetriever,
            model=mock_model,
            seed_prompt="seed",
            objective="objective",
            executor=mock_executor,
            evaluator=mock_evaluator,
            num_candidates=2,
            num_trials=2,
            num_demo_sets=2,
        )
        assert isinstance(result, MIPROv2Result)

    @patch("gaussia.prompt_optimizer.mipro.mipro.InstructionProposer")
    def test_run_exposes_optimized_prompt(self, mock_proposer_class, mock_model, mock_executor, mock_evaluator):
        mock_proposer_class.return_value.propose.return_value = ["inst1", "inst2"]
        result = MIPROv2Optimizer.run(
            PromptOptimizerRetriever,
            model=mock_model,
            seed_prompt="seed",
            objective="objective",
            executor=mock_executor,
            evaluator=mock_evaluator,
            num_candidates=2,
            num_trials=2,
            num_demo_sets=2,
        )
        assert isinstance(result.optimized_prompt, str)
        assert len(result.optimized_prompt) > 0
