"""Tests for RoleAdherence metric."""

from unittest.mock import MagicMock

import pytest

from gaussia.metrics.role_adherence import LLMJudgeStrategy, RoleAdherence, ScoringStrategy
from gaussia.schemas.common import IterationLevel
from gaussia.schemas.role_adherence import RoleAdherenceMetric
from gaussia.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_role_adherence_dataset, create_sample_batch
from tests.fixtures.mock_retriever import MockRetriever, RoleAdherenceDatasetRetriever


def make_mock_strategy(scores: list[float], reason: str = "mock reason") -> ScoringStrategy:
    """Create a mock ScoringStrategy that returns predetermined scores in order."""
    strategy = MagicMock(spec=ScoringStrategy)
    strategy.score.side_effect = [(s, reason) for s in scores]
    return strategy


class TestRoleAdherenceInit:
    def test_default_parameters(self):
        strategy = make_mock_strategy([1.0, 1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy)

        assert metric.scoring_strategy is strategy
        assert isinstance(metric.statistical_mode, FrequentistMode)
        assert metric.binary is True
        assert metric.strict_mode is False
        assert metric.threshold == 0.5
        assert metric.include_reason is False

    def test_custom_parameters(self):
        strategy = make_mock_strategy([])
        metric = RoleAdherence(
            retriever=MockRetriever,
            scoring_strategy=strategy,
            statistical_mode=BayesianMode(mc_samples=100),
            binary=False,
            strict_mode=True,
            threshold=0.7,
            include_reason=True,
        )

        assert isinstance(metric.statistical_mode, BayesianMode)
        assert metric.binary is False
        assert metric.strict_mode is True
        assert metric.threshold == 0.7
        assert metric.include_reason is True

    def test_raises_on_stream_batches(self):
        from gaussia.core.retriever import Retriever
        from gaussia.schemas.common import IterationLevel

        class StreamBatchRetriever(Retriever):
            @property
            def iteration_level(self):
                return IterationLevel.STREAM_BATCHES

            def load_dataset(self):
                return iter([])

        strategy = make_mock_strategy([])
        with pytest.raises(ValueError, match="STREAM_BATCHES"):
            RoleAdherence(retriever=StreamBatchRetriever, scoring_strategy=strategy)


class TestRoleAdherenceBatch:
    def test_batch_accumulates_turns(self):
        strategy = make_mock_strategy([1.0, 1.0, 1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
            create_sample_batch(qa_id="qa_003"),
        ]
        metric._current_chatbot_role = "Support agent role"
        metric.batch(session_id="s1", context="ctx", assistant_id="bot", batch=batches)
        metric.on_process_complete()

        assert len(metric.metrics) == 1
        m = metric.metrics[0]
        assert m.n_turns == 3
        assert len(m.turns) == 3

    def test_history_grows_incrementally(self):
        """Verify that T<i is passed correctly — first turn has empty history, second has one entry, etc."""
        strategy = make_mock_strategy([1.0, 1.0, 1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy)

        b0 = create_sample_batch(qa_id="qa_001")
        b1 = create_sample_batch(qa_id="qa_002")
        b2 = create_sample_batch(qa_id="qa_003")
        metric._current_chatbot_role = "role"
        metric.batch(session_id="s1", context="ctx", assistant_id="bot", batch=[b0, b1, b2])

        calls = strategy.score.call_args_list
        assert calls[0][0][1] == []        # first turn: empty history
        assert calls[1][0][1] == [b0]      # second turn: one prior turn
        assert calls[2][0][1] == [b0, b1]  # third turn: two prior turns

    def test_chatbot_role_passed_to_strategy(self):
        strategy = make_mock_strategy([1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy)

        metric._current_chatbot_role = "Fintech support agent"
        metric.batch(
            session_id="s1", context="ctx", assistant_id="bot", batch=[create_sample_batch(qa_id="qa_001")]
        )

        _, _, role_arg = strategy.score.call_args[0]
        assert role_arg == "Fintech support agent"

    def test_multiple_sessions_produce_separate_metrics(self):
        strategy = make_mock_strategy([1.0, 1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy)

        metric._current_chatbot_role = "role"
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        metric.batch(session_id="s2", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q2")])
        metric.on_process_complete()

        assert len(metric.metrics) == 2
        session_ids = {m.session_id for m in metric.metrics}
        assert session_ids == {"s1", "s2"}


class TestBinaryMode:
    def test_binary_true_binarizes_scores(self):
        strategy = make_mock_strategy([0.9, 0.3, 0.8])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, binary=True, threshold=0.5)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        turns = metric.metrics[0].turns
        assert turns[0].adherence_score == 1.0  # 0.9 >= 0.5 → 1
        assert turns[1].adherence_score == 0.0  # 0.3 < 0.5 → 0
        assert turns[2].adherence_score == 1.0  # 0.8 >= 0.5 → 1
        assert metric.metrics[0].role_adherence == pytest.approx(2 / 3)

    def test_binary_false_keeps_raw_scores(self):
        strategy = make_mock_strategy([0.9, 0.3, 0.8])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, binary=False, threshold=0.5)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        turns = metric.metrics[0].turns
        assert turns[0].adherence_score == pytest.approx(0.9)
        assert turns[1].adherence_score == pytest.approx(0.3)
        assert turns[2].adherence_score == pytest.approx(0.8)
        assert metric.metrics[0].role_adherence == pytest.approx((0.9 + 0.3 + 0.8) / 3)


class TestStrictMode:
    def test_strict_mode_false_uses_mean_threshold(self):
        strategy = make_mock_strategy([1.0, 1.0, 0.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, strict_mode=False, threshold=0.5)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        assert metric.metrics[0].role_adherence == pytest.approx(2 / 3)
        assert metric.metrics[0].adherent is True  # 0.667 >= 0.5

    def test_strict_mode_true_requires_all_turns_adherent(self):
        strategy = make_mock_strategy([1.0, 1.0, 0.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, strict_mode=True, threshold=0.5)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        assert metric.metrics[0].adherent is False  # one turn failed

    def test_strict_mode_true_all_pass(self):
        strategy = make_mock_strategy([1.0, 1.0, 1.0])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, strict_mode=True, threshold=0.5)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(3)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        assert metric.metrics[0].adherent is True


class TestThreshold:
    def test_custom_threshold(self):
        strategy = make_mock_strategy([0.6, 0.6])
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, threshold=0.7)

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(2)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        # 0.6 < 0.7 → not adherent per turn; mean = 0.0 < 0.7 → session not adherent
        for turn in metric.metrics[0].turns:
            assert turn.adherent is False
        assert metric.metrics[0].adherent is False


class TestIncludeReason:
    def test_reason_excluded_by_default(self):
        strategy = make_mock_strategy([1.0], reason="The response is within scope.")
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, include_reason=False)

        metric._current_chatbot_role = "role"
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        metric.on_process_complete()

        assert metric.metrics[0].turns[0].reason is None

    def test_reason_included_when_enabled(self):
        strategy = make_mock_strategy([1.0], reason="The response is within scope.")
        metric = RoleAdherence(retriever=MockRetriever, scoring_strategy=strategy, include_reason=True)

        metric._current_chatbot_role = "role"
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=[create_sample_batch(qa_id="q1")])
        metric.on_process_complete()

        assert metric.metrics[0].turns[0].reason == "The response is within scope."


class TestBayesianMode:
    def test_bayesian_mode_produces_ci(self):
        scores = [1.0, 1.0, 0.0, 1.0, 1.0]
        strategy = make_mock_strategy(scores)
        metric = RoleAdherence(
            retriever=MockRetriever,
            scoring_strategy=strategy,
            statistical_mode=BayesianMode(mc_samples=500, rng_seed=42),
        )

        metric._current_chatbot_role = "role"
        batches = [create_sample_batch(qa_id=f"q{i}") for i in range(5)]
        metric.batch(session_id="s1", context="c", assistant_id="a", batch=batches)
        metric.on_process_complete()

        m = metric.metrics[0]
        assert m.role_adherence_ci_low is not None
        assert m.role_adherence_ci_high is not None
        assert m.role_adherence_ci_low <= m.role_adherence <= m.role_adherence_ci_high


class TestRunMethod:
    def test_run_end_to_end(self):
        strategy = make_mock_strategy([1.0, 1.0, 1.0])
        metrics = RoleAdherence.run(
            RoleAdherenceDatasetRetriever,
            scoring_strategy=strategy,
            binary=True,
            strict_mode=False,
            threshold=0.5,
            include_reason=False,
        )

        assert len(metrics) > 0
        assert all(isinstance(m, RoleAdherenceMetric) for m in metrics)

    def test_run_captures_chatbot_role_from_dataset(self):
        dataset = create_role_adherence_dataset()
        expected_role = dataset.chatbot_role

        strategy = make_mock_strategy([1.0, 1.0, 1.0])
        RoleAdherence.run(RoleAdherenceDatasetRetriever, scoring_strategy=strategy)

        _, _, role_arg = strategy.score.call_args_list[0][0]
        assert role_arg == expected_role

    def test_run_empty_chatbot_role_defaults_to_empty_string(self):
        strategy = make_mock_strategy([1.0, 1.0])
        RoleAdherence.run(MockRetriever, scoring_strategy=strategy)

        _, _, role_arg = strategy.score.call_args_list[0][0]
        assert role_arg == ""


class TestLLMJudgeStrategy:
    def test_selects_binary_prompt(self):
        from gaussia.llm.prompts import role_adherence_binary_system_prompt

        strategy = LLMJudgeStrategy(model=MagicMock(), binary=True)
        assert strategy._prompt == role_adherence_binary_system_prompt

    def test_selects_continuous_prompt(self):
        from gaussia.llm.prompts import role_adherence_continuous_system_prompt

        strategy = LLMJudgeStrategy(model=MagicMock(), binary=False)
        assert strategy._prompt == role_adherence_continuous_system_prompt

    def test_format_history_empty(self):
        strategy = LLMJudgeStrategy(model=MagicMock())
        assert strategy._format_history([]) == "No prior conversation."

    def test_format_history_with_turns(self):
        strategy = LLMJudgeStrategy(model=MagicMock())
        turns = [
            create_sample_batch(qa_id="q1", query="Hello", assistant="Hi there"),
            create_sample_batch(qa_id="q2", query="How are you?", assistant="I'm fine"),
        ]
        result = strategy._format_history(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
        assert "User: How are you?" in result
        assert "Assistant: I'm fine" in result

    def test_score_calls_judge_with_correct_data(self):
        from unittest.mock import patch

        mock_model = MagicMock()
        strategy = LLMJudgeStrategy(model=mock_model, binary=True)
        turn = create_sample_batch(qa_id="q1", query="Test query", assistant="Test response")

        with patch("gaussia.metrics.role_adherence.Judge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge
            mock_judge.check.return_value = ("", {"score": 1.0, "reason": "Correct"})

            score, reason = strategy.score(turn, [], "Fintech agent role")

        assert score == 1.0
        assert reason == "Correct"
        call_data = mock_judge.check.call_args[0][2]
        assert call_data["chatbot_role"] == "Fintech agent role"
        assert call_data["query"] == "Test query"
        assert call_data["assistant_response"] == "Test response"
        assert call_data["history"] == "No prior conversation."

    def test_score_raises_on_none_result(self):
        from unittest.mock import patch

        strategy = LLMJudgeStrategy(model=MagicMock(), binary=True)
        turn = create_sample_batch(qa_id="q1")

        with patch("gaussia.metrics.role_adherence.Judge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge
            mock_judge.check.return_value = ("", None)

            with pytest.raises(ValueError, match="No valid response"):
                strategy.score(turn, [], "role")
