from __future__ import annotations

import pytest

from gaussia.integrations.evalhub.benchmarks import run_gaussia_benchmark
from gaussia.integrations.evalhub.config import ProviderConfig
from gaussia.integrations.evalhub.payloads import load_benchmark_input
from gaussia.schemas.agentic import AgenticMetric


class FakeJudgeConnector:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_agentic_benchmark_uses_ground_truth_metrics(monkeypatch) -> None:
    def fake_agentic_run(_retriever, **kwargs):
        assert kwargs["k"] == 2
        assert kwargs["threshold"] == 0.8
        return [
            AgenticMetric(
                session_id="session-1",
                assistant_id="assistant-1",
                total_interactions=2,
                correct_interactions=1,
                is_fully_correct=False,
                threshold=0.8,
                correctness_scores=[0.9, 0.4],
                correct_indices=[0],
                k=2,
                pass_at_k=0.75,
                pass_pow_k=0.25,
            )
        ]

    monkeypatch.setattr(
        "gaussia.integrations.evalhub.benchmarks._agentic_metric_kwargs",
        lambda _config: {"model": object(), "k": 2, "threshold": 0.8},
    )
    monkeypatch.setattr("gaussia.integrations.evalhub.benchmarks.Agentic.run", fake_agentic_run)

    execution = run_gaussia_benchmark(
        benchmark_id="agentic",
        benchmark_input=load_benchmark_input(_dataset_payload()),
        provider_id="gaussia",
        config=ProviderConfig(),
    )

    assert execution.primary_metric_name == "agentic_pass_at_k"
    assert execution.primary_metric_value == 0.75
    assert execution.overall_score == 0.75
    assert [(result.metric_name, result.metric_value) for result in execution.evaluation_results] == [
        ("agentic_pass_at_k", 0.75),
        ("agentic_pass_pow_k", 0.25),
        ("agentic_correctness_rate", 0.5),
        ("agentic_correct_interactions", 1.0),
        ("agentic_total_interactions", 2.0),
    ]
    assert execution.artifact_payload["summary"]["correct_interactions"] == 1
    assert execution.artifact_payload["summary"]["total_interactions"] == 2


def test_agentic_benchmark_requires_ground_truth_assistant() -> None:
    payload = _dataset_payload()
    payload["dataset"]["conversation"][1]["ground_truth_assistant"] = ""

    with pytest.raises(ValueError, match="requires ground_truth_assistant"):
        run_gaussia_benchmark(
            benchmark_id="agentic",
            benchmark_input=load_benchmark_input(payload),
            provider_id="gaussia",
            config=ProviderConfig(),
        )


def test_guardian_config_supports_served_model_and_tokenizer_model() -> None:
    config = ProviderConfig(
        guardian_model="granite-guardian-serving",
        guardian_tokenizer_model="ibm-granite/granite-guardian-3.1-2b",
        guardian_api_key="test-key",
        guardian_base_url="https://guardian.example.com/v1",
        guardian_chat_completions=True,
    )

    guardian_config = config.require_guardian_config()

    assert guardian_config.model == "granite-guardian-serving"
    assert guardian_config.tokenizer_model == "ibm-granite/granite-guardian-3.1-2b"
    assert guardian_config.url == "https://guardian.example.com/v1"
    assert guardian_config.chat_completions is True


def test_provider_config_accepts_injected_judge_connector() -> None:
    connector = object()

    assert ProviderConfig(judge_connector=connector).require_judge_model() is connector


def test_provider_config_builds_configured_langchain_connector() -> None:
    connector = ProviderConfig(
        judge_connector_class=f"{__name__}.FakeJudgeConnector",
        judge_connector_kwargs={"timeout": 30, "model": "custom-model-key"},
        judge_model="judge-model",
        judge_api_key="test-key",
        judge_base_url="https://judge.example.com/v1",
        judge_temperature=0.2,
    ).require_judge_model()

    assert isinstance(connector, FakeJudgeConnector)
    assert connector.kwargs == {
        "timeout": 30,
        "model": "custom-model-key",
        "api_key": "test-key",
        "base_url": "https://judge.example.com/v1",
        "temperature": 0.2,
    }


def test_provider_config_loads_judge_connector_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GAUSSIA_JUDGE_CONNECTOR_CLASS", f"{__name__}.FakeJudgeConnector")
    monkeypatch.setenv("GAUSSIA_JUDGE_CONNECTOR_KWARGS_JSON", '{"timeout": 15}')
    monkeypatch.setenv("GAUSSIA_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("GAUSSIA_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("GAUSSIA_JUDGE_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("GAUSSIA_JUDGE_TEMPERATURE", "0.3")

    connector = ProviderConfig.from_env().require_judge_model()

    assert isinstance(connector, FakeJudgeConnector)
    assert connector.kwargs == {
        "timeout": 15,
        "temperature": 0.3,
        "model": "judge-model",
        "api_key": "test-key",
        "base_url": "https://judge.example.com/v1",
    }


def test_provider_config_uses_langchain_provider_registry(monkeypatch) -> None:
    calls = {}
    connector = object()

    def fake_init_chat_model(model: str, **kwargs):
        calls["model"] = model
        calls["kwargs"] = kwargs
        return connector

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)

    result = ProviderConfig(
        judge_model="judge-model",
        judge_model_provider="test-provider",
        judge_api_key="test-key",
        judge_base_url="https://judge.example.com/v1",
        judge_temperature=0.4,
    ).require_judge_model()

    assert result is connector
    assert calls == {
        "model": "judge-model",
        "kwargs": {
            "temperature": 0.4,
            "api_key": "test-key",
            "base_url": "https://judge.example.com/v1",
            "model_provider": "test-provider",
        },
    }


def test_provider_config_requires_judge_model_or_connector() -> None:
    with pytest.raises(ValueError, match="GAUSSIA_JUDGE_MODEL or GAUSSIA_JUDGE_CONNECTOR_CLASS"):
        ProviderConfig().require_judge_model()


def _dataset_payload() -> dict:
    return {
        "dataset": {
            "session_id": "session-1",
            "assistant_id": "assistant-1",
            "language": "english",
            "context": "Answer from the support runbook.",
            "conversation": [
                {
                    "qa_id": "q1",
                    "query": "What should I check first?",
                    "assistant": "Verify connectivity.",
                    "ground_truth_assistant": "Verify connectivity.",
                },
                {
                    "qa_id": "q2",
                    "query": "What comes next?",
                    "assistant": "Check client version.",
                    "ground_truth_assistant": "Check client version.",
                },
            ],
        },
        "metadata": {"stream_id": "stream-1", "control_id": "control-1"},
    }
