from __future__ import annotations

import pytest

from gaussia.integrations.evalhub.benchmarks import run_gaussia_benchmark
from gaussia.integrations.evalhub.config import ProviderConfig
from gaussia.integrations.evalhub.payloads import load_benchmark_input
from gaussia.schemas.agentic import AgenticMetric


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
