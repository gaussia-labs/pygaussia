from __future__ import annotations

import pytest

from gaussia.integrations.evalhub.payloads import load_benchmark_input


def test_loads_preferred_dataset_payload() -> None:
    benchmark_input = load_benchmark_input(
        {
            "dataset": {
                "session_id": "session-1",
                "assistant_id": "assistant-1",
                "language": "english",
                "context": "Answer from the product docs.",
                "conversation": [
                    {
                        "qa_id": "q1",
                        "query": "How do I install it?",
                        "assistant": "Run pip install acme.",
                        "ground_truth_assistant": "",
                    }
                ],
            },
            "metadata": {"stream_id": "stream-1", "control_id": "control-1"},
        }
    )

    assert benchmark_input.payload_source == "dataset"
    assert benchmark_input.dataset.session_id == "session-1"
    assert benchmark_input.metadata["assistant_id"] == "assistant-1"
    assert benchmark_input.metadata["stream_id"] == "stream-1"


def test_prefers_dataset_over_legacy_context_persistance() -> None:
    benchmark_input = load_benchmark_input(
        {
            "dataset": {
                "session_id": "preferred-session",
                "assistant_id": "preferred-assistant",
                "language": "english",
                "context": "Preferred context.",
                "conversation": [
                    {
                        "qa_id": "q1",
                        "query": "Question?",
                        "assistant": "Answer.",
                        "ground_truth_assistant": "",
                    }
                ],
            },
            "context_persistance": _legacy_context_persistance_payload(),
        }
    )

    assert benchmark_input.payload_source == "dataset"
    assert benchmark_input.dataset.session_id == "preferred-session"


def test_loads_legacy_context_persistance_payload() -> None:
    benchmark_input = load_benchmark_input({"context_persistance": _legacy_context_persistance_payload()})

    assert benchmark_input.payload_source == "context_persistance"
    assert benchmark_input.dataset.session_id == "session-legacy"
    assert benchmark_input.dataset.assistant_id == "assistant-legacy"
    assert benchmark_input.dataset.conversation[0].query == "Hello"
    assert benchmark_input.dataset.conversation[0].assistant == "Hi there"
    assert benchmark_input.metadata["control_id"] == "control-legacy"


def test_rejects_missing_supported_payload() -> None:
    with pytest.raises(ValueError, match="dataset or context_persistance"):
        load_benchmark_input({})


def _legacy_context_persistance_payload() -> dict:
    return {
        "stream_id": "stream-legacy",
        "control_id": "control-legacy",
        "assistant_id": "assistant-legacy",
        "assistant_context": "Be concise.",
        "conversation": {
            "session_id": "session-legacy",
            "messages": [
                {"type": "human", "data": {"content": "Hello"}},
                {"type": "ai", "data": {"content": "Hi there"}},
            ],
        },
        "agentspace_id": "agentspace-legacy",
    }

