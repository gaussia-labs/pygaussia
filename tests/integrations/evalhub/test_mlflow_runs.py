from __future__ import annotations

import json

from gaussia.integrations.evalhub.mlflow_runs import MLflowRunLogger
from gaussia.integrations.evalhub.payloads import load_benchmark_input


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None, dict | None, dict]] = []

    def get(self, url: str, *, params=None, headers=None, verify=None, timeout=None):
        self.calls.append(("GET", url, params, None, headers or {}))
        return FakeResponse(payload={"experiment": {"experiment_id": "3", "name": "gaussia-evalhub"}})

    def post(self, url: str, *, json=None, headers=None, verify=None, timeout=None):
        self.calls.append(("POST", url, None, json, headers or {}))
        if url.endswith("/runs/create"):
            return FakeResponse(
                payload={
                    "run": {
                        "info": {
                            "run_id": "run-123",
                            "artifact_uri": "mlflow-artifacts:/workspaces/test/3/run-123/artifacts",
                        }
                    }
                }
            )
        if url.endswith("/logged-models"):
            return FakeResponse(payload={"model": {"info": {"model_id": "model-123"}}})
        return FakeResponse(payload={})

    def put(self, url: str, *, data=None, headers=None, verify=None, timeout=None):
        self.calls.append(("PUT", url, None, None, headers or {}))
        return FakeResponse(payload={})


def test_mlflow_run_logger_uses_workspace_and_creates_run() -> None:
    session = FakeSession()
    logger = MLflowRunLogger(
        base_url="https://mlflow.example.com",
        workspace="redhat-ods-applications",
        auth_token="token-123",
        verify_tls=False,
        session=session,
    )

    class Spec:
        experiment_name = "gaussia-evalhub"
        benchmark_id = "humanity"
        provider_id = "gaussia"
        id = "job-123"
        tags = [{"key": "assistant_id", "value": "assistant-1"}]
        model = type(
            "Model",
            (),
            {
                "name": "assistant-release-ops",
                "url": "https://example.invalid/model",
            },
        )()

    benchmark_input = load_benchmark_input(
        {
            "dataset": {
                "session_id": "session-1",
                "assistant_id": "assistant-1",
                "language": "english",
                "context": "Answer from docs.",
                "conversation": [
                    {
                        "qa_id": "q1",
                        "query": "Question?",
                        "assistant": "Answer.",
                        "ground_truth_assistant": "",
                    }
                ],
            },
            "metadata": {
                "stream_id": "stream-1",
                "control_id": "control-1",
                "agentspace_id": "agentspace-1",
            },
        }
    )

    run = logger.create_run(Spec(), benchmark_input)
    run.log_success(
        metrics=[
            ("humanity_assistant_emotional_entropy", 0.75),
            ("context_awareness", 0.88),
        ],
        artifact_payload={"ok": True},
    )

    assert run.run_id == "run-123"
    assert run.experiment_id == "3"
    assert run.dataset_name == "gaussia-gaussia-dataset-v1-session-1"
    assert run.model_id == "model-123"
    assert session.calls[0][0] == "GET"
    assert session.calls[0][4]["X-MLFLOW-WORKSPACE"] == "redhat-ods-applications"
    assert session.calls[0][4]["Authorization"] == "Bearer token-123"
    run_create = next(call for call in session.calls if call[1].endswith("/api/2.0/mlflow/runs/create"))
    tag_values = {tag["key"]: tag["value"] for tag in run_create[3]["tags"]}
    assert tag_values["mlflow.source.name"] == "gaussia.integrations.evalhub.adapter"
    assert tag_values["mlflow.source.type"] == "JOB"
    assert tag_values["evaluated_model_name"] == "assistant-release-ops"
    inputs = next(call for call in session.calls if call[1].endswith("/api/2.0/mlflow/runs/log-inputs"))
    assert inputs[3]["datasets"][0]["dataset"]["name"] == "gaussia-gaussia-dataset-v1-session-1"
    assert len(inputs[3]["datasets"][0]["dataset"]["digest"]) <= 36
    assert inputs[3]["models"] == [{"model_id": "model-123"}]
    metric_calls = [call for call in session.calls if call[1].endswith("/api/2.0/mlflow/runs/log-metric")]
    assert len(metric_calls) == 2
    assert metric_calls[0][3]["dataset_name"] == "gaussia-gaussia-dataset-v1-session-1"
    assert metric_calls[0][3]["model_id"] == "model-123"
    assert any("/api/2.0/mlflow-artifacts/artifacts/" in call[1] for call in session.calls)
