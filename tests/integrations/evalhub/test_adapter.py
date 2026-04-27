from __future__ import annotations

import pytest
from evalhub.adapter import JobCallbacks, JobResults, JobSpec, JobStatusUpdate, OCIArtifactResult

from gaussia.integrations.evalhub.adapter import GaussiaEvalHubAdapter
from gaussia.integrations.evalhub.benchmarks import BenchmarkExecution


class FakeCallbacks(JobCallbacks):
    def __init__(self) -> None:
        self.status_updates: list[JobStatusUpdate] = []
        self.oci_specs = []
        self.reported_results: list[JobResults] = []

    def report_status(self, update: JobStatusUpdate) -> None:
        self.status_updates.append(update)

    def create_oci_artifact(self, spec):
        self.oci_specs.append(spec)
        return OCIArtifactResult(
            digest="sha256:test",
            reference="quay.io/gaussia/evalhub-provider@sha256:test",
        )

    def report_results(self, results: JobResults) -> None:
        self.reported_results.append(results)


class FakeMLflowRun:
    def __init__(self) -> None:
        self.run_id = "run-123"
        self.experiment_id = "exp-123"
        self.success_calls: list[tuple[list[tuple[str, float]], dict]] = []
        self.failure_calls: list[Exception] = []

    def log_success(self, *, metrics: list[tuple[str, float]], artifact_payload: dict) -> None:
        self.success_calls.append((metrics, artifact_payload))

    def log_failure(self, exc: Exception) -> None:
        self.failure_calls.append(exc)


class FakeMLflowLogger:
    def __init__(self) -> None:
        self.calls = []
        self.run = FakeMLflowRun()

    def create_run(self, spec: JobSpec, benchmark_input) -> FakeMLflowRun:
        self.calls.append((spec, benchmark_input))
        return self.run


@pytest.mark.parametrize(
    ("benchmark_id", "primary_metric"),
    [
        ("humanity", "humanity_assistant_emotional_entropy"),
        ("context", "context_awareness"),
        ("conversational", "conversational_sensibleness"),
        ("bias", "bias_score"),
        ("toxicity", "toxicity_didt"),
    ],
)
def test_provider_dispatches_supported_benchmarks(monkeypatch, benchmark_id: str, primary_metric: str) -> None:
    adapter = build_adapter()
    callbacks = FakeCallbacks()

    monkeypatch.setattr(
        "gaussia.integrations.evalhub.adapter.run_gaussia_benchmark",
        lambda **_: _execution(benchmark_id=benchmark_id, primary_metric=primary_metric),
    )

    results = adapter.run_benchmark_job(build_job_spec(benchmark_id=benchmark_id), callbacks)

    assert [update.phase.value for update in callbacks.status_updates] == [
        "initializing",
        "running_evaluation",
    ]
    assert results.benchmark_id == benchmark_id
    assert results.evaluation_metadata["primary_metric_name"] == primary_metric
    assert results.evaluation_metadata["payload_source"] == "dataset"


def test_provider_accepts_legacy_context_persistance(monkeypatch) -> None:
    adapter = build_adapter()
    callbacks = FakeCallbacks()

    monkeypatch.setattr(
        "gaussia.integrations.evalhub.adapter.run_gaussia_benchmark",
        lambda **_: _execution(
            benchmark_id="humanity",
            primary_metric="humanity_assistant_emotional_entropy",
        ),
    )

    results = adapter.run_benchmark_job(build_job_spec(parameters={"context_persistance": legacy_payload()}), callbacks)

    assert results.evaluation_metadata["payload_source"] == "context_persistance"
    assert results.evaluation_metadata["stream_id"] == "stream-legacy"


def test_provider_creates_mlflow_run_when_logger_is_available(monkeypatch) -> None:
    adapter = build_adapter()
    callbacks = FakeCallbacks()
    fake_logger = FakeMLflowLogger()
    monkeypatch.setattr(
        "gaussia.integrations.evalhub.adapter.build_mlflow_run_logger_from_env",
        lambda: fake_logger,
    )
    monkeypatch.setattr(
        "gaussia.integrations.evalhub.adapter.run_gaussia_benchmark",
        lambda **_: _execution(
            benchmark_id="humanity",
            primary_metric="humanity_assistant_emotional_entropy",
        ),
    )

    results = adapter.run_benchmark_job(build_job_spec(), callbacks)

    assert len(fake_logger.calls) == 1
    assert fake_logger.run.success_calls[0][0][0][0] == "humanity_assistant_emotional_entropy"
    assert results.evaluation_metadata["mlflow_run_id"] == "run-123"
    assert results.mlflow_run_id == "run-123"


def test_provider_creates_oci_artifact_when_export_is_present(monkeypatch) -> None:
    adapter = build_adapter()
    callbacks = FakeCallbacks()
    monkeypatch.setattr(
        "gaussia.integrations.evalhub.adapter.run_gaussia_benchmark",
        lambda **_: _execution(
            benchmark_id="humanity",
            primary_metric="humanity_assistant_emotional_entropy",
        ),
    )

    results = adapter.run_benchmark_job(build_job_spec(with_exports=True), callbacks)

    assert [update.phase.value for update in callbacks.status_updates] == [
        "initializing",
        "running_evaluation",
        "persisting_artifacts",
    ]
    assert len(callbacks.oci_specs) == 1
    assert results.oci_artifact is not None
    assert results.evaluation_metadata["artifact_generated"] is True


def test_provider_rejects_unknown_benchmark() -> None:
    adapter = build_adapter()

    with pytest.raises(ValueError, match="Unsupported Gaussia benchmark"):
        adapter.run_benchmark_job(build_job_spec(benchmark_id="regulatory"), FakeCallbacks())


def test_provider_rejects_invalid_payload() -> None:
    adapter = build_adapter()

    with pytest.raises(ValueError, match="dataset or context_persistance"):
        adapter.run_benchmark_job(build_job_spec(parameters={}), FakeCallbacks())


def build_job_spec(
    *,
    benchmark_id: str = "humanity",
    parameters: dict | None = None,
    with_exports: bool = False,
) -> JobSpec:
    data = {
        "id": "job-123",
        "provider_id": "gaussia",
        "benchmark_id": benchmark_id,
        "benchmark_index": 0,
        "model": {
            "name": "test-model",
            "url": "https://example.invalid/model",
        },
        "parameters": (
            parameters
            if parameters is not None
            else {"dataset": dataset_payload(), "metadata": {"stream_id": "stream-1"}}
        ),
        "callback_url": "http://localhost:8080/callbacks",
    }
    if with_exports:
        data["exports"] = {
            "oci": {
                "coordinates": {
                    "oci_host": "quay.io",
                    "oci_repository": "gaussia/evalhub-provider",
                    "oci_tag": "job-123",
                    "annotations": {},
                }
            }
        }
    return JobSpec.model_validate(data)


def build_adapter() -> GaussiaEvalHubAdapter:
    return object.__new__(GaussiaEvalHubAdapter)


def dataset_payload() -> dict:
    return {
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
    }


def legacy_payload() -> dict:
    return {
        "stream_id": "stream-legacy",
        "control_id": "control-legacy",
        "assistant_id": "assistant-legacy",
        "assistant_context": "Be concise.",
        "conversation": {
            "session_id": "session-legacy",
            "messages": [
                {"type": "human", "data": {"content": "Hello"}},
                {"type": "ai", "data": {"content": "Hi"}},
            ],
        },
    }


def _execution(*, benchmark_id: str, primary_metric: str) -> BenchmarkExecution:
    from evalhub.adapter import EvaluationResult

    return BenchmarkExecution(
        benchmark_id=benchmark_id,
        primary_metric_name=primary_metric,
        primary_metric_value=0.81,
        overall_score=0.81,
        interaction_count=1,
        metric_count=1,
        evaluation_results=[
            EvaluationResult(
                metric_name=primary_metric,
                metric_value=0.81,
                metric_type="float",
            )
        ],
        artifact_payload={"summary": {primary_metric: 0.81}},
    )
