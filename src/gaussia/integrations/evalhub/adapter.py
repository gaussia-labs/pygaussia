from __future__ import annotations
# ruff: noqa: I001

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from . import _warnings as _evalhub_warnings  # noqa: F401
from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
)

from .artifacts import write_json_artifact
from .benchmarks import SUPPORTED_BENCHMARK_IDS, run_gaussia_benchmark
from .config import ProviderConfig
from .mlflow_runs import build_mlflow_run_logger_from_env
from .payloads import load_benchmark_input


class GaussiaEvalHubAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        started_at = time.monotonic()
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.05,
                message=MessageInfo(
                    message="Loading EvalHub payload for Gaussia",
                    message_code="initializing",
                ),
            )
        )

        if config.benchmark_id not in SUPPORTED_BENCHMARK_IDS:
            raise ValueError(f"Unsupported Gaussia benchmark: {config.benchmark_id}")

        benchmark_input = load_benchmark_input(config.parameters)
        provider_config = ProviderConfig.from_env()
        mlflow_logger = build_mlflow_run_logger_from_env()
        mlflow_run = mlflow_logger.create_run(config, benchmark_input) if mlflow_logger else None

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.35,
                    message=MessageInfo(
                        message=f"Running Gaussia {config.benchmark_id} benchmark",
                        message_code="running_evaluation",
                    ),
                )
            )
            execution = run_gaussia_benchmark(
                benchmark_id=config.benchmark_id,
                benchmark_input=benchmark_input,
                provider_id=config.provider_id,
                config=provider_config,
            )

            if mlflow_run is not None:
                mlflow_run.log_success(
                    metrics=[
                        (result.metric_name, float(result.metric_value))
                        for result in execution.evaluation_results
                    ],
                    artifact_payload=execution.artifact_payload,
                )

            oci_artifact = None
            if config.exports and config.exports.oci:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.PERSISTING_ARTIFACTS,
                        progress=0.85,
                        message=MessageInfo(
                            message=f"Persisting {config.benchmark_id} artifact as OCI",
                            message_code="persisting_artifacts",
                        ),
                    )
                )
                with TemporaryDirectory(prefix="gaussia-provider-") as temp_dir:
                    artifact_path = Path(temp_dir) / "results.json"
                    write_json_artifact(artifact_path, execution.artifact_payload)
                    coordinates = config.exports.oci.coordinates.model_copy(deep=True)
                    coordinates.annotations.update(
                        {
                            "io.github.eval-hub.provider_id": config.provider_id,
                            "io.github.eval-hub.benchmark_id": config.benchmark_id,
                            "io.github.gaussia.payload_source": benchmark_input.payload_source,
                            "io.github.gaussia.assistant_id": benchmark_input.metadata.get(
                                "assistant_id",
                                benchmark_input.dataset.assistant_id,
                            ),
                            "io.github.gaussia.session_id": benchmark_input.metadata.get(
                                "session_id",
                                benchmark_input.dataset.session_id,
                            ),
                        }
                    )
                    oci_artifact = callbacks.create_oci_artifact(
                        OCIArtifactSpec(
                            files_path=artifact_path.parent,
                            coordinates=coordinates,
                        )
                    )

            metadata = benchmark_input.metadata
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=execution.evaluation_results,
                overall_score=execution.overall_score,
                num_examples_evaluated=execution.interaction_count,
                duration_seconds=round(time.monotonic() - started_at, 3),
                evaluation_metadata={
                    "assistant_id": metadata.get("assistant_id", benchmark_input.dataset.assistant_id),
                    "session_id": metadata.get("session_id", benchmark_input.dataset.session_id),
                    "stream_id": metadata.get("stream_id", ""),
                    "control_id": metadata.get("control_id", ""),
                    "metric_count": execution.metric_count,
                    "interaction_count": execution.interaction_count,
                    "primary_metric_name": execution.primary_metric_name,
                    "primary_metric_value": execution.primary_metric_value,
                    "payload_source": benchmark_input.payload_source,
                    "artifact_generated": bool(config.exports and config.exports.oci),
                    "mlflow_run_id": mlflow_run.run_id if mlflow_run else None,
                    "mlflow_experiment_id": mlflow_run.experiment_id if mlflow_run else None,
                },
                oci_artifact=oci_artifact,
                mlflow_run_id=mlflow_run.run_id if mlflow_run else None,
            )
        except Exception as exc:
            if mlflow_run is not None:
                mlflow_run.log_failure(exc)
            raise


def main() -> None:
    adapter = GaussiaEvalHubAdapter()
    callbacks = DefaultCallbacks.from_adapter(adapter)
    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    callbacks.report_results(results)


if __name__ == "__main__":
    main()
