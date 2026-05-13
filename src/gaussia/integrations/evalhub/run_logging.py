from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, cast

from .extensions import load_dotted_object, load_json_object_env

if TYPE_CHECKING:
    from evalhub.adapter import JobSpec

    from .payloads import BenchmarkInput


class EvalHubRunContext(Protocol):
    run_id: str | None
    experiment_id: str | None

    def log_success(self, *, metrics: list[tuple[str, float]], artifact_payload: dict) -> None: ...

    def log_failure(self, exc: Exception) -> None: ...


class EvalHubRunLogger(Protocol):
    def create_run(self, config: JobSpec, benchmark_input: BenchmarkInput) -> EvalHubRunContext: ...


def build_run_logger_from_env() -> EvalHubRunLogger | None:
    factory_path = os.environ.get("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY")
    if factory_path:
        kwargs = load_json_object_env("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY_KWARGS_JSON") or {}
        return cast("EvalHubRunLogger", load_dotted_object(factory_path)(**kwargs))

    from .mlflow_runs import build_mlflow_run_logger_from_env

    return cast("EvalHubRunLogger | None", build_mlflow_run_logger_from_env())
