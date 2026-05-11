from __future__ import annotations

import os
from typing import Any, Protocol

from .extensions import load_dotted_object, load_json_object_env


class EvalHubRunContext(Protocol):
    run_id: str | None
    experiment_id: str | None

    def log_success(self, *, metrics: list[tuple[str, float]], artifact_payload: dict) -> None: ...

    def log_failure(self, exc: Exception) -> None: ...


class EvalHubRunLogger(Protocol):
    def create_run(self, config: Any, benchmark_input: Any) -> EvalHubRunContext: ...


def build_run_logger_from_env() -> EvalHubRunLogger | None:
    factory_path = os.environ.get("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY")
    if factory_path:
        kwargs = load_json_object_env("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY_KWARGS_JSON") or {}
        return load_dotted_object(factory_path)(**kwargs)

    from .mlflow_runs import build_mlflow_run_logger_from_env

    return build_mlflow_run_logger_from_env()
