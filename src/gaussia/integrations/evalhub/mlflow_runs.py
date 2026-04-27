from __future__ import annotations

import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from urllib.parse import quote

import requests

if TYPE_CHECKING:
    from evalhub.adapter import JobSpec

    from .payloads import BenchmarkInput


LOCAL_MLFLOW_PREFIXES = ("http://localhost", "http://127.0.0.1")


def build_mlflow_run_logger_from_env() -> MLflowRunLogger | None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    auth_token = os.environ.get("MLFLOW_TRACKING_TOKEN")
    token_path = os.environ.get("MLFLOW_TRACKING_TOKEN_PATH") or os.environ.get("MLFLOW_TOKEN_PATH")
    workspace = os.environ.get("MLFLOW_WORKSPACE")
    if workspace and tracking_uri.startswith(LOCAL_MLFLOW_PREFIXES):
        tracking_uri = f"https://mlflow.{workspace}.svc:8443"
        insecure = True
    else:
        insecure = tracking_uri.startswith(LOCAL_MLFLOW_PREFIXES)
    return MLflowRunLogger(
        base_url=tracking_uri,
        workspace=workspace,
        auth_token=auth_token,
        auth_token_path=token_path,
        verify_tls=not insecure,
    )


class MLflowRunLogger:
    def __init__(
        self,
        *,
        base_url: str,
        workspace: str | None = None,
        auth_token: str | None = None,
        auth_token_path: str | None = None,
        verify_tls: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.workspace = workspace
        self.auth_token = auth_token or _read_token(auth_token_path)
        self.verify_tls = verify_tls
        self.session = session or requests.Session()

    def create_run(self, config: JobSpec, benchmark_input: BenchmarkInput) -> MLflowRunContext:
        experiment_id = self._ensure_experiment(config.experiment_name)
        metadata = benchmark_input.metadata
        run = self._post(
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": experiment_id,
                "start_time": _now_ms(),
                "run_name": f"{config.benchmark_id}-{metadata.get('control_id', config.id)}",
                "tags": self._build_tags(config, benchmark_input),
            },
        )["run"]["info"]
        run_id = run["run_id"]

        for key, value in self._build_params(benchmark_input).items():
            self._post(
                "/api/2.0/mlflow/runs/log-parameter",
                {"run_id": run_id, "key": key, "value": value},
            )

        return MLflowRunContext(
            client=self,
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_uri=run["artifact_uri"],
        )

    def _ensure_experiment(self, experiment_name: str) -> str:
        response = self._get(
            "/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
            allow_not_found=True,
        )
        if response is not None:
            return response["experiment"]["experiment_id"]

        created = self._post("/api/2.0/mlflow/experiments/create", {"name": experiment_name})
        return created["experiment_id"]

    def _build_tags(self, config: JobSpec, benchmark_input: BenchmarkInput) -> list[dict[str, str]]:
        tags = [
            {"key": "source", "value": benchmark_input.metadata.get("source", benchmark_input.payload_source)},
            {"key": "metric_pack", "value": config.benchmark_id},
            {"key": "provider_id", "value": config.provider_id},
            {"key": "benchmark_id", "value": config.benchmark_id},
            {"key": "evaluation_job_id", "value": config.id},
            {"key": "payload_source", "value": benchmark_input.payload_source},
        ]
        for tag in config.tags:
            key = tag.get("key")
            value = tag.get("value")
            if key is not None and value is not None:
                tags.append({"key": str(key), "value": str(value)})
        return tags

    def _build_params(self, benchmark_input: BenchmarkInput) -> dict[str, str]:
        metadata = benchmark_input.metadata
        return {
            "assistant_id": metadata.get("assistant_id", benchmark_input.dataset.assistant_id),
            "session_id": metadata.get("session_id", benchmark_input.dataset.session_id),
            "stream_id": metadata.get("stream_id", ""),
            "control_id": metadata.get("control_id", ""),
            "agentspace_id": metadata.get("agentspace_id", ""),
            "source": metadata.get("source", benchmark_input.payload_source),
        }

    def _upload_json_artifact(
        self,
        *,
        artifact_uri: str,
        artifact_path: str,
        payload: dict,
    ) -> None:
        with NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as temp:
            temp.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
            temp_path = Path(temp.name)
        try:
            self._upload_file_artifact(
                artifact_uri=artifact_uri,
                artifact_path=artifact_path,
                local_path=temp_path,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _upload_file_artifact(
        self,
        *,
        artifact_uri: str,
        artifact_path: str,
        local_path: Path,
    ) -> None:
        endpoint = _artifact_endpoint_from_uri(artifact_uri, artifact_path)
        mime_type = "application/json" if local_path.suffix == ".json" else "application/octet-stream"
        with local_path.open("rb") as handle:
            response = self.session.put(
                f"{self.base_url}{endpoint}",
                data=handle,
                headers={"Content-Type": mime_type, **self._request_headers({})},
                verify=self.verify_tls,
                timeout=30,
            )
        _raise_for_status(response)

    def _get(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> dict | None:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._request_headers({}),
            verify=self.verify_tls,
            timeout=30,
        )
        if allow_not_found and response.status_code in {400, 404} and "RESOURCE_DOES_NOT_EXIST" in response.text:
            return None
        _raise_for_status(response)
        return response.json()

    def _post(self, path: str, payload: dict) -> dict:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._request_headers({"Content-Type": "application/json"}),
            verify=self.verify_tls,
            timeout=30,
        )
        _raise_for_status(response)
        return response.json()

    def _request_headers(self, extra: dict[str, str]) -> dict[str, str]:
        headers = dict(extra)
        if self.workspace:
            headers["X-MLFLOW-WORKSPACE"] = self.workspace
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers


class MLflowRunContext:
    def __init__(
        self,
        *,
        client: MLflowRunLogger,
        experiment_id: str,
        run_id: str,
        artifact_uri: str,
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.artifact_uri = artifact_uri

    def log_success(
        self,
        *,
        metrics: list[tuple[str, float]],
        artifact_payload: dict,
    ) -> None:
        for metric_key, metric_value in metrics:
            self.client._post(
                "/api/2.0/mlflow/runs/log-metric",
                {
                    "run_id": self.run_id,
                    "key": metric_key,
                    "value": metric_value,
                    "timestamp": _now_ms(),
                    "step": 0,
                },
            )
        self.client._upload_json_artifact(
            artifact_uri=self.artifact_uri,
            artifact_path="results/summary.json",
            payload=artifact_payload,
        )
        self.client._post(
            "/api/2.0/mlflow/runs/update",
            {
                "run_id": self.run_id,
                "status": "FINISHED",
                "end_time": _now_ms(),
            },
        )

    def log_failure(self, exc: Exception) -> None:
        self.client._post(
            "/api/2.0/mlflow/runs/set-tag",
            {
                "run_id": self.run_id,
                "key": "error",
                "value": str(exc),
            },
        )
        self.client._post(
            "/api/2.0/mlflow/runs/update",
            {
                "run_id": self.run_id,
                "status": "FAILED",
                "end_time": _now_ms(),
            },
        )


def _artifact_endpoint_from_uri(artifact_uri: str, artifact_path: str) -> str:
    if not artifact_uri.startswith("mlflow-artifacts:/"):
        raise RuntimeError(f"Unsupported artifact URI returned by MLflow: {artifact_uri}")

    base_path = artifact_uri.removeprefix("mlflow-artifacts:/").strip("/")
    combined = "/".join(part.strip("/") for part in (base_path, artifact_path) if part and part.strip("/"))
    return f"/api/2.0/mlflow-artifacts/artifacts/{quote(combined, safe='/')}"


def _read_token(token_path: str | None) -> str | None:
    if not token_path:
        return None
    path = Path(token_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    raise RuntimeError(f"MLflow request failed: {response.status_code} {payload}")
