from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import requests

if TYPE_CHECKING:
    from evalhub.adapter import JobSpec

    from .payloads import BenchmarkInput


LOCAL_MLFLOW_PREFIXES = ("http://localhost", "http://127.0.0.1")
MLFLOW_SOURCE_NAME = "gaussia.integrations.evalhub.adapter"
MLFLOW_SOURCE_TYPE = "JOB"
MODEL_INPUT_UNSUPPORTED_STATUSES = {400, 404, 405, 501}


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

        for key, value in self._build_params(config, benchmark_input).items():
            self._post(
                "/api/2.0/mlflow/runs/log-parameter",
                {"run_id": run_id, "key": key, "value": value},
            )

        dataset_input = self._build_dataset_input(config, benchmark_input)
        model_id = self._try_create_logged_model(
            experiment_id=experiment_id,
            run_id=run_id,
            config=config,
            benchmark_input=benchmark_input,
        )
        if model_id is None:
            self._record_legacy_logged_model(
                run_id=run_id,
                config=config,
                benchmark_input=benchmark_input,
            )
        self._log_inputs(
            run_id=run_id,
            dataset_input=dataset_input,
            model_id=model_id,
        )

        return MLflowRunContext(
            client=self,
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_uri=run["artifact_uri"],
            dataset_name=dataset_input["dataset"]["name"],
            dataset_digest=dataset_input["dataset"]["digest"],
            model_id=model_id,
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
            {"key": "mlflow.source.name", "value": MLFLOW_SOURCE_NAME},
            {"key": "mlflow.source.type", "value": MLFLOW_SOURCE_TYPE},
            {"key": "mlflow.user", "value": _run_user()},
            {"key": "source", "value": benchmark_input.metadata.get("source", benchmark_input.payload_source)},
            {"key": "metric_pack", "value": config.benchmark_id},
            {"key": "provider_id", "value": config.provider_id},
            {"key": "benchmark_id", "value": config.benchmark_id},
            {"key": "evaluation_job_id", "value": config.id},
            {"key": "payload_source", "value": benchmark_input.payload_source},
            {"key": "evaluated_model_name", "value": config.model.name},
        ]
        docker_image = _clean_env("GAUSSIA_PROVIDER_IMAGE") or _clean_env("IMAGE")
        if docker_image:
            tags.append({"key": "mlflow.docker.image.name", "value": docker_image})
        for tag in config.tags:
            key = tag.get("key")
            value = tag.get("value")
            if key is not None and value is not None:
                tags.append({"key": str(key), "value": str(value)})
        return tags

    def _build_params(self, config: JobSpec, benchmark_input: BenchmarkInput) -> dict[str, str]:
        metadata = benchmark_input.metadata
        params = {
            "assistant_id": metadata.get("assistant_id", benchmark_input.dataset.assistant_id),
            "session_id": metadata.get("session_id", benchmark_input.dataset.session_id),
            "stream_id": metadata.get("stream_id", ""),
            "control_id": metadata.get("control_id", ""),
            "agentspace_id": metadata.get("agentspace_id", ""),
            "source": metadata.get("source", benchmark_input.payload_source),
            "evaluated_model_name": config.model.name,
            "evaluated_model_url": config.model.url,
            "judge_model": os.environ.get("GAUSSIA_JUDGE_MODEL", ""),
            "guardian_model": os.environ.get("GAUSSIA_GUARDIAN_MODEL", ""),
            "toxicity_embedding_model": os.environ.get("GAUSSIA_TOXICITY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        }
        return {key: value for key, value in params.items() if value is not None}

    def _build_dataset_input(self, config: JobSpec, benchmark_input: BenchmarkInput) -> dict[str, Any]:
        metadata = benchmark_input.metadata
        dataset = benchmark_input.dataset
        dataset_payload = dataset.model_dump(mode="json", exclude_none=True)
        digest = _digest_json(
            {
                "dataset": dataset_payload,
                "metadata": metadata,
                "payload_source": benchmark_input.payload_source,
            }
        )
        source = {
            "payload_source": benchmark_input.payload_source,
            "source": metadata.get("source", benchmark_input.payload_source),
            "stream_id": metadata.get("stream_id", ""),
            "control_id": metadata.get("control_id", ""),
            "session_id": metadata.get("session_id", dataset.session_id),
            "assistant_id": metadata.get("assistant_id", dataset.assistant_id),
        }
        profile = {
            "num_interactions": len(dataset.conversation),
            "language": dataset.language,
            "assistant_id": metadata.get("assistant_id", dataset.assistant_id),
            "session_id": metadata.get("session_id", dataset.session_id),
            "benchmark_id": config.benchmark_id,
        }
        schema = {
            "type": "gaussia.Dataset",
            "fields": [
                {"name": "qa_id", "type": "string"},
                {"name": "query", "type": "string"},
                {"name": "assistant", "type": "string"},
                {"name": "ground_truth_assistant", "type": "string"},
            ],
        }
        return {
            "dataset": {
                "name": _dataset_name(benchmark_input),
                "digest": digest,
                "source_type": benchmark_input.payload_source,
                "source": _json_dumps(source),
                "schema": _json_dumps(schema),
                "profile": _json_dumps(profile),
            },
            "tags": [
                {"key": "mlflow.data.context", "value": "evaluation"},
                {"key": "benchmark_id", "value": config.benchmark_id},
                {"key": "provider_id", "value": config.provider_id},
                {"key": "payload_source", "value": benchmark_input.payload_source},
            ],
        }

    def _try_create_logged_model(
        self,
        *,
        experiment_id: str,
        run_id: str,
        config: JobSpec,
        benchmark_input: BenchmarkInput,
    ) -> str | None:
        metadata = benchmark_input.metadata
        payload = {
            "experiment_id": experiment_id,
            "name": _model_name(config, benchmark_input),
            "model_type": "Agent",
            "source_run_id": run_id,
            "params": [
                {"key": "evaluated_model_name", "value": config.model.name},
                {"key": "evaluated_model_url", "value": config.model.url},
                {"key": "assistant_id", "value": metadata.get("assistant_id", benchmark_input.dataset.assistant_id)},
                {"key": "session_id", "value": metadata.get("session_id", benchmark_input.dataset.session_id)},
            ],
            "tags": [
                {"key": "provider_id", "value": config.provider_id},
                {"key": "benchmark_id", "value": config.benchmark_id},
                {"key": "payload_source", "value": benchmark_input.payload_source},
            ],
        }
        response = self._post_optional("/api/2.0/mlflow/logged-models", payload)
        if response is None:
            return None

        model = response.get("model", {})
        info = model.get("info", {})
        model_id = info.get("model_id") or model.get("model_id")
        return str(model_id) if model_id else None

    def _record_legacy_logged_model(
        self,
        *,
        run_id: str,
        config: JobSpec,
        benchmark_input: BenchmarkInput,
    ) -> None:
        model_name = _model_name(config, benchmark_input)
        model_json = {
            "run_id": run_id,
            "artifact_path": f"models/{model_name}",
            "utc_time_created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "model_uuid": _digest_json(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "model_url": config.model.url,
                }
            ),
            "flavors": {
                "gaussia_external": {
                    "model_name": config.model.name,
                    "model_url": config.model.url,
                    "assistant_id": benchmark_input.metadata.get(
                        "assistant_id",
                        benchmark_input.dataset.assistant_id,
                    ),
                }
            },
        }
        self._post_optional(
            "/api/2.0/mlflow/runs/log-model",
            {
                "run_id": run_id,
                "model_json": _json_dumps(model_json),
            },
        )

    def _log_inputs(
        self,
        *,
        run_id: str,
        dataset_input: dict[str, Any],
        model_id: str | None,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "datasets": [dataset_input],
        }
        if model_id:
            payload["models"] = [{"model_id": model_id}]
        self._post("/api/2.0/mlflow/runs/log-inputs", payload)

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

    def _post_optional(self, path: str, payload: dict) -> dict | None:
        response = self.session.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._request_headers({"Content-Type": "application/json"}),
            verify=self.verify_tls,
            timeout=30,
        )
        if response.status_code in MODEL_INPUT_UNSUPPORTED_STATUSES:
            self._set_tag_if_possible(
                payload.get("source_run_id", ""),
                "gaussia.mlflow.model_input_status",
                f"unsupported:{response.status_code}",
            )
            return None
        _raise_for_status(response)
        return response.json()

    def _set_tag_if_possible(self, run_id: str, key: str, value: str) -> None:
        if not run_id:
            return
        response = self.session.post(
            f"{self.base_url}/api/2.0/mlflow/runs/set-tag",
            json={"run_id": run_id, "key": key, "value": value},
            headers=self._request_headers({"Content-Type": "application/json"}),
            verify=self.verify_tls,
            timeout=30,
        )
        if response.ok:
            return

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
        dataset_name: str,
        dataset_digest: str,
        model_id: str | None = None,
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.artifact_uri = artifact_uri
        self.dataset_name = dataset_name
        self.dataset_digest = dataset_digest
        self.model_id = model_id

    def log_success(
        self,
        *,
        metrics: list[tuple[str, float]],
        artifact_payload: dict,
    ) -> None:
        for metric_key, metric_value in metrics:
            payload = {
                "run_id": self.run_id,
                "key": metric_key,
                "value": metric_value,
                "timestamp": _now_ms(),
                "step": 0,
                "dataset_name": self.dataset_name,
                "dataset_digest": self.dataset_digest,
            }
            if self.model_id:
                payload["model_id"] = self.model_id
            self.client._post("/api/2.0/mlflow/runs/log-metric", payload)
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


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _run_user() -> str:
    return (
        _clean_env("MLFLOW_RUN_USER")
        or _clean_env("GAUSSIA_PROVIDER_USER")
        or _clean_env("USER")
        or getpass.getuser()
        or "gaussia-provider"
    )


def _dataset_name(benchmark_input: BenchmarkInput) -> str:
    metadata = benchmark_input.metadata
    source = metadata.get("source", benchmark_input.payload_source).replace(".", "-").replace("_", "-")
    session_id = metadata.get("session_id", benchmark_input.dataset.session_id)
    return _safe_name(f"gaussia-{source}-{session_id}")


def _model_name(config: JobSpec, benchmark_input: BenchmarkInput) -> str:
    assistant_id = benchmark_input.metadata.get("assistant_id", benchmark_input.dataset.assistant_id)
    return _safe_name(f"{assistant_id}-{config.model.name}")


def _safe_name(value: str, max_length: int = 250) -> str:
    name = re.sub(r"[^A-Za-z0-9_.@:/+-]+", "-", value).strip("-")
    return (name or "gaussia-evalhub")[:max_length]


def _digest_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
