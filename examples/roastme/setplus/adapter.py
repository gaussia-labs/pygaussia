from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

from gaussia.core.target_assistant import TargetAssistant
from gaussia.schemas.roastme import TargetResponse

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path

    from configuration import TargetConfig, ToolAuditConfig


def _drive(coroutine: Coroutine[Any, Any, TargetResponse]) -> TargetResponse:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _nested_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            names.add(name)
        for nested in value.values():
            names.update(_nested_names(nested))
    elif isinstance(value, list):
        for nested in value:
            names.update(_nested_names(nested))
    return names


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for nested in value.values() for text in _nested_strings(nested)]
    if isinstance(value, list):
        return [text for nested in value for text in _nested_strings(nested)]
    return []


def _event_content(event: dict[str, Any]) -> str:
    payload = event.get("data") or event.get("result")
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    for key in ("content", "value", "text", "message", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != "{}":
            return value.strip()
    return ""


class KapsoWebhookAssistant(TargetAssistant):
    def __init__(
        self,
        target: TargetConfig,
        audit: ToolAuditConfig,
        assistant_id: str,
        api_token: str,
        webhook_secret: str,
        channel_account_id: str,
        actor_subjects: list[str],
    ) -> None:
        self._target = target
        self._audit = audit
        self._assistant_id = assistant_id
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self._webhook_secret = webhook_secret.encode()
        self._channel_account_id = channel_account_id
        self._actor_subjects = iter(actor_subjects)
        self.audits: list[dict[str, Any]] = []

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        try:
            subject = next(self._actor_subjects)
            return _drive(self._exchange(query, subject))
        except Exception as error:
            return TargetResponse(
                content="",
                failed=True,
                failure_reason=f"{type(error).__name__}: {error}",
            )

    async def _exchange(self, query: str, subject: str) -> TargetResponse:
        message_id = f"roastme-{uuid.uuid4().hex}"
        body = json.dumps(
            {
                "event": "whatsapp.message.received",
                "data": {
                    "phone_number_id": self._channel_account_id,
                    "message": {
                        "id": message_id,
                        "from": subject,
                        "type": "text",
                        "text": {"body": query},
                        "kapso": {"direction": "inbound"},
                    },
                    "conversation": {
                        "phone_number_id": self._channel_account_id,
                        "phone_number": subject,
                    },
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        signature = hmac.new(self._webhook_secret, body, hashlib.sha256).hexdigest()
        task_audit: dict[str, Any] = {
            "probe_number": len(self.audits) + 1,
            "task_id": None,
            "session_id": None,
            "exposed_tools": [],
            "invoked_tools": [],
            "tool_errors": [],
        }
        timeout = httpx.Timeout(
            self._target.request_timeout_seconds,
            read=self._target.stream_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                (f"{self._target.runtime_url.rstrip('/')}/event/infer/{self._assistant_id}/{self._target.channel_id}"),
                params={"agentspace_id": self._target.agentspace_id},
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": f"sha256={signature}",
                },
            )
            response.raise_for_status()
            metadata = response.json()
            task_id = metadata["taskid"]
            task_audit["task_id"] = task_id
            task_audit["session_id"] = metadata["sessionid"]
            content, blocked = await self._stream(client, task_id, task_audit)

        self.audits.append(task_audit)
        if blocked:
            return TargetResponse(
                content="",
                failed=True,
                failure_reason=f"agent blocked on {blocked}",
                session_id=metadata["sessionid"],
                raw=metadata,
            )
        if not content:
            return TargetResponse(
                content="",
                failed=True,
                failure_reason="empty stream",
                session_id=metadata["sessionid"],
                raw=metadata,
            )
        return TargetResponse(
            content=content,
            session_id=metadata["sessionid"],
            raw=metadata,
        )

    async def _stream(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        task_audit: dict[str, Any],
    ) -> tuple[str, str | None]:
        content = ""
        exposed: set[str] = set()
        invoked: set[str] = set()
        errors: list[str] = []
        async with client.stream(
            "GET",
            f"{self._target.runtime_url.rstrip('/')}/event/stream/{task_id}",
            headers=self._headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                event_class = event.get("event_class") or event.get("eventClass")
                if event_class in self._audit.blocking_events:
                    return "", str(event_class)
                if event_class == "ToolSchemaResponse":
                    exposed.update(_nested_names(event.get("data")))
                if event_class in {"ResponseInferenceResponse", "ServerToolExecution"}:
                    invoked.update(_nested_names(event.get("data")))
                if event_class == "ToolExecutionResponse" and event.get("status") == "error":
                    errors.extend(
                        text
                        for text in _nested_strings(event.get("data"))
                        if "error" in text.lower() or "failed" in text.lower()
                    )
                extracted = _event_content(event)
                if event_class == "ResponseInferenceResponse" and extracted:
                    content = extracted
                if event_class == "AssistantInferenceResponse":
                    content = extracted or content
                    break
        task_audit["exposed_tools"] = sorted(exposed)
        task_audit["invoked_tools"] = sorted(invoked)
        task_audit["tool_errors"] = list(dict.fromkeys(errors))
        return content, None


class ReplayAssistant(TargetAssistant):
    def __init__(self, answers: dict[str, str], audits: list[dict[str, Any]]) -> None:
        self._answers = answers
        self.audits = audits

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        return TargetResponse(content=self._answers[query], session_id=session_id)

    @classmethod
    def from_run(cls, run_dir: Path) -> ReplayAssistant:
        dataset = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
        answers = {turn["query"]: turn["assistant"] for turn in dataset["conversation"]}
        audits = json.loads((run_dir / "transport-audit.json").read_text(encoding="utf-8"))
        return cls(answers, audits)
