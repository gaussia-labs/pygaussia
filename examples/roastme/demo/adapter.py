"""Target adapters. Two of them, and the difference is only where the answers come from.

`ReplayAssistant` answers from a table, so the whole run works with no runtime and no credentials.
That is not a separate mode of the framework — a recorded response set is an implementation of the
same interface as a live one, which is what makes rehearsing offline the same code path. The table
comes from a previous run's Roast Dataset (`persistence.load_replay_answers`), so a rehearsal replays
what the assistant actually answered instead of a fixture somebody wrote.

`AlquimiaAssistant` is the live one, ported from `roast-me/exploiter/src/roastme/connectors/
alquimia.py`. Three things changed in the port and each is deliberate:

* the connector is `async` and `TargetAssistant.send` is not, so the coroutine is driven by
  `asyncio.run`. The interface anticipates this: a user with an async transport wraps it here.
* the connector returns `content=""` and logs a warning when the stream yields nothing. Gaussia's
  `TargetResponse` refuses empty content unless it is marked failed, so that case becomes
  `failed=True`. It has to: an empty answer graded as compliance is a silent pass.
* a fresh `session_id` per exchange. The connector takes one from its caller, and reusing it would
  let probe five answer with probes one to four still in context.

The live path has run against `assistant-bpd` in production with zero transport failures. What it has
*not* been exercised on is its own failure branches: no run has yet produced a blocking event, an
empty stream or a timeout, so those three are reasoned rather than observed.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Coroutine

from gaussia.core.target_assistant import TargetAssistant
from gaussia.schemas.roastme import TargetResponse

_FINAL_EVENT = "AssistantInferenceResponse"
_PARTIAL_EVENT = "ResponseInferenceResponse"
# The agent can stop and wait for a person. That is not an answer, so it is reported as a failure.
_BLOCKING_EVENTS = frozenset({"HumanApprovalRequired", "ClientToolExecution"})


def _drive(coroutine: Coroutine[Any, Any, TargetResponse]) -> TargetResponse:
    """Corre la corutina, haya o no un event loop andando ya.

    `TargetAssistant.send` es sincrónico y el connector es `async`, así que alguien tiene que manejar el
    loop. `asyncio.run` alcanza desde un script, y **falla dentro de un kernel de Jupyter**: ipykernel ya
    tiene un loop corriendo y `asyncio.run` levanta `RuntimeError: cannot be called from a running event
    loop`.

    Eso importa porque el entregable es un notebook. Y el modo en que fallaba era el peor posible: el
    `except` de `send` convertía el RuntimeError en `failed=True`, así que las 60 consultas volvían como
    fallos de transporte, el Profiler las registraba ungraded —correctamente, una caída no puede leerse
    como buen comportamiento— y la corrida terminaba en tres segundos con un dataset vacío y sin un error
    a la vista.

    La salida es un hilo con su propio loop, en lugar de `nest_asyncio`: no agrega dependencia y no
    parchea el loop de nadie.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _pick(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """The runtime answers in snake_case, flatcase and camelCase depending on the field."""
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return default


def _extract_content(event: dict[str, Any]) -> str:
    for key in ("data", "result"):
        payload = event.get(key)
        if isinstance(payload, str) and payload.strip() and payload.strip() != "{}":
            return payload
        if isinstance(payload, dict):
            text = _pick(payload, "content", "text", "message", "answer")
            if isinstance(text, str) and text.strip():
                return text
    return ""


class ReplayAssistant(TargetAssistant):
    """Answers from a table. A query with no entry is a transport failure, not an empty answer."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        answer = self._answers.get(query)
        if answer is None:
            return TargetResponse(
                content="",
                failed=True,
                failure_reason="no recorded answer for this query",
                session_id=session_id,
            )
        return TargetResponse(content=answer, session_id=session_id)


class AlquimiaAssistant(TargetAssistant):
    """The live runtime: POST event/infer, then read the SSE stream of the task it returns."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        assistant_id: str,
        agentspace_id: str = "default",
        user_id: str = "roastme",
        timeout: float = 60.0,
        stream_timeout: float = 300.0,
    ) -> None:
        self._assistant_id = assistant_id
        self._agentspace_id = agentspace_id
        self._user_id = user_id
        self._base_url = base_url.rstrip("/") + "/"
        self._headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self._timeout = httpx.Timeout(timeout, read=stream_timeout)

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        coroutine = self._exchange(query, session_id or str(uuid.uuid4()))
        try:
            return _drive(coroutine)
        except Exception as error:
            coroutine.close()
            return TargetResponse(content="", failed=True, failure_reason=f"{type(error).__name__}: {error}")

    async def _exchange(self, query: str, session_id: str) -> TargetResponse:
        async with httpx.AsyncClient(base_url=self._base_url, headers=self._headers, timeout=self._timeout) as client:
            answer = await client.post(
                f"event/infer/{self._assistant_id}",
                params={"agentspace_id": self._agentspace_id},
                json={"query": query, "user_id": self._user_id, "session_id": session_id},
            )
            answer.raise_for_status()
            meta = answer.json()
            task_id = _pick(meta, "task_id", "taskid", "taskId")
            if not task_id:
                return TargetResponse(content="", failed=True, failure_reason=f"no task id in {meta}")

            content, blocked = await self._stream(client, str(task_id))

        if blocked:
            return TargetResponse(content="", failed=True, failure_reason=f"agent blocked on {blocked}")
        if not content.strip():
            return TargetResponse(content="", failed=True, failure_reason="empty stream")
        return TargetResponse(
            content=content,
            session_id=str(_pick(meta, "session_id", "sessionid", "sessionId", default=session_id)),
            raw=meta,
        )

    async def _stream(self, client: httpx.AsyncClient, task_id: str) -> tuple[str, str | None]:
        content = ""
        seen: list[dict[str, Any]] = []
        async with client.stream("GET", f"event/stream/{task_id}") as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                seen.append(event)
                event_class = _pick(event, "event_class", "eventClass", "type")
                if event_class in _BLOCKING_EVENTS:
                    return "", str(event_class)
                extracted = _extract_content(event)
                if event_class == _FINAL_EVENT:
                    return extracted or content, None
                if event_class == _PARTIAL_EVENT and extracted:
                    content = extracted
        if not content:
            content = next((text for event in reversed(seen) if (text := _extract_content(event))), "")
        return content, None
