"""Lee la traza del MCP que el stream del runtime ya trae, y que el adapter del demo descartaba.

Por qué existe este módulo. El agente bajo prueba no tiene RAG: su conocimiento entra por un MCP
contra un cerebro de Boltzmann. Eso cambia lo que se puede afirmar de una falla. En un asistente
RAG nadie sabe qué se recuperó, así que el `NearMissVerifier` **adivina** el boundary con distancia
de edición y umbrales de similitud, y su propio test admite que esos umbrales sirven para nombres
cortos y fallan en entidades con forma de oración.

Acá no hace falta adivinar. El stream SSE trae, en el mismo intercambio HTTP que la respuesta:

* `ResponseInferenceResponse` con `tool_calls` — qué tool llamó el agente y con qué parámetros.
  Es lo único que permite ver una falla de *uso de herramienta*: buscar con `memory_types`
  equivocado, conformarse con un `limit` bajo, o dejar de buscar antes de encontrar. RoastMe no
  modela esa superficie: gradúa el texto final.
* `ToolExecutionResponse` con los `block_id` que el cerebro devolvió — el `sha256` exacto de cada
  bloque que el agente leyó antes de contestar.

Los `block_id` son lo que convierte un `doc=0` de veredicto en hipótesis verificable. Cuando la
corrida diga "inventó X", se puede chequear en orden: ¿X estaba en los bloques que trajo?, ¿está
en el corpus?, ¿existe en el cerebro fuera del corpus? Un falso positivo queda detectable uno por
uno, en vez de quedar promediado dentro de un número.

Y hay una tool de **escritura**, `capture_interaction`, que commitea un bloque episódico y
reindexa en el acto — el comentario del propio MCP dice que es para que "cualquier search que el
MISMO servicio haga después" lo vea. Si el agente la llama durante una corrida, el probe N+1 puede
recuperar lo que escribió el probe N, y el snapshot deja de ser inmutable mientras se mide contra
él. `TOOLS_DE_ESCRITURA` está acá para que eso se detecte y no se descubra después.
"""

from __future__ import annotations

from typing import Any

TOOLS_DE_ESCRITURA = frozenset({"capture_interaction"})
"""Las tools del MCP que mutan el cerebro. Una llamada a cualquiera contamina la corrida."""

_RESPUESTA_INFERENCIA = "ResponseInferenceResponse"
_RESPUESTA_TOOL = "ToolExecutionResponse"


def leer_traza(eventos: list[dict[str, Any]]) -> dict[str, Any]:
    """Qué tools llamó el agente, con qué parámetros, y qué bloques le devolvió el cerebro.

    Args:
        eventos: Los eventos SSE del intercambio, en orden de llegada.

    Returns:
        Un dict apto para `TargetResponse.raw`: `tool_calls`, `block_ids`, `pasos` y `escrituras`.
        Todo plano y serializable, porque de ahí baja al artefacto de findings.
    """
    llamadas: list[dict[str, Any]] = []
    bloques: list[str] = []

    for evento in eventos:
        clase = evento.get("event_class")
        datos = evento.get("data")
        if not isinstance(datos, dict):
            continue
        if clase == _RESPUESTA_INFERENCIA:
            for llamada in datos.get("tool_calls") or []:
                llamadas.append({"tool": llamada.get("name"), "args": llamada.get("args") or {}})
        elif clase == _RESPUESTA_TOOL:
            bloques.extend(_bloques_de(datos))

    escrituras = [llamada["tool"] for llamada in llamadas if llamada["tool"] in TOOLS_DE_ESCRITURA]
    return {
        "tool_calls": llamadas,
        "block_ids": _sin_repetir(bloques),
        "pasos": len(llamadas),
        "escrituras": escrituras,
    }


def _bloques_de(datos: dict[str, Any]) -> list[str]:
    """Los `block_id` del payload de una tool, sea del `search` o del `resolve`.

    El MCP contesta en el envoltorio de FastMCP: `content` es una lista de partes, y la de texto
    trae el JSON de vuelta. `search` devuelve `{"matches": [...]}` y `resolve` devuelve un bloque
    suelto, así que las dos formas se leen acá y no en el llamador.
    """
    import json

    encontrados: list[str] = []
    for parte in datos.get("content") or []:
        if not isinstance(parte, dict) or not isinstance(parte.get("text"), str):
            continue
        try:
            cuerpo = json.loads(parte["text"])
        except json.JSONDecodeError:
            continue
        if isinstance(cuerpo, dict):
            if isinstance(cuerpo.get("block_id"), str):
                encontrados.append(cuerpo["block_id"])
            for coincidencia in cuerpo.get("matches") or []:
                if isinstance(coincidencia, dict) and isinstance(coincidencia.get("block_id"), str):
                    encontrados.append(coincidencia["block_id"])
    return encontrados


def _sin_repetir(bloques: list[str]) -> list[str]:
    """Únicos, conservando el orden en que el agente los leyó.

    El orden se conserva porque es evidencia: el primer bloque recuperado no es equivalente al
    séptimo cuando se está juzgando si la respuesta se apoyó en lo que trajo.
    """
    vistos: dict[str, None] = {}
    for bloque in bloques:
        vistos.setdefault(bloque, None)
    return list(vistos)
