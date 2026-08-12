"""El entregable: la corrida escrita a disco.

Sin esto una corrida no es una auditoría, es un print. El Profiler mide y devuelve todo lo que hace falta
para defender el número —la respuesta textual del asistente, un `PrincipleGrade` por principio con el
grader, el método y el modelo que lo produjeron, y la evidencia— y nada de eso sobrevive al proceso si
no se escribe.

Importa especialmente acá porque **el grader no es determinista**: Groq no expone logprobs con
`llama-3.3-70b`, así que `LogprobGrader` cae a su camino de respaldo y decide por votación entre cinco
muestras a temperatura 1.0. Volver a correr da otro número. Sin el dataset en disco, el número que se
reporta no lo puede verificar nadie, ni siquiera quien lo corrió.

Tres archivos por corrida, y cada uno tiene su razón de estar separado:

* **`-dataset.json`** — el Roast Dataset. Un registro por consulta, en las formas que el resto del SDK
  ya consume: las métricas que leen la respuesta del asistente sola toman esto sin cambios. Es el
  entregable.
* **`-profile.json`** — el perfil `theta = (omega, H)`. Es el único artefacto que cruza del Profiler al
  Exploiter, y tenerlo aparte permite re-correr el Exploiter sin volver a pagar el Profiler.
* **`-exploiter.json`** — el `FailureReport`, que ya trae `components` con qué implementación de cada
  pieza sustituible corrió y con qué `kappa` y `delta`. Eso es la mitad de la auditoría: dos de las tres
  piezas son construcción de gaussia y no del paper, así que un resultado flojo tiene que ser atribuible
  a la parte que se puede cambiar.

La conversión la hace `to_dataset` del SDK, que valida que cada probe tenga su outcome y ya resuelve las
decisiones que un dataset de trampas tiene que tomar: `ground_truth_assistant=""` porque una trampa no
tiene respuesta correcta e inventarle una dejaría que una métrica puntúe contra ella, y
`evidence_available` viajando en el registro porque "no había evidencia" y "se buscó y no se encontró"
son hallazgos distintos.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gaussia.generators.roastme.dataset import to_dataset
from gaussia.schemas.common import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.schemas.roastme import FailureReport, Probe, ProfilerResult

OUT = Path(__file__).resolve().parent / "out"

ASSISTANT_ID = "assistant-bpd"

# `language` llega a la metadata de la sesión. El default de `to_dataset` es "english", que etiquetaría
# como inglés un corpus escrito en español y probes escritos en español.
LANGUAGE = "spanish"

CONTEXT = (
    "Roast Me sobre el asistente RAG de Banco Popular Dominicano, contra el corpus de páginas de "
    "producto del sitio institucional. Cada consulta es una trampa deliberada o un control; no hay "
    "respuesta correcta contra la que puntuar."
)

_DATASET = "{session_id}-dataset.json"
_PROFILE = "{session_id}-profile.json"
_EXPLOITER = "{session_id}-exploiter.json"


def save_profile_run(
    probes: Sequence[Probe],
    result: ProfilerResult,
    *,
    session_id: str,
    out_dir: Path = OUT,
) -> tuple[Path, Path]:
    """Escribe el Roast Dataset y el perfil. Devuelve las dos rutas, para poder imprimirlas."""
    dataset = to_dataset(
        probes,
        result.outcomes,
        session_id=session_id,
        assistant_id=ASSISTANT_ID,
        context=CONTEXT,
        language=LANGUAGE,
    )
    return (
        _write(out_dir, _DATASET.format(session_id=session_id), _dataset_json(dataset)),
        _write(out_dir, _PROFILE.format(session_id=session_id), result.profile.model_dump_json(indent=2)),
    )


def _dataset_json(dataset: Dataset) -> str:
    """El dataset serializado **con** el registro de Roast Me adentro.

    `serialize_as_any=True` no es decoración. `Dataset.conversation` está tipado `list[Batch]` y lo que
    lleva adentro son `RoastBatch`, que es una subclase. Pydantic v2 serializa según el tipo *declarado*,
    así que por defecto descarta los campos que agrega la subclase: el JSON sale con la consulta y la
    respuesta y **sin `roast`** — sin la violación, sin los principios imputados y sin el rationale del
    juez. Verificado en pydantic 2.13.4: `'roast' in ...` da `False` sin este argumento y `True` con él.

    O sea que sin esto el entregable se escribe, pesa, se abre, y no contiene la medición. Es la peor
    forma de fallar, porque no hay error.
    """
    return dataset.model_dump_json(indent=2, serialize_as_any=True)


def save_failure_report(report: FailureReport, *, session_id: str, out_dir: Path = OUT) -> Path:
    """Escribe el reporte del Exploiter, con los componentes y los umbrales en vigor."""
    return _write(out_dir, _EXPLOITER.format(session_id=session_id), report.model_dump_json(indent=2))


def load_replay_answers(session_id: str, out_dir: Path = OUT) -> dict[str, str] | None:
    """Las respuestas reales de una corrida anterior, o `None` si no hay dataset para esa sesión.

    Es lo que hace que el ensayo offline use evidencia en lugar de un fixture inventado: el replay
    contesta lo que el asistente contestó de verdad. Los turnos se releen como `Batch` y no como
    `RoastBatch` —`Dataset.conversation` está tipado con el primero— y eso alcanza, porque un replay
    solo necesita la consulta y la respuesta.

    Un intercambio que falló en el transporte quedó con la respuesta vacía. Se omite en lugar de
    replayarse como respuesta legítima: `ReplayAssistant` reporta `failed=True` para una consulta que no
    tiene entrada, que es exactamente lo que ese intercambio fue.
    """
    archivo = out_dir / _DATASET.format(session_id=session_id)
    if not archivo.exists():
        return None
    dataset = Dataset.model_validate_json(archivo.read_text(encoding="utf-8"))
    return {turn.query: turn.assistant for turn in dataset.conversation if turn.assistant}


def _write(out_dir: Path, name: str, payload: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / name
    destino.write_text(payload, encoding="utf-8")
    return destino
