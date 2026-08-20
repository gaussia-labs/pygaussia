"""Ata `contract.json` a un grader y avisa si `tau` es alcanzable.

Por qué existe este archivo y no alcanza el JSON solo: `Principle.grader` es una instancia de
una ABC, no data — el modelo lleva `arbitrary_types_allowed` justamente por eso. Así que el
contrato se parte en dos: el JSON tiene lo que es del negocio (qué cuenta como falla, y con
cuánta severidad) y este módulo ata el juez. La *configuración* del juez sí es data y vive en
`config.json`.

El techo del Exploiter es lo que este módulo existe para hacer visible. El juez nunca ve el
corpus: sólo recibe la consulta, la respuesta y el `meta` del probe. Una rúbrica de "¿inventó
algo?" necesita saber qué premisa falsa se le plantó, y eso llega por `false_value` del meta.
El Exploiter gradea queries que él mismo generó, sin probe detrás y por lo tanto sin meta, así
que esas rúbricas terminan en "si el contexto NO trae `false_value`, CUMPLE" y devuelven 0.0
**siempre** en esa mitad del método.

O sea que el Exploiter sólo puede imputar los principios decidibles mirando la respuesta sola,
y la suma de sus pesos es el techo. Un `tau` por encima del techo hace que ninguna categoría
pueda pasar jamás: la corrida termina, el reporte se puebla, y el cero se lee como "el
asistente está bien" cuando era imposible que diera otra cosa.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gaussia.schemas.roastme import BehavioralContract, ExploiterConfig, GraderConfig, Principle

if TYPE_CHECKING:
    from gaussia.core.grader import Grader

AQUI = Path(__file__).resolve().parent
CONTRATO = AQUI / "contract.json"
CONFIG = AQUI / "config.json"


def _principios() -> list[dict[str, object]]:
    return [
        entrada
        for entrada in json.loads(CONTRATO.read_text(encoding="utf-8"))["principles"]
        if not str(entrada["id"]).startswith("_")
    ]


DEPENDEN_DEL_CONTEXTO = frozenset(
    str(entrada["id"]) for entrada in _principios() if entrada.get("context_dependent")
)
"""Los principios cuya rúbrica se decide leyendo `false_value` del meta del probe.

Se declara en el JSON y no se deriva del texto de la rúbrica: quien la escribe es el único que
lo sabe, y adivinarlo leyendo el string sería exactamente la clase de inferencia que este
módulo existe para no hacer.
"""


def build_contract(grader: Grader) -> BehavioralContract:
    """El contrato, con un grader atado a cada principio.

    Una sola instancia sirve para todos: lo que la especificación fija es que cada principio
    tenga exactamente un grader, así que comparar jueces es correr la evaluación de nuevo y no
    promediar dos adentro de un principio.
    """
    return BehavioralContract(
        principles=[
            Principle(
                id=str(entrada["id"]),
                weight=float(entrada["weight"]),  # type: ignore[arg-type]
                rubric=str(entrada["rubric"]),
                grader=grader,
            )
            for entrada in _principios()
        ]
    )


def build_exploiter_config() -> ExploiterConfig:
    return ExploiterConfig.model_validate(_seccion("exploiter"))


def build_grader_config() -> GraderConfig:
    return GraderConfig.model_validate(_seccion("grader"))


def _seccion(nombre: str) -> dict[str, object]:
    seccion = json.loads(CONFIG.read_text(encoding="utf-8"))[nombre]
    return {clave: valor for clave, valor in seccion.items() if not clave.startswith("_")}


def techo_del_exploiter(contract: BehavioralContract) -> float:
    """El `v` máximo que una respuesta puede sacar en el Exploiter. **No es 1.0.**"""
    return sum(
        principio.weight
        for principio in contract.principles
        if principio.id not in DEPENDEN_DEL_CONTEXTO
    )


def diagnosticar_tau(contract: BehavioralContract, tau: float) -> str:
    """Una línea que dice si el `tau` configurado es alcanzable, para imprimir junto al reporte."""
    techo = techo_del_exploiter(contract)
    ciegos = sorted(
        principio.id for principio in contract.principles if principio.id in DEPENDEN_DEL_CONTEXTO
    )
    veredicto = "IMPOSIBLE: ninguna categoría puede alcanzarlo" if tau > techo else "alcanzable"
    return (
        f"tau={tau:.2f} · techo del Exploiter={techo:.2f} · {veredicto}\n"
        f"ciegos sin contexto (siempre 0.0 en el Exploiter): {', '.join(ciegos) or 'ninguno'}"
    )


def contrato_estructural() -> BehavioralContract:
    """El contrato con un juez de mentira, para lo que sólo necesita su FORMA.

    `validate_catalogue` chequea que cada plugin resuelva contra un principio del contrato, y eso
    se decide con los ids y nada más. Sin esta función, validar un catálogo obligaría a construir
    el juez —y con él el modelo y las credenciales— antes de poder rechazar un catálogo roto, que
    es justo el orden que FR-025 existe para invertir: todo lo rechazable se rechaza antes de
    gastar una llamada.

    El marcador tiene que heredar de `Grader` de verdad: `Principle` valida el tipo, y esa
    validación es lo que impide que un principio llegue al contrato sin juez y aporte un cero
    silencioso a `v`. Por eso no alcanza con `None` y por eso `grade` levanta en vez de devolver
    0.0 — un contrato estructural que se usara para graduar tiene que romperse, no medir.
    """
    from gaussia.core.grader import Grader as _Grader

    class _GraderNulo(_Grader):
        def grade(self, query, response, principle, meta=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError("contrato estructural: este grader no gradea")

    return build_contract(_GraderNulo())


def build_grader():
    """El juez de verdad: el grader por logprobs, atado al modelo del usuario."""
    from gaussia.graders.logprob import LogprobGrader

    from modelo import build_modelo

    return LogprobGrader(build_modelo(temperature=0.0), build_grader_config())


def main() -> None:
    """Valida los dos JSON y muestra el diagnóstico, sin llamar a ningún modelo."""
    contract = contrato_estructural()
    exploiter = build_exploiter_config()
    build_grader_config()

    print(f"contrato válido · {len(contract.principles)} principios")
    for principio in contract.principles:
        marca = "ciego en Exploiter" if principio.id in DEPENDEN_DEL_CONTEXTO else "ambas etapas"
        print(f"  {principio.weight:>5.2f}  {principio.id:26} {marca}")
    print(f"  {sum(p.weight for p in contract.principles):>5.2f}  (suma)")
    print()
    print(diagnosticar_tau(contract, exploiter.tau))


if __name__ == "__main__":
    main()
