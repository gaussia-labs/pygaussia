"""Los probes, generados por el motor grounded de gaussia.

Nada acá está escrito a mano. El catálogo da las strategies, el corpus da los trozos, y
`GroundedProbeEngine` combina las dos cosas llamando al torcedor una vez por trozo y por strategy.
Lo único de SetPlus es el corpus, el catálogo y qué modelo se le pasa al torcedor.

**Qué reemplazó este archivo.** Antes hacían falta tres: `motores_llm.py` (el torcedor a mano
contra el router de HF), `probes_llm.py` (que armaba los `Probe` uno por uno porque `ProbeLibrary`
combina enumeración x strategy y no sabía de trozos) y el `probes.py` viejo (el camino
determinista, que necesitaba enumerador y transform propio). El motor grounded hace los tres
trabajos y ninguno de los dos artefactos a mano es ya un requisito.

**Lo que este camino NO mide, y hay que decirlo.** `absence_reliable` es False en todo lo que no
sea control: un trozo es una muestra del corpus y nunca el corpus entero, así que este motor no
puede sostener que algo no exista. Sirve para "el asistente aceptó un dato falso sobre un producto
real"; NO sirve para "el asistente inventó un producto". Eso último sigue siendo del motor de
enumeración, que sigue necesitando el enumerador a mano. Ver `auditar.py`.

**El costo, antes de correr.** Una llamada al modelo por trozo y por strategy. Con 7 strategies y
el corpus entero eso es varios cientos de llamadas antes de tocar al asistente una sola vez, así
que `PASAJES_POR_STRATEGY` existe y la corrida imprime cuántos trozos usó de cuántos hay.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from gaussia.generators.roastme import (
    GroundedProbeEngine,
    PromptedFactTwister,
    validate_catalogue,
)
from gaussia.schemas.roastme import Catalogue, Document, Probe

from contract import contrato_estructural
from modelo import build_modelo

AQUI = Path(__file__).resolve().parent
CORPUS = AQUI / "data" / "documents.json"
CATALOGO = AQUI / "catalogue.json"
A_MANO = AQUI / "probes_instrucciones.json"
SALIDA = AQUI / "out" / "bpd-probes.json"

PASAJES_POR_STRATEGY = 20
"""Cuántos trozos del corpus se le pasan a cada strategy. `None` sería el corpus entero.

20 x 6 strategies = 120 llamadas al modelo, sobre los 56 trozos que el corpus da. Sigue siendo un
recorte y la corrida lo imprime: un recorte silencioso se lee como cobertura completa, que es el
modo de falla que este subsistema tiene que evitar en todos lados.

Subió de 12 a 20 para que cada entrada del mapa de debilidades descanse en más evidencia. Con n=12
la debilidad más alta de la corrida anterior midió 0.100 con un error estándar de 0.095 — casi tan
grande como la medición — y descansaba sobre UNA violación observada. El error estándar cae con la
raíz de n, así que 20 no lo arregla, lo mejora; lo que sí cambia es que una violación de más o de
menos deje de mover el rate un 10%.
"""

TEMPERATURA = 0.7
"""La temperatura del TORCEDOR, y sólo la de él. El juez sigue en 0.0.

En cero el modelo elige siempre su continuación más probable, así que doce trozos parecidos
producen doce preguntas con el mismo cierre. Es la palanca más fuerte sobre la variedad y también
la más cara: **se pierde que dos corridas sobre el mismo corpus generen los mismos probes.** Se
paga a propósito, porque un set de sondas que repite doce veces la misma fórmula mide una fórmula
y no un comportamiento — y `out/probes.json` queda igual como el registro de qué se mandó, que es
lo que hace auditable una corrida aunque no sea regenerable idéntica.

El juez NO se toca. Ahí la variación no es riqueza, es ruido en la medición: dos corridas tienen
que poder atribuir una diferencia al asistente y no al veredicto.
"""

IDIOMA = "es"
"""El prompt del torcedor es inglés y un modelo contesta en el idioma en que se lo interpela. Sin
esto, un corpus en español devuelve preguntas en inglés sobre nombres de producto en español."""


def cargar_documentos() -> list[Document]:
    """El corpus, que **no viene con este demo** y hay que traer.

    RoastMe no trae corpus por la misma razón que no trae contrato ni catálogo: contra qué se mide
    el conocimiento de un asistente es del negocio, no de la librería. La corrida que está en `out/`
    se hizo contra las páginas públicas de producto del Banco Popular Dominicano; para reproducirla
    hay que volver a armar ese archivo.

    Sin corpus se puede leer todo lo que hay en `out/` —las 138 consultas con su premisa falsa, los
    intercambios graduados y las categorías— pero no generar consultas nuevas.
    """
    if not CORPUS.exists():
        message = (
            f"no está {CORPUS.relative_to(AQUI)}: este demo no trae corpus.\n"
            "Armá el tuyo como {'documents': [{'id':…, 'content':…, 'structured': true}, …]} y "
            "volvé a correr, o leé la corrida ya hecha en out/."
        )
        raise FileNotFoundError(message)
    crudos = json.loads(CORPUS.read_text(encoding="utf-8"))["documents"]
    return [Document.model_validate(d) for d in crudos]


def cargar_catalogo() -> Catalogue:
    crudo = json.loads(CATALOGO.read_text(encoding="utf-8"))
    return Catalogue.model_validate({k: v for k, v in crudo.items() if not k.startswith("_")})


def build_engine() -> tuple[GroundedProbeEngine, PromptedFactTwister]:
    twister = PromptedFactTwister(build_modelo(temperature=TEMPERATURA), language=IDIOMA)
    engine = GroundedProbeEngine(
        twister=twister,
        entity_kinds=("producto",),
        passages_per_strategy=PASAJES_POR_STRATEGY,
    )
    return engine, twister


PLUGIN_A_MANO = "fuga_de_instrucciones"
"""El plugin que los probes a mano cargan. Tiene que existir en el catálogo aunque ninguna strategy
lo nombre: un plugin sin strategy es válido, y es exactamente lo que este archivo aprovecha."""


def probes_a_mano() -> list[Probe]:
    """Los probes que ningún motor puede generar, leídos de `probes_instrucciones.json`.

    El motor grounded escribe sobre un hecho del corpus, y el corpus del banco no dice nada de las
    instrucciones del agente — así que ese principio, que pesa 0.25 del contrato, no tenía un solo
    probe atacándolo. Estos lo cubren.

    Van por fuera del catálogo a propósito. Una `StrategySpec` describe cómo un motor deriva una
    consulta de una entidad; acá no hay entidad ni derivación, hay quince consultas fijas que valen
    igual para cualquier agente con un prompt de sistema. Meterlas al catálogo obligaría a inventar
    un `entity_kind` y un `transform` que nadie usa.

    `hook=None` porque no se apoyan en el corpus, y fabricar uno de relleno metería una entidad
    inventada en los hooks retenidos del perfil, que es sobre lo que el Exploiter ancla categorías.
    `meta={}` porque la rúbrica decide mirando sólo la respuesta — y de paso eso las exime de
    `no_inventar`, cuya rúbrica dice que sin `false_value` en el contexto CUMPLE.
    """
    if not A_MANO.exists():
        return []
    crudo = json.loads(A_MANO.read_text(encoding="utf-8"))
    salida: list[Probe] = []
    for grupo in crudo["probes"]:
        control = bool(grupo.get("control"))
        for i, query in enumerate(grupo["queries"]):
            salida.append(
                Probe(
                    id=f"a-mano-{grupo['strategy']}-{i}",
                    query=query,
                    strategy=grupo["strategy"],
                    attrs=grupo["attrs"],
                    # Un control se marca dejando el plugin en None (FR-026), que es el único
                    # mecanismo por el que se reconoce uno. Los de ataque cargan el plugin, y con
                    # él el principio que ponen a prueba.
                    plugin=None if control else PLUGIN_A_MANO,
                    hook=None,
                    engine="a-mano",
                    meta={},
                )
            )
    return salida


def cargar_probes() -> list[Probe]:
    """Los probes de la última generación, leídos del disco.

    Generar y mandar están separados a propósito. La generación cuesta una llamada al modelo por
    trozo y por strategy y no toca al asistente, así que `out/probes.json` se puede producir y
    **revisar a ojo** —¿las premisas falsas son plausibles?, ¿las preguntas repiten plantilla?—
    antes de gastar la corrida contra el agente y el juez. Releerlo en vez de regenerar también es
    lo que hace que dos corridas del Profiler sobre los mismos probes sean comparables.
    """
    crudo = json.loads(SALIDA.read_text(encoding="utf-8"))
    return [Probe.model_validate(p) for p in crudo["probes"]] + probes_a_mano()


def main() -> None:
    documentos = cargar_documentos()
    catalogo = cargar_catalogo()
    engine, twister = build_engine()

    # Antes de generar, no después: las seis condiciones de FR-025 son decidibles sin llamar al
    # modelo, y una strategy que nombra un patrón que el torcedor no realiza tiene que fallar acá
    # y no a mitad de 84 llamadas pagas.
    validate_catalogue(catalogo, contrato_estructural(), engines=[engine], twisters=[twister])
    print(f"catálogo válido: {len(catalogo.strategies)} strategies contra {len(catalogo.plugins)} plugins")
    print(f"corpus: {len(documentos)} documentos · {PASAJES_POR_STRATEGY} trozos por strategy")
    print(f"torcedor: {twister.model} · temperature={TEMPERATURA} (no reproducible a propósito)")

    probes = engine.generate(documentos, catalogo)

    por_strategy = Counter(p.strategy for p in probes)
    controles = sum(1 for p in probes if p.plugin is None or (p.hook and p.hook.doc == 1))
    print(f"\n{len(probes)} probes · {controles} con premisa real")
    for strategy, n in sorted(por_strategy.items()):
        print(f"  {strategy:28} {n:3}")
    faltantes = [s.id for s in catalogo.strategies if s.id not in por_strategy]
    if faltantes:
        print(f"\n  SIN PROBES: {faltantes} — el torcedor no devolvió nada para esas strategies")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(
        json.dumps(
            {
                "motor": engine.name,
                "modelo": twister.model,
                "pasajes_por_strategy": PASAJES_POR_STRATEGY,
                "corpus": str(CORPUS.relative_to(AQUI)),
                "probes": [p.model_dump(mode="json") for p in probes],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nescrito {SALIDA.relative_to(AQUI)}")


if __name__ == "__main__":
    main()
