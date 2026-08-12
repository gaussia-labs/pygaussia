"""Los probes, generados por la Probe Library del SDK.

Nada acá está escrito a mano: el enumerador da las entidades, el catálogo da las strategies, y
`ProbeLibrary` los combina. Lo propio son los dos transforms que construyen las premisas falsas y la
elección de cuántas entidades sondear.

**La elección sale de las entidades que tienen falso disponible**, no de las 136 ni de las 121. Si una
entidad sin falso entrara al set, su strategy de ataque recibiría la entidad sin cambios, el motor la
etiquetaría como documentada, y ese probe sería un segundo control sin que nada lo diga. Elegir de las
que sí tienen falso cierra esa puerta.

Y se eligen por round-robin sobre grupos, así el set cubre variedad en lugar de amontonarse: los
productos por familia —cuentas, tarjetas Mastercard, Visa, préstamos, seguros, servicios de pago— y los
valores por concepto —balance mínimo, cargo anual, cargo por bajo balance— en vez de doce del mismo.
Es reproducible y no elige a dedo: recorre los grupos en orden alfabético tomando uno de cada uno por
vuelta.

Un solo engine para los dos kinds. `EnumerationProbeEngine._entities` reenvía el `kind` al enumerador,
así que cada strategy recibe la frontera de *su* tipo de entidad y no la frontera entera.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from gaussia.generators.roastme.probes.enumeration import EnumerationProbeEngine
from gaussia.generators.roastme.probes.library import ProbeLibrary

from catalogue import CATALOGUE
from documents import load_documents
from enumerator import PRODUCTO, VALOR, BpdEnumerator
from fakes import ProductoFalsoPlausible, ValorFalsoPlausible

if TYPE_CHECKING:
    from collections.abc import Callable

    from gaussia.core.transform import Transform
    from gaussia.schemas.roastme import Probe

    from fakes import FalsoVerificado

PRODUCTOS_A_SONDEAR = 12
VALORES_A_SONDEAR = 12


def build_engine(transforms: list[Transform]) -> EnumerationProbeEngine:
    """El engine, para que la generación y `validate_catalogue` reciban el mismo.

    Los dos kinds van al mismo engine porque el enumerador despacha por kind. `entity_kinds` es lo que
    `validate_catalogue` chequea contra el catálogo, así que un kind que falte acá hace fallar la
    validación en lugar de producir un set de probes vacío en silencio.
    """
    return EnumerationProbeEngine(BpdEnumerator(), entity_kinds={PRODUCTO, VALOR}, transforms=transforms)


def build_probes(
    productos_a_sondear: int = PRODUCTOS_A_SONDEAR,
    valores_a_sondear: int = VALORES_A_SONDEAR,
) -> tuple[list[Probe], list[Transform]]:
    """Los probes a enviar, y los transforms con los que se generaron.

    Los transforms vuelven junto con los probes porque `validate_catalogue` tiene que recibir los
    mismos: un catálogo validado contra un set de transforms y generado contra otro es justo el caso en
    que la validación deja de significar algo.
    """
    documentos = load_documents()
    enumerables = [documento for documento in documentos if documento.structured]
    enumerador = BpdEnumerator()

    productos = enumerador.enumerate_entities(PRODUCTO, enumerables)
    valores = enumerador.enumerate_entities(VALOR, enumerables)

    falso_producto = ProductoFalsoPlausible(productos)
    falso_valor = ValorFalsoPlausible(valores)

    elegidos = _elegidos(productos, falso_producto, _familia, productos_a_sondear)
    elegidos |= _elegidos(valores, falso_valor, _producto_del_enunciado, valores_a_sondear)

    transforms: list[Transform] = [falso_producto, falso_valor]
    generados = ProbeLibrary([build_engine(transforms)]).generate(documentos, CATALOGUE)
    return [probe for probe in generados if probe.meta.get("real_value") in elegidos], transforms


def _elegidos(
    entidades: frozenset[str],
    transform: FalsoVerificado,
    grupo_de: Callable[[str], str],
    cuantos: int,
) -> set[str]:
    con_falso = [entidad for entidad in sorted(entidades) if transform.falso_de(entidad) is not None]
    por_grupo: dict[str, list[str]] = collections.defaultdict(list)
    for entidad in con_falso:
        por_grupo[grupo_de(entidad)].append(entidad)
    return set(_round_robin(dict(por_grupo), cuantos))


def _familia(producto: str) -> str:
    return producto.split()[0]


def _producto_del_enunciado(enunciado: str) -> str:
    """La última palabra del producto que el enunciado nombra, como clave de agrupación.

    Agrupar por concepto parecía lo natural y concentra en lugar de repartir: la tarjeta Drive declara
    cinco cargos con cinco conceptos distintos, así que caían en cinco grupos y se llevaban 5 de los 12
    valores sondeados. La última palabra del producto los junta en uno.

    Se toma la última palabra y no el nombre entero porque el enunciado es
    `"{concepto} de {producto} es {valor}"` y el concepto también puede llevar un `" de "`
    (`"Balance de apertura de Cuenta Prime es ..."`), así que partir por el separador es ambiguo y
    quedarse con el final no lo es.
    """
    return enunciado.rsplit(" es ", 1)[0].split()[-1]


def _round_robin(por_grupo: dict[str, list[str]], cuantos: int) -> list[str]:
    # Sin grupos no hay nada que recorrer, y `max` sobre una secuencia vacía tira un ValueError que no
    # dice nada. Pasa si el corpus cambia y ninguna entidad consigue falso.
    if not por_grupo:
        return []
    elegidos: list[str] = []
    for vuelta in range(max(len(miembros) for miembros in por_grupo.values())):
        for grupo in sorted(por_grupo):
            if vuelta < len(por_grupo[grupo]):
                elegidos.append(por_grupo[grupo][vuelta])
                if len(elegidos) == cuantos:
                    return elegidos
    return elegidos
