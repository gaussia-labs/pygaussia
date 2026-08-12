"""Las premisas falsas: un producto que no existe, y un valor que el producto no tiene.

Los cuatro transforms que shipea gaussia asumen la forma de entidad del corpus del paper. El que
inventa una entidad le agrega el sufijo `-2`, que sobre `POLICY-1` se lee como un artículo vecino y
sobre `Cuenta Digital Libre` se lee como un typo. Y un probe cuya premisa se lee como un typo **no
discrimina**: si el asistente la rechaza puede ser porque detectó que no existe o porque le pareció
texto roto, y si la describe puede ser porque inventa o porque asumió el typo.

`flip_value`, que sería el candidato para atacar un valor, desplaza cada corrida de dígitos, así que
sobre un número con separador de miles produce basura: `RD$500,000` se convierte en `RD$501,1`.

Así que las dos premisas se construyen acá. **Aportar un transform propio es parte del contrato de la
librería**, no un atajo: `EnumerationProbeEngine` toma `transforms=[...]` y el registro resuelve los
cuatro shipped más los que se le pasen. Lo único cerrado es que un key propio no puede colisionar con
uno de los cuatro.

La garantía que comparten los dos es la misma, y es la que los vuelve defendibles: **verificar contra
la enumeración completa que el resultado no exista**. El enumerador conoce los 136 productos y los 121
enunciados de valor que el corpus documenta, así que "no existe" es una afirmación y no una suposición.

**Cuando la regla no puede construir un falso devuelve la entidad sin cambios**, y eso importa: el
motor entonces la etiqueta como documentada y la strategy se vuelve un segundo control en silencio.
Por eso `falso_de` está separado de `apply` —el llamador tiene que poder preguntar *antes* de sondear—
y por eso `probes.py` elige las entidades a sondear de las que sí tienen falso.
"""

from __future__ import annotations

import collections
import re
from abc import abstractmethod

from gaussia.core.transform import Transform

KEY_PRODUCTO = "producto_falso_plausible"
KEY_VALOR = "valor_falso_plausible"

_MONTO = re.compile(r"RD\$[\d.,]+")
_SEPARADOR_CONCEPTO = " de "


class FalsoVerificado(Transform):
    """Un transform que verifica su propia premisa contra la enumeración completa.

    `apply` está escrito una sola vez porque los dos transforms lo necesitan idéntico, y porque la
    equivalencia entre "no pude construir un falso" y "devolvé la entidad" es justo la que no conviene
    reimplementar dos veces: es la que convierte una strategy en un control sin avisar.
    """

    def apply(self, entity: str) -> str:
        return self.falso_de(entity) or entity

    @abstractmethod
    def falso_de(self, entity: str) -> str | None:
        """La premisa falsa para esta entidad, o `None` si la regla no puede construir una."""


class ProductoFalsoPlausible(FalsoVerificado):
    """Un nombre que suena a producto del banco y que el banco no ofrece.

    La regla: cambiar la última palabra del producto por la última palabra de otro producto de la misma
    familia —misma primera palabra y misma cantidad de palabras—, y verificar que el resultado no exista.

    Dos límites, medidos y no estimados:

    * la regla necesita al menos dos productos parecidos, así que cubre **51 de los 136** en 8 familias.
      Un producto que es el único de su familia no consigue falso;
    * la sustitución es ciega a la gramática, así que sobre nombres que terminan en una frase con
      preposición produce cosas torpes: `Seguros Tu Servicios`, `Préstamos Interinos a la Joven`. De doce
      probes salen unos tres así. No invalidan la medición —un asistente bien anclado tiene que decir "no
      existe" también ante un nombre torpe— pero para medir el catálogo completo hace falta un LLM que
      escriba el nombre, no una regla.

    Args:
        reales: La lista **completa** de productos que el corpus documenta, del enumerador. Es contra
            esto que se verifica que el falso no exista, así que una lista parcial convierte cada
            premisa en una adivinanza.
    """

    def __init__(self, reales: frozenset[str]) -> None:
        self._reales = reales
        self._finales: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
        for producto in reales:
            partes = producto.split()
            self._finales[(partes[0], len(partes))].add(partes[-1])

    @property
    def key(self) -> str:
        return KEY_PRODUCTO

    def falso_de(self, entity: str) -> str | None:
        partes = entity.split()
        if len(partes) < 2:
            return None
        for final in sorted(self._finales[(partes[0], len(partes))]):
            if final == partes[-1]:
                continue
            candidato = " ".join(partes[:-1] + [final])
            if candidato not in self._reales:
                return candidato
        return None


class ValorFalsoPlausible(FalsoVerificado):
    """El monto de un producto, reemplazado por el que el corpus trae para otro producto.

    La regla: sustituir el monto por **el más cercano pero distinto que el corpus declara para el mismo
    concepto en otro producto**. Es una cifra real del banco atribuida al producto equivocado, que es la
    confusión que un cliente comete de verdad —`Cargo anual de Clásica es RD$1,200.00` cuando son
    RD$900.00 y RD$1,200.00 es el de otra tarjeta— y es inequívocamente falsa.

    El más cercano y no el primero: un near miss, por la misma razón que el otro transform no produce
    gibberish. Iterar de menor a mayor mandaba casi todos los falsos a `RD$0`, que es plausible pero
    monótono: 30 montos falsos distintos contra los 9 que salían así.

    ⚠️ **Los montos se comparan normalizados.** El corpus escribe `RD$0` y `RD$0.00` como cadenas
    distintas para el mismo valor. Sin normalizar, la regla produciría un "falso" que en realidad es
    verdadero, el juez lo cargaría como violación, y el número quedaría envenenado en la dirección
    contraria — un asistente que contestó bien contado como que aceptó una premisa falsa.

    El límite, medido: **69 de los 121** enunciados consiguen falso. Un concepto que el corpus declara
    con un solo monto —o con el mismo en todos los productos— no lo consigue, y son 33 de los 52
    conceptos.

    Args:
        reales: Los enunciados de valor **completos** del enumerador, para verificar que el resultado
            no exista.
    """

    def __init__(self, reales: frozenset[str]) -> None:
        self._reales = reales
        # Concepto -> monto normalizado -> el texto tal como el corpus lo escribe. El dict por valor
        # normalizado es lo que colapsa `RD$0` con `RD$0.00` en una sola opción.
        self._montos: dict[str, dict[float, str]] = collections.defaultdict(dict)
        for enunciado in reales:
            monto = _MONTO.search(enunciado)
            if monto is not None:
                self._montos[_concepto(enunciado)].setdefault(_normalizar(monto.group()), monto.group())

    @property
    def key(self) -> str:
        return KEY_VALOR

    def falso_de(self, entity: str) -> str | None:
        monto = _MONTO.search(entity)
        if monto is None:
            return None
        propio = _normalizar(monto.group())
        for texto in _por_cercania(self._montos[_concepto(entity)], propio):
            candidato = entity[: monto.start()] + texto + entity[monto.end() :]
            if candidato not in self._reales:
                return candidato
        return None


def _concepto(enunciado: str) -> str:
    return enunciado.split(_SEPARADOR_CONCEPTO)[0]


def _normalizar(monto: str) -> float:
    """`RD$1,000` y `RD$1,000.00` al mismo número, que es lo que decide si un falso es realmente falso."""
    return float(monto.removeprefix("RD$").replace(",", ""))


def _por_cercania(montos: dict[float, str], propio: float) -> list[str]:
    """Los montos distintos del propio, del más cercano al más lejano.

    El orden es total —distancia, y después el valor— así que dos corridas sobre el mismo corpus
    eligen el mismo falso.
    """
    candidatos = sorted((abs(valor - propio), valor, texto) for valor, texto in montos.items() if valor != propio)
    return [texto for _, _, texto in candidatos]
