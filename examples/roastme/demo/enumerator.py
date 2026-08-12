"""El enumerador del corpus: la lista cerrada de lo que el banco documenta.

Por qué hace falta uno propio: los tres engines por defecto extraen entidades con una sola regex
(`[A-Za-z0-9]+([-_][A-Za-z0-9]+)+`), pensada para corpus con identificadores tipo `POLICY-1`. Sobre este
corpus —prosa en español con nombres de producto de varias palabras— devuelve 208 falsos positivos:
teléfonos, nombres de PDF, anclas de footer. Y no falla: genera probes sobre esa basura en silencio.

`EnumerationProbeEngine` es el único engine que escapa a la regex, y el único que puede afirmar ausencia
con fundamento. Su contrato es la **completitud**: si devuelve una muestra, cada etiqueta de ausencia
pasa a ser una adivinanza. Así que acá se devuelve la lista entera; recortar cuántos probes mandar es
una decisión posterior, sobre la lista ya generada.

Dos tipos de entidad, y el engine respeta la distinción. `EnumerationProbeEngine._entities` reenvía el
`kind` a este objeto, a diferencia de los engines de regex, que lo ignoran y le dan a cada strategy la
frontera entera. Así que un solo enumerador que despacha por `kind` alcanza, y devolver vacío para un
kind desconocido es deliberado: el engine no produce probes para esa strategy, que es preferible a
inventar entidades de un tipo que nadie sabe leer.

* **`producto`** — los títulos `##`. Los `#` son categorías (TARJETAS DE CRÉDITO) y los `###` son las
  secciones de cada producto (Beneficios, Tarifas, Requisitos), así que ninguno de los dos entra.
  Los títulos traen el alias adentro, entre paréntesis: `Clásica Popular Bank (Clásica Internacional)`.
  El paréntesis se corta, porque viaja a la premisa del probe y produce preguntas que nadie haría
  (`...(Clásica Internacional)-2`). Verificado antes de aplicarlo: cortarlo colapsa un solo par
  —`Inversiones Mercado de Valores` con y sin su alias— que es el mismo producto anotado dos veces, así
  que el colapso deduplica en lugar de perder algo. Quedan 136.

* **`valor`** — un enunciado por fila de las tablas `| Concepto | Monto |` que cada producto trae bajo
  `### Costos` o `### Requisitos y costos`. Son 121 enunciados sobre 47 productos.
  La entidad es el enunciado entero y no el monto suelto, porque el monto suelto no dice de qué producto
  habla: `"hint: RD$1,001"` produce un probe que el asistente contesta preguntando "¿de cuál?", y eso no
  mide nada. `"Balance mínimo de Cuenta Digital Libre es RD$0.00"` sí.

Solo entran las filas cuya celda de valor arranca con `RD$`. Eso deja afuera los `N/A`, los encabezados,
y las tablas de tasas escalonadas (`| RD$0.01 – RD$100,000 | 0.01% |`), cuya celda de valor es un
porcentaje: un rango no es una clave única y su tasa no se atribuye a un producto sino a un tramo.
"""

import re
from collections.abc import Iterator

from gaussia.core.entity_enumerator import EntityEnumerator
from gaussia.schemas.roastme import Document

PRODUCTO = "producto"
VALOR = "valor"

# Un producto por título `##`.
_TITULO = re.compile(r"^## +", re.M)

# Títulos que no son productos: aparecen en los `*-detalle.md` como encabezados de página.
_NO_PRODUCTOS = frozenset({"Documentos enlazados", "Índice del documento"})

# El alias que el corpus anota entre paréntesis al final del título.
_ALIAS = re.compile(r"\s*\([^)]*\)\s*$")

# `| Balance mínimo | RD$0.00 |`, y también `| Tarjeta débito física | RD$65.00 mensuales |`: el sufijo
# se conserva porque es parte del dato que el enunciado afirma.
_FILA_MONTO = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(RD\$[\d.,]+[^|]*?)\s*\|\s*$", re.M)

_ENUNCIADO = "{concepto} de {producto} es {valor}"


class BpdEnumerator(EntityEnumerator):
    """Las entidades que el corpus documenta: los productos, y los valores que cada uno declara."""

    def enumerate_entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        """La lista completa para `kind`, o vacía si no es un kind que este enumerador conozca."""
        lectores = {PRODUCTO: self._productos, VALOR: self._valores}
        lector = lectores.get(kind)
        return frozenset(lector(documents)) if lector is not None else frozenset()

    def _productos(self, documents: list[Document]) -> Iterator[str]:
        for producto, _ in self._bloques(documents):
            yield producto

    def _valores(self, documents: list[Document]) -> Iterator[str]:
        for producto, cuerpo in self._bloques(documents):
            for concepto, valor in _FILA_MONTO.findall(cuerpo):
                yield _ENUNCIADO.format(concepto=concepto.strip("* ").strip(), producto=producto, valor=valor)

    @staticmethod
    def _bloques(documents: list[Document]) -> Iterator[tuple[str, str]]:
        """Cada producto con el texto que lo describe, que es lo que las dos lecturas comparten."""
        for document in documents:
            for bloque in _TITULO.split(document.content)[1:]:
                titulo, _, cuerpo = bloque.partition("\n")
                producto = _ALIAS.sub("", titulo.strip()).strip()
                if producto and producto not in _NO_PRODUCTOS:
                    yield producto, cuerpo
