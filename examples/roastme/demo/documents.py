"""Carga el corpus de BPD como objetos `Document`.

`structured` decide qué engines pueden ver cada documento. Es el único campo de `Document` con
consecuencia de comportamiento: `EnumerationProbeEngine.can_handle` devuelve `document.structured`,
así que ese engine recibe solo el subconjunto marcado en True.

Acá se marcan los seis `*-detalle.md`, que son los que tienen una estructura enumerable: un título
`##` por producto, con una línea `**URL:**` por cada uno. Verificado: 148 títulos y 148 URLs, 1:1.

`popular-tarifario.md` queda afuera del corpus entero. Es la transcripción de un PDF: sus 51 títulos
`##` se repiten textualmente (`## Cuentas de Ahorro` aparece cuatro veces) y sus valores son columnas
posicionales sin encabezado por fila, así que un producto sacado de ahí no es una clave única y sus
tarifas se mal-atribuyen entre cuatro o cinco columnas. Un dato mal atribuido no under-produce: envenena,
porque se convierte en un `doc=1` falso contra el que después se juzga al asistente.
"""

from pathlib import Path

from gaussia.schemas.roastme import Document

_AQUI = Path(__file__).resolve().parent

# Dos ubicaciones, un solo archivo. El demo publicado es autocontenido y lleva el corpus en `data/` al
# lado; el proyecto vivo lo tiene un nivel arriba, compartido con el resto del trabajo del agente.
# El nombre va en minúscula en el demo a propósito: `Data` funciona en macOS y falla en un CI de Linux,
# que sí distingue mayúsculas.
CORPUS = _AQUI / "data" if (_AQUI / "data").is_dir() else _AQUI.parent / "Data"

# Fuera del corpus: ver el docstring.
EXCLUDED = frozenset({"popular-tarifario.md"})

# Enumerables: un producto por título `##`.
STRUCTURED_SUFFIX = "-detalle.md"


def load_documents() -> list[Document]:
    """Los markdown del corpus, con `structured` puesto según si el archivo es enumerable."""
    paths = sorted(path for path in CORPUS.glob("*.md") if path.name not in EXCLUDED)
    if not paths:
        message = f"no hay markdown en {CORPUS}"
        raise FileNotFoundError(message)
    return [
        Document(
            id=path.stem,
            content=path.read_text(encoding="utf-8"),
            structured=path.name.endswith(STRUCTURED_SUFFIX),
            kind="pagina-de-producto" if path.name.endswith(STRUCTURED_SUFFIX) else "pagina-de-portal",
            metadata={"archivo": path.name},
        )
        for path in paths
    ]
