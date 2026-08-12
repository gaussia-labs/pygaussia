"""Carga `.env` en el entorno, sin dependencias.

Diez líneas en lugar de `python-dotenv` a propósito: el README afirma que este camino corre con
`pip install gaussia` y nada más, y agregar una dependencia para leer un archivo de `clave=valor` haría
que esa afirmación deje de ser cierta.

Las variables que ya están en el entorno **ganan** sobre el archivo. Es lo que permite pisar una sola
para una corrida —`GROQ_MODEL=... python run_profile.py`— sin editar el `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = Path(__file__).resolve().parent / ".env"

_COMENTARIO = "#"
_ASIGNACION = "="
_COMILLAS = "\"'"


def load_env(archivo: Path = ENV) -> list[str]:
    """Mete en `os.environ` lo que el archivo declare. Devuelve los nombres que cargó, para poder decirlo.

    Un archivo que no existe no es un error: la corrida puede venir con las variables ya exportadas, y
    quien no tenga credenciales corre por replay y con el grader de reglas.
    """
    if not archivo.exists():
        return []
    cargadas = []
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        limpia = linea.strip()
        if not limpia or limpia.startswith(_COMENTARIO) or _ASIGNACION not in limpia:
            continue
        nombre, _, valor = limpia.partition(_ASIGNACION)
        nombre = nombre.strip().removeprefix("export ").strip()
        valor = valor.strip().strip(_COMILLAS)
        if nombre and valor and nombre not in os.environ:
            os.environ[nombre] = valor
            cargadas.append(nombre)
    return cargadas
