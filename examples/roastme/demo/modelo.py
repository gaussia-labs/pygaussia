"""El modelo del usuario, y las credenciales, en un solo lugar.

Existía copiado en `run_profile.py`, `run_exploiter.py` y `motores_llm.py`, y las tres copias
tenían que coincidir para que las corridas fueran comparables — el juez, el torcedor y el generador
de consultas del Exploiter apuntan al mismo router y comparten cabeceras de facturación. Tres
copias de una constante que tiene que ser la misma es una que se va a separar.

Lo que gaussia pide es **un chat model de LangChain**, y nada más: ni una API key ni un proveedor.
Eso es lo que hace que el proveedor sea decisión del usuario y no de la librería (FR-043). Acá es
el router de HuggingFace porque es el único de los cinco que sirven este modelo que devuelve
`top_logprobs`, y sin eso el grader se niega a graduar en vez de degradar al muestreo.
"""

from __future__ import annotations

import os
import pathlib

AQUI = pathlib.Path(__file__).resolve().parent
BASE_URL = "https://router.huggingface.co/v1"


def cargar_env() -> None:
    """Las credenciales desde `.env`, sin pisar lo que ya esté en el ambiente."""
    archivo = AQUI / ".env"
    if not archivo.exists():
        return
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, _, valor = linea.partition("=")
            os.environ.setdefault(clave.strip(), valor.strip())


def build_modelo(temperature: float = 0.0):
    """El chat model. `temperature=0` para el juez y el torcedor, `1.0` para generar consultas.

    El generador de consultas del Exploiter necesita variedad —veinte consultas de una categoría no
    pueden ser la misma— así que ese es el único que sube la temperatura. El juez y el torcedor van
    en cero para que dos corridas sobre el mismo corpus sean comparables.
    """
    from langchain_openai import ChatOpenAI

    cargar_env()
    cabeceras = {}
    if os.environ.get("HF_BILL_TO"):
        cabeceras["X-HF-Bill-To"] = os.environ["HF_BILL_TO"]
    return ChatOpenAI(
        model=os.environ["HF_JUDGE_MODEL"],
        api_key=os.environ["HF_TOKEN"],
        base_url=BASE_URL,
        temperature=temperature,
        default_headers=cabeceras or None,
        # El router encola y devuelve 429 `queue_exceeded` bajo carga. Con los 2 reintentos por
        # default, un pico mata la generación entera en el trozo 40 de 84 y se pierde todo lo
        # anterior: el motor devuelve la lista al final y no hay checkpoint. 8 con el backoff del
        # SDK aguanta los picos que se midieron. El timeout sube por la misma razón: la cola es
        # espera, no falla, y cortarla a los 60s la convierte en una.
        max_retries=8,
        timeout=180,
    )


GROQ_URL = "https://api.groq.com/openai/v1"


def build_modelo_generador(temperature: float = 1.0):
    """El modelo que ESCRIBE consultas, que no es el mismo que las juzga.

    Dos proveedores en una corrida no es descuido. El juez necesita `top_logprobs` y por eso va
    contra el router de HF; el generador de consultas y el filtro on-profile no necesitan logprobs
    y sí necesitan ser rápidos y baratos, porque el Exploiter los llama veinte veces por categoría.

    `temperature=1.0` por default y no cero: veinte consultas de una misma categoría tienen que ser
    distintas entre sí, y en cero el generador devuelve la misma frase reformulada. El filtro
    on-profile usa la misma función en cero, porque ahí la variedad es ruido.
    """
    from langchain_openai import ChatOpenAI

    cargar_env()
    return ChatOpenAI(
        model=os.environ["GROQ_MODEL"],
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_URL,
        temperature=temperature,
    )
