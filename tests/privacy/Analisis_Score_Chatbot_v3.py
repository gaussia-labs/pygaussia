#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  Analisis_Score_Chatbot_v3.py
#  Domain-Adjusted Privacy Detection Score  —  Experimento Sandbox
#
#  PROPÓSITO
#  ─────────
#  Evalúa cuán bien cada modelo de detección de PII identifica entidades
#  sensibles dentro de conversaciones de chatbot.  El resultado es un
#  Score compuesto que pondera calidad de detección, cobertura taxonómica,
#  ajuste al dominio y penalización por falsos negativos críticos.
#
#  DIFERENCIAS RESPECTO A LA VERSIÓN 2
#  ─────────────────────────────────────
#  • Se eliminan las capas REGEX y NER: los modelos se evalúan en su estado
#    nativo, sin suplemento de patrones aprendidos del corpus.
#  • Se elimina el InfraScore: el experimento se centra exclusivamente en la
#    capacidad de detección, no en latencia ni costos operativos.
#  • Nueva fórmula del score:
#
#      Score(M,d) = DetectionScore × Coverage × DomainFit
#                  × RegulatoryFit × Penalty_FN
#
#  ARCHIVOS DE ENTRADA  (configurables en la sección 2)
#  ─────────────────────────────────────────────────────
#  TAGGED_FILE    Corpus con entidades marcadas inline: <TIPO>valor</TIPO>
#                 Uso: ground truth para la evaluación.
#
#  UNTAGGED_FILE  El mismo corpus SIN etiquetas.
#                 Uso: texto que se entrega a los modelos para predicción.
#
#  El par de archivos se genera con el script de generación de datasets.
#  Ejemplo: eval_tagged_100.txt / eval_untagged_100.txt
#
#  SALIDA
#  ──────
#  JSON compatible con panel_chatbot_v3.py
# =============================================================================


# =============================================================================
#  SECCIÓN 0 — Auto-instalación de dependencias
#
#  Antes de importar cualquier librería de terceros, este bloque verifica
#  si las dependencias necesarias están disponibles y las instala si no.
#  Esto permite ejecutar el script en un entorno limpio sin configuración
#  previa (comportamiento típico de experimento sandbox).
# =============================================================================

import sys
import subprocess


def _pip_install(*packages: str) -> None:
    """Instala uno o más paquetes usando pip de forma silenciosa."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_dependencies() -> None:
    """
    Verifica e instala las dependencias requeridas.

    Obligatorias:
      - presidio-analyzer  : motor de análisis de PII de Microsoft Presidio.
                             Incluye reconocedores basados en regex para los
                             tipos estándar (email, teléfono, IBAN, etc.).
      - spacy              : procesamiento de lenguaje natural. Se usa en modo
                             "blank" (sin modelo de red neuronal) para que
                             Presidio pueda tokenizar sin descargar modelos.

    Opcionales (solo si se evalúan modelos HuggingFace):
      - transformers       : librería de modelos de lenguaje pre-entrenados.
      - torch              : backend de deep learning requerido por transformers.
    """
    missing = []
    for package, import_name in [
        ("presidio-analyzer", "presidio_analyzer"),
        ("spacy",             "spacy"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"📦  Instalando dependencias faltantes: {', '.join(missing)}")
        _pip_install(*missing)
        print("✅  Instalación completada.\n")

    # Informar sobre transformers si no está presente
    try:
        import transformers  # noqa: F401
    except ImportError:
        print(
            "ℹ️  'transformers' no está instalado.\n"
            "   Los modelos HuggingFace se marcarán como no disponibles.\n"
            "   Para habilitarlos ejecute:  pip install transformers torch\n"
        )


_ensure_dependencies()


# =============================================================================
#  SECCIÓN 1 — Imports principales
#
#  Se importan aquí, después del auto-instalador, para garantizar que las
#  librerías ya existen en el entorno antes de intentar cargarlas.
# =============================================================================

import os
import re
import gc
import json
import time
import math
import warnings
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional

warnings.filterwarnings("ignore")

# HuggingFace (importación condicional: solo disponible si transformers está instalado)
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Presidio (debería estar disponible tras el auto-instalador)
try:
    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


# =============================================================================
#  SECCIÓN 2 — Configuración general
#
#  Parámetros que el usuario puede ajustar para personalizar el experimento.
# =============================================================================

# Dispositivo para modelos HuggingFace:
#   -1  = CPU  (seguro para cualquier entorno)
#    0  = primera GPU  (requiere CUDA)
DEVICE = -1

# Umbral de IoU (Intersection over Union) para considerar que una predicción
# coincide con el ground truth. Un valor de 0.50 significa que al menos el 50%
# del span predicho debe solapar con el span real.
IOU_THRESHOLD = 0.50

# Si es True, ejecuta los modelos en paralelo (solo relevante para HuggingFace).
# Desactivado por defecto para evitar saturación de memoria en CPU.
USE_PARALLEL = False
MAX_WORKERS  = 2

# Ruta y nombre del archivo JSON de resultados.
# Este archivo es leído por panel_chatbot_v3.py para generar el panel HTML.
SAVE_RESULTS_JSON = True
OUTPUT_JSON_PATH  = "chatbot_v3_results.json"

# Si es True y transformers no está disponible, los modelos HuggingFace se
# omiten sin lanzar error. El script continúa con Presidio.
SKIP_HF_IF_UNAVAILABLE = True

# ── Rutas de los archivos de corpus ──────────────────────────────────────────
# Ajustar si los archivos están en otro directorio o tienen otro nombre.
TAGGED_FILE   = "chatbot_conversations_tagged_500.txt"    # corpus con etiquetas <TIPO>valor</TIPO>
UNTAGGED_FILE = "chatbot_conversations_500.txt"  # mismo corpus sin etiquetas


# =============================================================================
#  SECCIÓN 3 — Taxonomía del dominio
#
#  La taxonomía define los tipos de entidad PII que se evalúan.
#  Se deriva directamente de los tipos presentes en el corpus tageado.
#
#  ¿Por qué esta taxonomía?
#  Los archivos de corpus fueron generados para conversaciones de chatbot
#  generalistas (soporte, banca, salud, e-commerce, HR, telecomunicaciones).
#  Los 9 tipos capturan la mayoría del PII relevante en ese contexto.
#
#  Mapeo corpus-tag → etiqueta del dominio (lowercase con guiones bajos):
#    PERSON         → person          (nombre completo o parcial)
#    EMAIL_ADDRESS  → email_address
#    PHONE_NUMBER   → phone_number
#    STREET_ADDRESS → street_address
#    IBAN_CODE      → iban_code
#    CREDIT_CARD    → credit_card
#    DATE_TIME      → date_time
#    IP_ADDRESS     → ip_address
#    US_SSN         → us_ssn
# =============================================================================

# Orden en que se presentan los tipos en tablas y gráficos
DOMAIN_TAXONOMY: List[str] = [
    "person",
    "email_address",
    "phone_number",
    "street_address",
    "iban_code",
    "credit_card",
    "date_time",
    "ip_address",
    "us_ssn",
]

# Set para búsquedas rápidas de pertenencia
DOMAIN_CLASSES: Set[str] = set(DOMAIN_TAXONOMY)

# Mapeo explícito corpus-tag → etiqueta del dominio
CORPUS_TAG_TO_DOMAIN: Dict[str, str] = {
    "PERSON":         "person",
    "EMAIL_ADDRESS":  "email_address",
    "PHONE_NUMBER":   "phone_number",
    "STREET_ADDRESS": "street_address",
    "IBAN_CODE":      "iban_code",
    "CREDIT_CARD":    "credit_card",
    "DATE_TIME":      "date_time",
    "IP_ADDRESS":     "ip_address",
    "US_SSN":         "us_ssn",
}


# =============================================================================
#  SECCIÓN 4 — Pesos de la taxonomía
#
#  Existen dos conjuntos de pesos, ambos deben sumar exactamente 1.0:
#
#  CLASS_CRITICALITY_WEIGHTS  (w_{c,d})
#    Importancia relativa de cada tipo dentro del DetectionScore.
#    Un tipo con peso mayor contribuye más al score cuando el modelo
#    lo detecta bien.
#
#  CRITICAL_FN_SEVERITY_WEIGHTS  (ρ_{c,d})
#    Penalización por falsos negativos (entidades no detectadas).
#    Un tipo con peso mayor penaliza más cuando el modelo lo pierde.
#
#  En esta versión sandbox ambos diccionarios tienen los mismos valores
#  para reflejar que la importancia de detección y la severidad de omisión
#  son simétricas para este dominio.  En producción pueden diferir.
# =============================================================================

CLASS_CRITICALITY_WEIGHTS: Dict[str, float] = {
    "person":         0.15,   # Identificación de personas: alta criticidad
    "email_address":  0.15,   # Email: vector de contacto y ataque
    "phone_number":   0.12,   # Teléfono: datos de contacto directos
    "street_address": 0.08,   # Dirección: baja vs. otros pero relevante
    "iban_code":      0.15,   # IBAN: dato financiero de alta sensibilidad
    "credit_card":    0.13,   # Tarjeta: dato financiero crítico
    "date_time":      0.05,   # Fecha: contexto temporal, menor riesgo
    "ip_address":     0.09,   # IP: identificación de dispositivos
    "us_ssn":         0.08,   # SSN: identificador nacional sensible
}

CRITICAL_FN_SEVERITY_WEIGHTS: Dict[str, float] = {
    "person":         0.15,
    "email_address":  0.15,
    "phone_number":   0.12,
    "street_address": 0.08,
    "iban_code":      0.15,
    "credit_card":    0.13,
    "date_time":      0.05,
    "ip_address":     0.09,
    "us_ssn":         0.08,
}

# Validación automática de los pesos al cargar el módulo
assert abs(sum(CLASS_CRITICALITY_WEIGHTS.values()) - 1.0) < 1e-9, \
    "ERROR: CLASS_CRITICALITY_WEIGHTS no suma 1.0"
assert abs(sum(CRITICAL_FN_SEVERITY_WEIGHTS.values()) - 1.0) < 1e-9, \
    "ERROR: CRITICAL_FN_SEVERITY_WEIGHTS no suma 1.0"

# Parámetros de ajuste del dominio y regulatorio.
# Valores fijos en esta versión sandbox; en producción se ajustan por modelo
# según evidencia documentada de alineación al dominio clínico/legal.
DEFAULT_DOMAIN_FIT     = 0.90  # Qué tan bien el modelo se alinea al dominio
DEFAULT_REGULATORY_FIT = 0.85  # Qué tan bien cumple requisitos regulatorios


# =============================================================================
#  SECCIÓN 5 — Alias de etiquetas
#
#  Los modelos HuggingFace y Presidio usan nomenclaturas propias.
#  Este diccionario mapea esas etiquetas hacia la taxonomía del dominio.
#
#  Ejemplo:
#    - OpenMed puede predecir "first_name" o "last_name" → mapeamos a "person"
#    - Presidio estándar usa "DATE_TIME" → canonicalize lo convierte a "date_time"
#
#  Cualquier etiqueta no mapeada explícitamente, y que no pertenezca a
#  DOMAIN_CLASSES, será ignorada en la evaluación (no cuenta como FP).
# =============================================================================

LABEL_ALIASES: Dict[str, str] = {
    # OpenMed (modelos clínicos) → dominio chatbot
    "first_name":          "person",
    "last_name":           "person",
    "date":                "date_time",
    "date_of_birth":       "date_time",
    "birthday":            "date_time",
    "email":               "email_address",
    "phone":               "phone_number",
    "address":             "street_address",
    "location":            "street_address",
    "ip":                  "ip_address",
    "ssn":                 "us_ssn",
    "bank_routing_number": "iban_code",
    "account_number":      "iban_code",
    "credit_card_number":  "credit_card",
    # Presidio estándar → dominio
    "mac_address":         "ip_address",
}


def canonicalize(label: str) -> str:
    """
    Normaliza una etiqueta de entidad hacia la taxonomía del dominio.

    Pasos:
      1. Elimina prefijos BIO/BILOU (B-, I-, E-, S-, U-, L-)
      2. Convierte a minúsculas
      3. Reemplaza espacios y guiones por guiones bajos
      4. Aplica LABEL_ALIASES si existe un mapeo

    Devuelve la etiqueta normalizada, o "unknown" si la entrada es vacía.
    """
    if not label:
        return "unknown"
    # Eliminar prefijos de esquema de etiquetado (BIO/BILUO)
    normalized = re.sub(r"^(B|I|E|S|U|L)[-_]", "", str(label).strip(),
                        flags=re.IGNORECASE)
    normalized = normalized.lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)  # colapsar guiones múltiples
    return LABEL_ALIASES.get(normalized, normalized)


# =============================================================================
#  SECCIÓN 6 — Configuración de modelos
#
#  Se definen dos grupos de modelos:
#
#  HF_MODELS_CONFIG
#    Modelos de la familia OpenMed en HuggingFace Hub.
#    Todos son modelos de NER fine-tuneados sobre textos clínicos para
#    detectar PII médico.  Se cargan mediante el pipeline de transformers.
#
#  PRESIDIO_MODELS_CONFIG
#    Presidio Baseline: el motor de análisis estándar de Microsoft Presidio
#    sin ninguna configuración adicional.  Usa reconocedores basados en regex
#    y listas de valores para los tipos PII más comunes.
#
#  Nota sobre domain_fit y regulatory_fit:
#    Los modelos OpenMed fueron entrenados sobre textos clínicos (HIPAA),
#    por lo que su domain_fit y regulatory_fit son ligeramente más altos
#    que Presidio para el dominio de salud.  En chatbot general son iguales.
# =============================================================================

HF_MODELS_CONFIG: Dict[str, Dict] = {
    "OpenMed-PII-SuperClinical-434M": {
        "path":            "openmed/OpenMed-PII-SuperClinical-Large-434M-v1",
        "paradigm":        "huggingface",
        "domain_fit":      DEFAULT_DOMAIN_FIT,
        "regulatory_fit":  DEFAULT_REGULATORY_FIT,
    },
    "OpenMed-PII-BigMed-BioClinical": {
        "path":            "openmed/OpenMed-PII-BigMed-Large-278M-v1",
        "paradigm":        "huggingface",
        "domain_fit":      DEFAULT_DOMAIN_FIT,
        "regulatory_fit":  DEFAULT_REGULATORY_FIT,
    },
    "OpenMed-PII-ModernMed-Large-395M": {
        "path":            "OpenMed/OpenMed-PII-ModernMed-Large-395M-v1",
        "paradigm":        "huggingface",
        "domain_fit":      DEFAULT_DOMAIN_FIT,
        "regulatory_fit":  DEFAULT_REGULATORY_FIT,
    },
    "OpenMed-PII-SuperMedical-Large-355M": {
        "path":            "OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1",
        "paradigm":        "huggingface",
        "domain_fit":      DEFAULT_DOMAIN_FIT,
        "regulatory_fit":  DEFAULT_REGULATORY_FIT,
    },
    "OpenMed-privacy-filter-nemotron": {
        "path":            "OpenMed/privacy-filter-nemotron",
        "paradigm":        "huggingface",
        "domain_fit":      DEFAULT_DOMAIN_FIT,
        "regulatory_fit":  DEFAULT_REGULATORY_FIT,
    },
}

PRESIDIO_MODELS_CONFIG: Dict[str, Dict] = {
    "Presidio-Baseline": {
        "paradigm":       "presidio_baseline",
        "domain_fit":     DEFAULT_DOMAIN_FIT,
        "regulatory_fit": DEFAULT_REGULATORY_FIT,
    },
}

# Unión de todos los modelos a evaluar
ALL_MODELS: Dict[str, Dict] = {**HF_MODELS_CONFIG, **PRESIDIO_MODELS_CONFIG}


# =============================================================================
#  SECCIÓN 7 — Parsers de archivos .txt
#
#  El corpus usa el formato de transcripción de chatbot con etiquetas inline:
#
#    CHATBOT: Hello! Welcome to support.
#    USER: My email is <EMAIL_ADDRESS>foo@bar.com</EMAIL_ADDRESS>.
#    USER: My name is <PERSON>John Doe</PERSON>.
#
#  El parser extrae:
#    - clean_text: el texto sin etiquetas (lo que ve el modelo)
#    - spans: lista de entidades con posición en clean_text (ground truth)
# =============================================================================

# Patrón para encontrar tags inline: <TIPO>valor</TIPO>
_TAG_INLINE  = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.DOTALL)

# Patrón para eliminar el prefijo del hablante (CHATBOT:, USER:, etc.)
_SPEAKER_PRE = re.compile(
    r"^(CHATBOT|PATIENT|CUSTOMER|USER|EMPLOYEE)\s*:\s*",
    re.IGNORECASE,
)


def _strip_speaker(line: str) -> str:
    """Elimina el prefijo del hablante de una línea de conversación."""
    match = _SPEAKER_PRE.match(line)
    return line[match.end():] if match else line


def parse_tagged_file(path: str) -> List[Tuple[str, List[Dict]]]:
    """
    Lee el archivo tageado y extrae pares (clean_text, ground_truth_spans).

    Para cada línea del archivo:
      1. Elimina el prefijo del hablante.
      2. Busca todas las ocurrencias de <TIPO>valor</TIPO>.
      3. Calcula la posición de cada entidad en el texto limpio (sin tags).
      4. Construye el ground truth como lista de dicts con 'text', 'label',
         'start' y 'end'.

    Nota sobre el cálculo de posiciones:
      Al eliminar las etiquetas del texto, los índices cambian.  Se lleva
      un contador 'offset' que acumula cuántos caracteres de tags se han
      eliminado hasta cada punto, ajustando las posiciones correctamente.

    Devuelve lista de (clean_text, [{"text":..., "label":..., "start":..., "end":...}])
    Omite líneas vacías y separadores "---".
    """
    records = []

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").strip()
            if not line or line == "---":
                continue

            text   = _strip_speaker(line)
            spans  = []
            offset = 0  # caracteres de tags acumulados hasta aquí

            for match in _TAG_INLINE.finditer(text):
                corpus_tag   = match.group(1)
                entity_value = match.group(2)
                domain_label = CORPUS_TAG_TO_DOMAIN.get(corpus_tag)

                # Ignorar tipos de entidad no mapeados al dominio
                if not domain_label or not entity_value.strip():
                    # Aun así acumular el desplazamiento de los tags
                    offset += len(f"<{corpus_tag}>") + len(f"</{corpus_tag}>")
                    continue

                # Posición en el texto limpio = posición en texto tageado - offset
                start_clean = match.start() - offset
                end_clean   = start_clean + len(entity_value)

                spans.append({
                    "text":  entity_value,
                    "label": domain_label,
                    "start": start_clean,
                    "end":   end_clean,
                })

                # Actualizar offset con la longitud de las etiquetas eliminadas
                offset += len(f"<{corpus_tag}>") + len(f"</{corpus_tag}>")

            # Texto sin ninguna etiqueta
            clean_text = _TAG_INLINE.sub(r"\2", text)
            records.append((clean_text, spans))

    return records


def parse_untagged_file(path: str) -> List[str]:
    """
    Lee el archivo sin etiquetar y devuelve una lista de textos limpios.

    Estos textos son el input real que reciben los modelos:
    no contienen ninguna etiqueta ni indicación de qué entidades existen.
    """
    texts = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").strip()
            if not line or line == "---":
                continue
            texts.append(_strip_speaker(line))
    return texts


def align_files(
    tagged_records: List[Tuple[str, List[Dict]]],
    untagged_texts: List[str],
) -> List[Tuple[str, List[Dict]]]:
    """
    Empareja línea a línea los dos archivos del corpus.

    Devuelve lista de (untagged_text, ground_truth_spans) donde:
      - untagged_text: texto limpio del archivo sin etiquetar
                       (input para el modelo)
      - ground_truth_spans: posiciones de entidades extraídas del tageado
                            (verdad de terreno para evaluación)

    Si los archivos tienen distinto número de líneas, usa el mínimo
    y emite un aviso.  Esto puede ocurrir si un archivo tiene
    separadores "---" adicionales.
    """
    n = min(len(tagged_records), len(untagged_texts))

    if len(tagged_records) != len(untagged_texts):
        print(
            f"  [ADVERTENCIA] Número de líneas diferente: "
            f"tagged={len(tagged_records)}, untagged={len(untagged_texts)}. "
            f"Se usarán las primeras {n} líneas."
        )

    # El texto de referencia para el modelo es el del archivo sin etiquetar
    # (garantiza que el modelo no "ve" las etiquetas)
    return [(untagged_texts[i], tagged_records[i][1]) for i in range(n)]


# =============================================================================
#  SECCIÓN 8 — Motor Presidio
#
#  Presidio se configura con un modelo spaCy en blanco ("blank") para evitar
#  la descarga de modelos de red neuronal. El modelo blank solo tokeniza,
#  lo que es suficiente para que los reconocedores basados en regex funcionen.
#
#  Mapa de etiquetas Presidio → dominio:
#    Los reconocedores estándar de Presidio producen etiquetas en mayúsculas
#    (PERSON, EMAIL_ADDRESS, etc.).  Este mapa las convierte a la taxonomía
#    del dominio (minúsculas).
# =============================================================================

# Tipos Presidio estándar que se mapean al dominio chatbot.
# Tipos NO listados aquí serán ignorados (no producen ni TP ni FP).
_PRESIDIO_TO_DOMAIN: Dict[str, str] = {
    "PERSON":          "person",
    "EMAIL_ADDRESS":   "email_address",
    "EMAIL":           "email_address",
    "PHONE_NUMBER":    "phone_number",
    "IBAN_CODE":       "iban_code",
    "CREDIT_CARD":     "credit_card",
    "DATE_TIME":       "date_time",
    "IP_ADDRESS":      "ip_address",
    "US_SSN":          "us_ssn",
    "LOCATION":        "street_address",
    "NRP":             "person",
}

# Clases del dominio que Presidio cubre de forma nativa (sin capa adicional)
_PRESIDIO_SUPPORTED: Set[str] = set(_PRESIDIO_TO_DOMAIN.values())


def _build_blank_presidio_engine() -> "AnalyzerEngine":
    """
    Construye un AnalyzerEngine de Presidio usando spaCy en blanco.

    Se inyecta el modelo blank en el caché interno de span_to_tag de
    presidio_evaluator (si está instalado) para evitar que intente
    descargar 'en_core_web_sm' desde internet.

    No requiere conexión a red.
    """
    # Intentar parchear el caché de presidio_evaluator si está presente
    try:
        import presidio_evaluator  # noqa: F401
        _st = sys.modules.get("presidio_evaluator.span_to_tag")
        if _st and hasattr(_st, "loaded_spacy"):
            _nlp_patch = spacy.blank("en")
            _nlp_patch.add_pipe("sentencizer")
            _st.loaded_spacy["en_core_web_sm"] = _nlp_patch
    except ImportError:
        pass  # presidio_evaluator no requerido en v3

    # Construir motor con modelo blank
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": "blank_en"}]
    )
    nlp_engine.nlp = {"en": nlp}

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )


def _presidio_predict(
    text:     str,
    analyzer: "AnalyzerEngine",
) -> List[Dict]:
    """
    Ejecuta Presidio sobre un texto y mapea las predicciones al dominio.

    Solo se conservan predicciones cuyo tipo está en _PRESIDIO_TO_DOMAIN.
    El resto se descarta silenciosamente (no son FP en esta evaluación).
    """
    results = analyzer.analyze(text=text, language="en")
    predictions = []

    for res in results:
        domain_label = _PRESIDIO_TO_DOMAIN.get(res.entity_type)
        if domain_label is None:
            continue  # tipo no relevante para el dominio chatbot

        predictions.append({
            "text":  text[res.start:res.end],
            "label": domain_label,
            "start": res.start,
            "end":   res.end,
            "score": res.score,
        })

    return predictions


# =============================================================================
#  SECCIÓN 9 — Utilidades de métricas
#
#  Funciones de bajo nivel para calcular métricas de evaluación.
#  Todas son puras (sin estado) para facilitar pruebas y depuración.
# =============================================================================

def _safe_div(numerator: float, denominator: float) -> float:
    """División segura: devuelve 0 si el denominador es 0."""
    return numerator / denominator if denominator else 0.0


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """
    Calcula el F-beta score.

    beta=2.0 (F2) pondera el recall dos veces más que la precision.
    Apropiado para detección de PII donde un falso negativo (entidad
    no detectada) es más costoso que un falso positivo.

    Devuelve 0.0 si precision + recall = 0.
    """
    denominator = beta ** 2 * precision + recall
    if not denominator:
        return 0.0
    return (1 + beta ** 2) * precision * recall / denominator


def _spans_overlap(a: Dict, b: Dict) -> bool:
    """
    Determina si dos spans de texto se solapan.
    Dos spans [a_start, a_end) y [b_start, b_end) solapan cuando
    max(a_start, b_start) < min(a_end, b_end).
    """
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _remove_overlapping_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Elimina predicciones solapadas conservando la de mayor score.

    Algoritmo greedy: ordena por score descendente, acepta una predicción
    solo si no solapa con ninguna ya aceptada.
    """
    selected = []
    for pred in sorted(predictions, key=lambda e: (-e["score"], e["start"])):
        if not any(_spans_overlap(pred, s) for s in selected):
            selected.append(pred)
    return sorted(selected, key=lambda e: e["start"])


def _span_iou(a: Dict, b: Dict) -> float:
    """
    Calcula la IoU (Intersection over Union) entre dos spans de texto.

    IoU = longitud_intersección / longitud_unión

    Utilizado para matching fuzzy: una predicción "casi correcta" en posición
    pero que cubre al menos IOU_THRESHOLD del span real se cuenta como TP.
    """
    intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if not intersection:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return intersection / union


# =============================================================================
#  SECCIÓN 10 — Evaluación agregada sobre el corpus
#
#  Esta función es el núcleo de la evaluación.
#
#  Para cada par (texto_limpio, ground_truth) del corpus:
#    1. Ejecuta el modelo para obtener predicciones.
#    2. Filtra predicciones fuera del dominio (no generan FP).
#    3. Realiza matching uno-a-uno entre predicciones y ground truth
#       usando IoU >= IOU_THRESHOLD y coincidencia de etiqueta.
#    4. Clasifica cada predicción como TP, FP o FN.
#
#  Al final agrega TP/FP/FN a nivel de corpus para calcular métricas globales.
#
#  IMPORTANTE — filtro de dominio:
#    Las predicciones con etiquetas fuera de DOMAIN_CLASSES se descartan.
#    Esto evita que etiquetas propias de los modelos (age, api_key, cvv, etc.)
#    inflen los FP y distorsionen la Micro-Precision, que no está incluida
#    en el score final pero se muestra como indicador diagnóstico.
# =============================================================================

def aggregate_predictions(
    pairs:      List[Tuple[str, List[Dict]]],
    predict_fn,                               # callable(text) → List[Dict]
    iou_thr:    float = IOU_THRESHOLD,
) -> Dict[str, Any]:
    """
    Corre predict_fn sobre cada par del corpus y agrega métricas globales.

    Parámetros:
      pairs       : lista de (clean_text, ground_truth_spans)
      predict_fn  : función que recibe un texto y devuelve lista de predicciones
      iou_thr     : umbral mínimo de IoU para contar una predicción como TP

    Devuelve diccionario con:
      metrics_by_label      : TP/FP/FN/precision/recall/f2/fn_rate por etiqueta
      micro_metrics         : métricas agregadas sobre todas las etiquetas
      corpus_samples        : número total de turnos evaluados
      total_gt_entities     : entidades en el ground truth
      total_pred_entities   : predicciones dentro del dominio
      total_pred_raw        : predicciones totales antes del filtro
      out_of_domain_filtered: predicciones descartadas por ser fuera del dominio
    """
    counts: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_gt   = 0
    total_raw  = 0
    total_dom  = 0

    for clean_text, gt_spans in pairs:

        # 1. Obtener predicciones brutas del modelo
        raw_preds  = predict_fn(clean_text)
        total_raw += len(raw_preds)

        # 2. Filtrar solo predicciones dentro del dominio
        domain_preds = [p for p in raw_preds if p["label"] in DOMAIN_CLASSES]
        predictions  = _remove_overlapping_predictions(domain_preds)
        total_dom   += len(predictions)
        total_gt    += len(gt_spans)

        # 3. Matching greedy predicción → ground truth
        #    Se recorre de mayor a menor score para priorizar predicciones
        #    de mayor confianza en el matching.
        unmatched_gt = set(range(len(gt_spans)))

        for pred in sorted(predictions, key=lambda e: e["score"], reverse=True):
            best_iou  = 0.0
            best_gt_i = -1

            for gt_i in unmatched_gt:
                gt = gt_spans[gt_i]
                # Solo hacer matching si la etiqueta coincide
                if pred["label"] == gt["label"]:
                    iou = _span_iou(pred, gt)
                    if iou >= iou_thr and iou > best_iou:
                        best_iou  = iou
                        best_gt_i = gt_i

            if best_gt_i >= 0:
                # Predicción correcta → True Positive
                unmatched_gt.remove(best_gt_i)
                counts[gt_spans[best_gt_i]["label"]]["tp"] += 1
            else:
                # Sin match → False Positive
                counts[pred["label"]]["fp"] += 1

        # 4. Los GT no emparejados son False Negatives
        for gt_i in unmatched_gt:
            counts[gt_spans[gt_i]["label"]]["fn"] += 1

    # 5. Calcular métricas por etiqueta
    metrics_by_label: Dict[str, Dict] = {}
    all_labels = DOMAIN_CLASSES | set(counts.keys())

    for label in sorted(all_labels):
        tp  = counts[label]["tp"]
        fp  = counts[label]["fp"]
        fn  = counts[label]["fn"]
        p   = _safe_div(tp, tp + fp)
        r   = _safe_div(tp, tp + fn)
        metrics_by_label[label] = {
            "tp":                tp,
            "fp":                fp,
            "fn":                fn,
            "precision":         p,
            "recall":            r,
            "f2":                _f_beta(p, r),
            "fn_rate":           _safe_div(fn, tp + fn),
            "criticality_weight": CLASS_CRITICALITY_WEIGHTS.get(label, 0.0),
            "fn_severity_weight": CRITICAL_FN_SEVERITY_WEIGHTS.get(label, 0.0),
        }

    # 6. Métricas micro-agregadas (sobre todas las etiquetas)
    ttp = sum(m["tp"] for m in metrics_by_label.values())
    tfp = sum(m["fp"] for m in metrics_by_label.values())
    tfn = sum(m["fn"] for m in metrics_by_label.values())
    mp  = _safe_div(ttp, ttp + tfp)
    mr  = _safe_div(ttp, ttp + tfn)

    return {
        "metrics_by_label":        metrics_by_label,
        "micro_metrics": {
            "tp":        ttp,
            "fp":        tfp,
            "fn":        tfn,
            "precision": mp,
            "recall":    mr,
            "f2":        _f_beta(mp, mr),
        },
        "corpus_samples":           len(pairs),
        "total_gt_entities":        total_gt,
        "total_pred_entities":      total_dom,
        "total_pred_raw":           total_raw,
        "out_of_domain_filtered":   total_raw - total_dom,
    }


# =============================================================================
#  SECCIÓN 11 — Cálculo del score Domain-Adjusted
#
#  Fórmula (sin InfraScore):
#
#    Score(M,d) = DetectionScore(M,d)
#               × Coverage(M,d)
#               × DomainFit(M,d)
#               × RegulatoryFit(M,d)
#               × Penalty_FN(M,d)
#
#  Cada componente:
#
#  DetectionScore  = Σ_c [ w_{c,d} × F2_{M,c} ]
#    Promedio ponderado del F2 por clase.  Pondera más las clases críticas.
#
#  Coverage  = |clases_soportadas_M ∩ DOMAIN_CLASSES| / |DOMAIN_CLASSES|
#    Proporción del dominio que el modelo puede cubrir en principio.
#    Un modelo que solo detecta personas y fechas tiene cobertura baja.
#
#  DomainFit  [0, 1]
#    Valor fijo que representa qué tan bien el modelo fue entrenado para
#    el dominio evaluado (chatbot de soporte/banca/salud).
#
#  RegulatoryFit  [0, 1]
#    Qué tan bien el modelo se alinea con requisitos normativos (GDPR,
#    HIPAA, etc.) para el tipo de datos que maneja.
#
#  Penalty_FN  = 1 − CriticalFN
#    CriticalFN = Σ_c [ ρ_{c,d} × FNrate_{M,c} ]
#    Penalización por entidades criticas que el modelo NO detectó.
#    Si el modelo pierde muchas entidades de alto riesgo (api_key, iban),
#    la penalización es alta y el score cae.
#
#  Índices de riesgo:
#
#  R1 = max_c [ ρ_{c,d} × FNrate_{M,c} ]
#    Riesgo concentrado en la clase más débil del modelo.
#
#  R2 = 1 − Penalty_FN × Coverage
#    Riesgo sistémico: combinación de fallos distribuidos y cobertura baja.
#
#  R_final = 1 − (1 − R1) × (1 − R2)
#    Unión probabilística de R1 y R2.  Solo es 0 cuando ambos son 0.
# =============================================================================

def compute_score(
    evaluation:  Dict,
    cfg:         Dict,
    load_time:   float,
    latency:     float,
    supported:   Set[str],
    cov_basis:   str,
    model_name:  str,
    paradigm:    str,
    extra:       Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Calcula todos los componentes del score y empaqueta el resultado.

    Parámetros:
      evaluation  : salida de aggregate_predictions()
      cfg         : configuración del modelo (domain_fit, regulatory_fit)
      load_time   : tiempo de carga del modelo en segundos
      latency     : latencia de inferencia acumulada en segundos
      supported   : conjunto de etiquetas del dominio que el modelo soporta
      cov_basis   : descripción de cómo se determinó 'supported'
      model_name  : nombre del modelo para el resultado
      paradigm    : "huggingface" o "presidio_baseline"
      extra       : campos adicionales opcionales a incluir en el resultado

    Devuelve diccionario completo compatible con panel_chatbot_v3.py.
    """
    mbl   = evaluation["metrics_by_label"]
    micro = evaluation["micro_metrics"]

    # ── DetectionScore ────────────────────────────────────────────────────────
    # Suma ponderada del F2 por clase del dominio.
    # Solo considera clases del DOMAIN_TAXONOMY (no etiquetas extra del modelo).
    det_score   = 0.0
    det_contrib = {}
    for label in DOMAIN_TAXONOMY:
        weight = CLASS_CRITICALITY_WEIGHTS[label]
        f2     = mbl.get(label, {}).get("f2", 0.0)
        contrib = weight * f2
        det_score += contrib
        det_contrib[label] = {"weight": weight, "f2": f2, "contribution": contrib}

    # ── Coverage ──────────────────────────────────────────────────────────────
    # Proporción de clases del dominio que el modelo puede, en principio, detectar.
    # Para HuggingFace: clases listadas en id2label del modelo.
    # Para Presidio:    clases cubiertas por sus reconocedores estándar.
    covered  = supported & DOMAIN_CLASSES
    missing  = DOMAIN_CLASSES - supported
    coverage = len(covered) / len(DOMAIN_CLASSES) if DOMAIN_CLASSES else 0.0

    # ── Penalty_FN ────────────────────────────────────────────────────────────
    # Penalización por falsos negativos en clases críticas.
    # CriticalFN = suma de (severidad × tasa_de_FN) por clase.
    critical_fn   = 0.0
    cfn_contrib   = {}
    for label in DOMAIN_TAXONOMY:
        severity = CRITICAL_FN_SEVERITY_WEIGHTS[label]
        fn_rate  = mbl.get(label, {}).get("fn_rate", 0.0)
        contrib  = severity * fn_rate
        critical_fn += contrib
        cfn_contrib[label] = {
            "severity_weight": severity,
            "fn_rate":         fn_rate,
            "contribution":    contrib,
        }
    penalty_fn = max(0.0, 1.0 - critical_fn)

    # ── Índices de riesgo ─────────────────────────────────────────────────────
    r1       = 0.0
    r1_label = None
    for label in DOMAIN_TAXONOMY:
        c = CRITICAL_FN_SEVERITY_WEIGHTS[label] * mbl.get(label, {}).get("fn_rate", 0.0)
        if c > r1:
            r1       = c
            r1_label = label

    r2      = 1.0 - penalty_fn * coverage
    r_final = 1.0 - (1.0 - r1) * (1.0 - r2)

    # ── Score final ───────────────────────────────────────────────────────────
    # Producto de los 5 componentes (sin InfraScore).
    score = (
        det_score
        * coverage
        * cfg["domain_fit"]
        * cfg["regulatory_fit"]
        * penalty_fn
    )
    score_100 = score * 100.0

    # ── Interpretación cualitativa ────────────────────────────────────────────
    def _interpret(s: float) -> str:
        if s < 40:  return "No adecuado"
        if s < 60:  return "Adecuado solo como línea base o complemento"
        if s < 75:  return "Adecuado con mitigaciones"
        if s < 85:  return "Operacionalmente adecuado"
        if s < 95:  return "Recomendado"
        return "Recomendado con fuerte evidencia local"

    result = {
        "name":                 model_name,
        "paradigm":             paradigm,
        "success":              True,
        "load_time":            load_time,
        "inference_latency":    latency,

        # Conteos del corpus
        "ground_truth_count":   evaluation["total_gt_entities"],
        "raw_prediction_count": evaluation["total_pred_raw"],
        "prediction_count":     evaluation["total_pred_entities"],
        "corpus_samples":       evaluation["corpus_samples"],
        "out_of_domain_filtered": evaluation["out_of_domain_filtered"],

        # Evaluación completa
        "evaluation":           evaluation,

        # Score y componentes
        "detection_score":      det_score,
        "detection_score_contributions": det_contrib,
        "coverage":             coverage,
        "covered_domain_classes": sorted(covered),
        "missing_domain_classes": sorted(missing),
        "supported_classes":    sorted(supported),
        "coverage_basis":       cov_basis,
        "domain_fit":           cfg["domain_fit"],
        "regulatory_fit":       cfg["regulatory_fit"],
        "critical_fn":          critical_fn,
        "penalty_fn":           penalty_fn,
        "critical_fn_contributions": cfn_contrib,

        # Riesgo
        "r1_weakest_class_risk":     r1,
        "r1_weakest_class_risk_100": r1 * 100.0,
        "r1_weakest_class":          r1_label,
        "r2_systemic_risk":          r2,
        "r2_systemic_risk_100":      r2 * 100.0,
        "r_final":                   r_final,
        "r_final_100":               r_final * 100.0,

        # Score final
        "final_score":   score,
        "score_100":     score_100,
        "interpretation": _interpret(score_100),
    }

    if extra:
        result.update(extra)

    return result


# =============================================================================
#  SECCIÓN 12 — Extracción de clases soportadas por modelos HuggingFace
#
#  Para calcular Coverage necesitamos saber qué tipos de entidad puede
#  detectar cada modelo.  HuggingFace expone esta información en el
#  atributo id2label del config del modelo.
# =============================================================================

def _get_hf_supported_classes(ner_pipe) -> Set[str]:
    """
    Extrae los tipos de entidad del modelo HuggingFace y los mapea al dominio.

    Lee model.config.id2label, canonicaliza cada etiqueta (aplica LABEL_ALIASES),
    y filtra etiquetas irrelevantes (O, OUTSIDE, etc.).

    Si id2label no está disponible, devuelve un conjunto vacío.  En ese caso
    compute_score usará las etiquetas observadas en las predicciones reales como
    fallback (cobertura conservadora).
    """
    model   = getattr(ner_pipe, "model",  None)
    config  = getattr(model,    "config", None)
    id2label = getattr(config,  "id2label", None)

    if not isinstance(id2label, dict):
        return set()

    supported = set()
    irrelevant = {"o", "outside", "none", "other", "unknown"}

    for raw_label in id2label.values():
        label = canonicalize(raw_label)
        # Ignorar etiquetas genéricas y etiquetas tipo "label_42"
        if label not in irrelevant and not re.match(r"^label_\d+$", label):
            supported.add(label)

    return supported


# =============================================================================
#  SECCIÓN 13 — Evaluación de todos los modelos
#
#  Itera sobre ALL_MODELS y evalúa cada uno sobre el corpus completo.
#  Genera exactamente UN resultado por modelo (sin variantes base/+Domain).
#
#  Para modelos HuggingFace no disponibles (transformers no instalado):
#    Se registra un resultado de fallo con score=0 y success=False.
#    El script continúa sin interrumpirse.
# =============================================================================

def evaluate_all_models(
    pairs: List[Tuple[str, List[Dict]]],
) -> List[Dict[str, Any]]:
    """
    Evalúa todos los modelos definidos en ALL_MODELS sobre el corpus.

    Parámetros:
      pairs : lista de (untagged_text, ground_truth_spans) del corpus

    Devuelve lista de resultados ordenados por score_100 descendente.
    """
    results = []
    icons   = {"huggingface": "🤗", "presidio_baseline": "🛡️"}

    print(f"\n  Evaluando {len(pairs)} turnos × {len(ALL_MODELS)} modelos …\n")

    for model_name, cfg in ALL_MODELS.items():
        paradigm = cfg.get("paradigm", "?")
        icon     = icons.get(paradigm, "⏳")
        print(f"{icon}  {model_name} …", end=" ", flush=True)

        # ── HuggingFace ───────────────────────────────────────────────────────
        if paradigm == "huggingface":

            # Caso: transformers no disponible
            if SKIP_HF_IF_UNAVAILABLE and not HF_AVAILABLE:
                print("⚠️  transformers no disponible.")
                results.append({
                    "name":            model_name,
                    "paradigm":        paradigm,
                    "success":         False,
                    "error":           "transformers no instalado",
                    "score_100":       0.0,
                    "interpretation":  "Evaluación fallida",
                })
                continue

            ner_pipe = None
            try:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # Cargar el modelo desde HuggingFace Hub
                load_start = time.time()
                ner_pipe   = hf_pipeline(
                    "ner",
                    model=cfg["path"],
                    aggregation_strategy="simple",
                    device=DEVICE,
                )
                load_time  = time.time() - load_start

                # Función de predicción que usa el pipeline HF
                def _hf_predict(text: str, _pipe=ner_pipe) -> List[Dict]:
                    """Ejecuta el pipeline NER y normaliza las salidas."""
                    raw_out  = _pipe(text)
                    preds    = []
                    for entity in raw_out:
                        start = int(entity.get("start", -1))
                        end   = int(entity.get("end",   -1))
                        if start < 0 or end <= start:
                            continue
                        label = canonicalize(
                            entity.get("entity_group",
                                       entity.get("entity", "unknown"))
                        )
                        preds.append({
                            "text":  text[start:end],
                            "label": label,
                            "start": start,
                            "end":   end,
                            "score": float(entity.get("score", 0.0)),
                        })
                    return preds

                # Evaluar sobre el corpus completo
                t0         = time.time()
                evaluation = aggregate_predictions(pairs, _hf_predict)
                latency    = time.time() - t0

                # Clases soportadas por el modelo
                hf_sup = _get_hf_supported_classes(ner_pipe)
                if not hf_sup:
                    # Fallback: usar las clases observadas en las predicciones
                    hf_sup = {
                        canonicalize(e.get("entity_group",
                                           e.get("entity", "")))
                        for text, _ in pairs
                        for e in ner_pipe(text)
                        if e.get("entity_group") or e.get("entity")
                    } & DOMAIN_CLASSES

                result = compute_score(
                    evaluation, cfg, load_time, latency,
                    hf_sup, "model_id2label",
                    model_name, paradigm,
                )
                print(f"Score {result['score_100']:.2f}")
                results.append(result)

            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "name":           model_name,
                    "paradigm":       paradigm,
                    "success":        False,
                    "error":          str(exc),
                    "score_100":      0.0,
                    "interpretation": "Evaluación fallida",
                })
            finally:
                # Liberar memoria del modelo de GPU/CPU
                if ner_pipe:
                    del ner_pipe
                gc.collect()

        # ── Presidio Baseline ─────────────────────────────────────────────────
        elif paradigm == "presidio_baseline":

            if not PRESIDIO_AVAILABLE:
                print("⚠️  presidio-analyzer no disponible.")
                results.append({
                    "name":           model_name,
                    "paradigm":       paradigm,
                    "success":        False,
                    "error":          "presidio-analyzer no instalado",
                    "score_100":      0.0,
                    "interpretation": "Evaluación fallida",
                })
                continue

            try:
                # Construir motor Presidio (sin descarga de modelos)
                load_start = time.time()
                analyzer   = _build_blank_presidio_engine()
                load_time  = time.time() - load_start

                # Función de predicción que usa Presidio
                def _presidio_predict_fn(
                    text: str, _analyzer=analyzer
                ) -> List[Dict]:
                    return _presidio_predict(text, _analyzer)

                # Evaluar sobre el corpus completo
                t0         = time.time()
                evaluation = aggregate_predictions(pairs, _presidio_predict_fn)
                latency    = time.time() - t0

                result = compute_score(
                    evaluation, cfg, load_time, latency,
                    _PRESIDIO_SUPPORTED, "presidio_standard_recognizers",
                    model_name, paradigm,
                )
                print(f"Score {result['score_100']:.2f}")
                results.append(result)

            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "name":           model_name,
                    "paradigm":       paradigm,
                    "success":        False,
                    "error":          str(exc),
                    "score_100":      0.0,
                    "interpretation": "Evaluación fallida",
                })

    # Ordenar por score descendente para el ranking
    results.sort(key=lambda r: r.get("score_100", 0.0), reverse=True)
    return results


# =============================================================================
#  SECCIÓN 14 — Impresión de resultados en consola
#
#  Muestra el ranking final y un resumen por modelo.
#  Formato tabular diseñado para lectura rápida en experimento sandbox.
# =============================================================================

def _pct(value: float) -> str:
    """Formatea un valor [0,1] como porcentaje con 2 decimales."""
    return f"{100 * value:6.2f}%"


def print_results(results: List[Dict]) -> None:
    """Imprime el ranking y el detalle de cada modelo evaluado."""

    # ── Ranking global ────────────────────────────────────────────────────────
    print("\n\n" + "═" * 125)
    print("🏁  RANKING FINAL  —  Domain-Adjusted Privacy Detection Score (v3, sin InfraScore)")
    print("    Fórmula: Score = DetectionScore × Coverage × DomainFit × RegulatoryFit × Penalty_FN")
    print("═" * 125)

    header = f"{'#':>4}  {'Modelo':50}  {'Paradigma':20}  {'Score /100':>10}  Clasificación"
    print(header)
    print("-" * 125)

    icons = {"huggingface": "🤗", "presidio_baseline": "🛡️"}

    for rank, r in enumerate(results, 1):
        icon  = icons.get(r.get("paradigm", "?"), "❓")
        score = f"{r.get('score_100', 0.0):.2f}"
        interp = r.get("interpretation", "—")
        print(f"{rank:4d}  {r['name'][:50]:50}  "
              f"{icon} {r.get('paradigm','?'):18}  {score:>10}  {interp}")

    print("═" * 125)

    # ── Detalle por modelo ────────────────────────────────────────────────────
    for r in results:
        print(f"\n\n{'━' * 100}")

        if not r.get("success"):
            print(f"❌  {r['name']}  [{r.get('paradigm', '?')}]")
            print(f"   Error: {r.get('error', '—')}")
            continue

        micro = r["evaluation"]["micro_metrics"]
        icon  = icons.get(r["paradigm"], "")

        print(f"{icon}  MODELO: {r['name']}  [{r['paradigm']}]")
        print(f"🏆  Score final: {r['score_100']:.2f} / 100   →   {r['interpretation']}")

        # Componentes del score
        print(f"\n  {'─' * 50}")
        print(f"  Componentes del score (v3, sin InfraScore):")
        print(f"    DetectionScore  : {r['detection_score']:.4f}")
        print(f"    Coverage        : {r['coverage']:.4f}  ({r.get('coverage_basis', '?')})")
        print(f"    DomainFit       : {r['domain_fit']:.4f}")
        print(f"    RegulatoryFit   : {r['regulatory_fit']:.4f}")
        print(f"    Penalty_FN      : {r['penalty_fn']:.4f}  "
              f"(CriticalFN: {r['critical_fn']:.4f})")

        # Índices de riesgo
        print(f"\n  Índices de riesgo:")
        print(f"    R1 (clase más débil) : {r['r1_weakest_class_risk_100']:.2f}%"
              f"  [{r.get('r1_weakest_class', '—')}]")
        print(f"    R2 (sistémico)       : {r['r2_systemic_risk_100']:.2f}%")
        print(f"    R_final              : {r['r_final_100']:.2f}%")

        # Métricas micro
        print(f"\n  Métricas micro (corpus):")
        print(f"    TP: {micro['tp']}  FP: {micro['fp']}  FN: {micro['fn']}")
        print(f"    Precision: {_pct(micro['precision'])}  "
              f"Recall: {_pct(micro['recall'])}  "
              f"F2: {_pct(micro['f2'])}")
        print(f"    GT total: {r['ground_truth_count']}  "
              f"Predicciones dominio: {r['prediction_count']}  "
              f"Filtradas fuera dominio: {r['out_of_domain_filtered']}")

        # Tiempos
        print(f"\n  Tiempos:")
        print(f"    Carga del modelo : {r['load_time']:.4f}s")
        print(f"    Latencia total   : {r['inference_latency']:.4f}s")

        # Tabla por clase del dominio
        print(f"\n  {'Clase PII':28} {'TP':>4} {'FP':>4} {'FN':>4}"
              f"  {'P':>8}  {'R':>8}  {'F2':>8}  {'FN rate':>8}")
        print(f"  {'─' * 90}")
        for label in DOMAIN_TAXONOMY:
            m = r["evaluation"]["metrics_by_label"].get(label, {})
            if not m:
                continue
            print(
                f"  {label:28}"
                f" {m['tp']:4d} {m['fp']:4d} {m['fn']:4d}"
                f"  {_pct(m['precision']):>8}"
                f"  {_pct(m['recall']):>8}"
                f"  {_pct(m['f2']):>8}"
                f"  {_pct(m['fn_rate']):>8}"
            )


# =============================================================================
#  SECCIÓN 15 — Serialización JSON
#
#  El resultado se guarda como JSON para ser leído por panel_chatbot_v3.py.
#  Python sets y otros tipos no serializables se convierten a listas/strings.
# =============================================================================

def _to_json_serializable(obj: Any) -> Any:
    """Convierte recursivamente tipos no serializables a JSON."""
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_results_json(results: List[Dict], path: str) -> None:
    """Guarda los resultados en un archivo JSON con indentación legible."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(results), f,
                  ensure_ascii=False, indent=4)
    print(f"\n💾  Resultados guardados en: {path}")


# =============================================================================
#  SECCIÓN 16 — Función principal (entry point)
#
#  Orquesta el flujo completo del experimento:
#    1. Parsear archivos de corpus
#    2. Evaluar todos los modelos
#    3. Mostrar resultados en consola
#    4. Guardar JSON para el panel HTML
# =============================================================================

def main() -> None:
    """Punto de entrada del experimento sandbox v3."""

    print("=" * 70)
    print("  Analisis_Score_Chatbot_v3.py")
    print("  Domain-Adjusted Privacy Detection Score — Sandbox")
    print("  Sin capas REGEX/NER  |  Sin InfraScore")
    print("=" * 70)
    print(f"\n  Corpus tageado  (GT)         : {TAGGED_FILE}")
    print(f"  Corpus sin tagear (model in) : {UNTAGGED_FILE}")
    print(f"  IoU threshold               : {IOU_THRESHOLD}")
    print(f"  Modelos a evaluar           : {len(ALL_MODELS)}")

    # ── Paso 1: Parsear archivos ──────────────────────────────────────────────
    print("\n📂  Leyendo corpus …")
    tagged_records  = parse_tagged_file(TAGGED_FILE)
    untagged_texts  = parse_untagged_file(UNTAGGED_FILE)
    pairs           = align_files(tagged_records, untagged_texts)

    # Estadísticas del corpus
    total_gt  = sum(len(sp) for _, sp in pairs)
    with_ents = sum(1 for _, sp in pairs if sp)

    print(f"  Turnos totales        : {len(pairs)}")
    print(f"  Turnos con entidades  : {with_ents}")
    print(f"  Entidades GT totales  : {total_gt}")

    # Distribución por tipo de entidad en el ground truth
    dist: Dict[str, int] = defaultdict(int)
    for _, spans in pairs:
        for sp in spans:
            dist[sp["label"]] += 1

    print("\n  Distribución del ground truth:")
    for label in DOMAIN_TAXONOMY:
        count = dist.get(label, 0)
        bar   = "█" * (count // 2)
        print(f"    {label:25s}  {count:4d}  {bar}")

    # ── Paso 2: Evaluar modelos ───────────────────────────────────────────────
    print("\n🏃  Evaluando modelos …")
    results = evaluate_all_models(pairs)

    # ── Paso 3: Mostrar resultados ────────────────────────────────────────────
    print_results(results)

    # ── Paso 4: Guardar JSON ──────────────────────────────────────────────────
    if SAVE_RESULTS_JSON:
        save_results_json(results, OUTPUT_JSON_PATH)
        print(f"\n▶  Para generar el panel HTML:")
        print(f"   python panel_chatbot_v3.py {OUTPUT_JSON_PATH} --open")


# =============================================================================
#  Ejecución directa
# =============================================================================

if __name__ == "__main__":
    main()
