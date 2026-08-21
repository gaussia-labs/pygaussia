# Plan: Refactor `Judge` — Add Logprob-Based Scoring Path (PR A)

**Branch a crear:** `refactor/judge-logprobs` (desde `develop`)
**PR objetivo:** abrir contra `develop` con título `refactor(judge): add logprob-based scoring path`
**Origen del trabajo:** [gaussia-labs/pygaussia#5](https://github.com/gaussia-labs/pygaussia/pull/5) — review CHANGES_REQUESTED de `alexFiorenza`
**Reviewer del PR a abrir:** `alexFiorenza`
**Idioma de los textos en el PR:** español neutro (estilo del usuario), código y comments en inglés

> **Para la sesión que ejecute este plan:** asumí cero contexto de la conversación previa. Todo lo que necesitás está acá adentro o referenciado por path absoluto. Si dudás de una decisión, **no la cambies** — están todas confirmadas con el usuario. Si encontrás una contradicción entre este doc y el código actual, gana **este doc** (porque el código actual es lo que hay que refactorizar).

---

## 1. Contexto

### Por qué este PR existe

El PR #5 introdujo la métrica `RoleAdherence` con una clase `LLMJudgeStrategy` local en `src/gaussia/metrics/role_adherence.py`. Alex (reviewer) pidió dos cosas en comentarios inline:

1. **Comentario inline #1** (sobre la clase `LLMJudgeStrategy`):
   > "This might be better to be inside the Judge class. We talked about using logprobs instead of asking the judge to return a score. We might be able to use this new approach but in the Judge base class, so the other metrics that use llm as a judge can now use logprobs instead of asking the judge to return a score."

2. **Comentario inline #2** (sobre el método `score()`):
   > "In addition to that, we must have a check if the passed provider supports the use of logprobs. In case that key is not available simply raise an exception in the code."

### Cómo se decidió encarar

Se descartó hacer todo en un solo PR. Va a haber **dos PRs apilados**:

- **PR A (este plan):** agregar capability de logprob scoring al `Judge` base. **No migra ninguna métrica existente.** Solo expone el método nuevo.
- **PR B (futuro, no es este plan):** rebasear `feature/role-adherence-metric` sobre PR A, eliminar `LLMJudgeStrategy` local, y hacer que `RoleAdherence` consuma el nuevo método de `Judge`. Docs y aws-lambda example ya están en PR #5 y se conservan.

### Decisiones tomadas (no re-litigar)

| Decisión | Valor | Razón |
|----------|-------|-------|
| Fallback cuando provider no soporta logprobs | **Raise excepción siempre**, sin fallback automático | Alex lo pidió explícito, y el paper no propone fallback. |
| Métricas a migrar en PR A | **Ninguna.** Solo se agrega el método; nadie lo consume todavía. | Mantener el PR chico y revisable. PR B migra RoleAdherence. |
| Otras métricas (Context, Conversational, BestOf, Agentic) | **No se tocan en PR A ni PR B.** | Logprob no encaja con multi-dimensión ni outputs de texto. Path actual queda intacto. |
| API pública del constructor `Judge` | **No cambia.** | Solo se agrega un método nuevo. |
| Path actual `Judge.check()` / `_check_structured()` / `_check_regex()` | **No se toca.** | Cualquier cambio rompe 5 métricas y ~73 tests. |
| Tests existentes (`TestJudge`, 19 tests) | **No se modifican.** | Solo se agrega `TestJudgeLogprob` con tests nuevos. |
| Estructura del PR #5 actual | El usuario decidirá si lo cierra o lo convierte en draft cuando PR A esté abierto. **No tocar PR #5 desde PR A.** | Separación de scope. |

---

## 2. Estado actual del código (lo que vas a leer antes de tocar nada)

### Files clave (con resumen de qué contienen)

| Path | Qué hay hoy | Qué hay que hacer |
|------|-------------|-------------------|
| `src/gaussia/llm/judge.py` | Clase `Judge` con constructor + `check()` + `_check_structured()` + `_check_regex()` + `_extract_json()` + helpers. 201 líneas. | **Agregar** `check_logprob_binary()` + `_supports_logprobs()` + `_aggregate_logprobs()` y constantes de registry. **No modificar** lo existente. |
| `src/gaussia/llm/__init__.py` | Exporta `Judge` y schemas. | **Agregar** exports de las excepciones nuevas si seguís el patrón actual (revisar; quizás solo van expuestas desde `gaussia.core.exceptions`). |
| `src/gaussia/core/exceptions.py` | `GaussiaError` base + 5 subclases (Retriever/Metric/Guardian/Loader/StatisticalMode). 26 líneas. | **Agregar** `LogprobsNotSupportedError` y `LogprobsExtractionError`, ambas heredando de `GaussiaError`. |
| `tests/llm/test_judge.py` | 19 tests en `TestJudge` cubriendo el path actual con mocks `MagicMock`. | **Agregar** clase nueva `TestJudgeLogprob` con tests del path nuevo. **No tocar** `TestJudge`. |

### Snippets del estado actual relevantes

**Excepciones (estado actual completo, `src/gaussia/core/exceptions.py`):**

```python
"""Custom exceptions for Gaussia."""


class GaussiaError(Exception):
    """Base exception for Gaussia."""


class RetrieverError(GaussiaError):
    """Exception raised when a retriever fails to load data."""


class MetricError(GaussiaError):
    """Exception raised when a metric calculation fails."""


class GuardianError(GaussiaError):
    """Exception raised when a guardian fails to detect bias."""


class LoaderError(GaussiaError):
    """Exception raised when a loader fails to load data."""


class StatisticalModeError(GaussiaError):
    """Exception raised when a statistical mode calculation fails."""
```

**Constructor de `Judge` (estado actual, `src/gaussia/llm/judge.py:36-52`):**

```python
def __init__(
    self,
    model: BaseChatModel,
    use_structured_output: bool = False,
    strict: bool = True,
    bos_json_clause: str = "```json",
    eos_json_clause: str = "```",
    verbose: bool = False,
):
    self.model = model
    self.use_structured_output = use_structured_output
    self.strict = strict
    self.bos_json_clause = bos_json_clause
    self.eos_json_clause = eos_json_clause
    self.verbose = verbose
    self.chat_history: list[tuple[str, str]] = []
    self.logger = VerboseLogger(verbose=verbose)
```

**Helper a reutilizar (`src/gaussia/llm/judge.py:85-86`):**

```python
def _render_system_prompt(self, system_prompt: str, data: dict) -> str:
    return system_prompt.format_map(data)
```

---

## 3. Especificación del método nuevo (extraída del paper)

> Fuente: `papers/role-adherence/logprob_judge_proposal.md` en branch `feat/role_adherence` del repo `gaussia-papers` (path local: `/Users/frino/Desktop/Alquimia/Gaussia/gaussia-papers`). Commit relevante: `801f2da feat(role-adherence): integrate logprob continuous mode and revise paper structure`.

### Fórmula

Dado un prompt que pide al modelo responder solo con `YES` o `NO`:

```
score = P(YES) / (P(YES) + P(NO))
```

Equivalente numéricamente estable:

```
score = sigmoid(log_p_yes - log_p_no) = 1 / (1 + exp(log_p_no - log_p_yes))
```

donde `log_p_yes` y `log_p_no` se obtienen aplicando **log-sum-exp** sobre las logprobs de **todas las variantes de tokenización** del término (mayúsculas, minúsculas, con/sin espacio inicial).

### Parámetros de inferencia (no negociables)

- `logprobs=True`
- `top_logprobs=10` (el paper usa exactamente 10)
- `temperature=1.0` (el paper usa exactamente 1)
- System prompt: mismo formato que el binary judge — debe instruir explícitamente "answer ONLY with YES or NO".

### Por qué `temperature=1` y no `0`

El binary judge clásico usa temp=0 (greedy decoding). Para logprobs el paper usa temp=1 para que la distribución de probabilidades del primer token refleje la incertidumbre real del modelo. Si hacés temp=0 los logprobs siguen estando, pero la distribución queda colapsada y perdés señal.

### Fenómeno "Okay" (de Gemma)

El paper documenta que **Gemma 3-12B asigna ~99% de probabilidad al token `"Okay"`** como primer token de respuesta (es un sesgo de su instruction tuning — abre la respuesta con un acknowledgment). YES y NO aparecen en el top10 con probabilidades bajas (~0.5% y ~0.001%), pero **la relación P(YES)/P(NO) sigue siendo discriminativa** (AUC=0.999). Por eso `top_logprobs=10` y no algo más chico. **No filtrar el token top1**: leer todos los top_logprobs y agregar los que matcheen YES o NO.

### Caso de error: ni YES ni NO aparecen en top10

Si después de filtrar no aparece **ninguna** variante de YES y ninguna de NO en el top10, levantar `LogprobsExtractionError`. Es indicador de que el modelo no entendió el prompt o que `top_logprobs=10` es insuficiente para este modelo.

Si aparece solo uno de los dos (ej. YES sí, NO no), tratar el ausente como `log_prob = -inf`. El score resultante será 1.0 o 0.0 respectivamente — eso es informativo, no un error.

### Estructura de la respuesta de LangChain

Para `ChatOpenAI` (y compatibles vía OpenAI API — esto incluye HuggingFace TGI con `base_url` override, que es como el paper usó Gemma):

```python
response = model.invoke("...")
# Estructura:
response.response_metadata = {
    "logprobs": {
        "content": [
            {
                "token": "Okay",
                "logprob": -0.01,
                "bytes": [...],
                "top_logprobs": [
                    {"token": "Okay", "logprob": -0.01, "bytes": [...]},
                    {"token": " Sure", "logprob": -5.2, "bytes": [...]},
                    {"token": "YES", "logprob": -5.3, "bytes": [...]},
                    {"token": "NO", "logprob": -13.8, "bytes": [...]},
                    # ... hasta 10 entries
                ],
            },
            # ... más tokens si la respuesta es larga
        ]
    },
    # ... otras keys
}
```

**Solo importa `response.response_metadata["logprobs"]["content"][0]["top_logprobs"]`** — los top_logprobs del primer token generado. El resto se ignora.

Para `ChatOllama` la estructura es equivalente (LangChain normaliza). Verificar empíricamente si surgen dudas.

---

## 4. Cambios a realizar (file by file)

### 4.1 `src/gaussia/core/exceptions.py`

**Acción:** agregar al final del archivo:

```python
class LogprobsNotSupportedError(GaussiaError):
    """Raised when the configured LLM provider does not expose logprobs."""


class LogprobsExtractionError(GaussiaError):
    """Raised when expected tokens are absent from the model's top_logprobs."""
```

### 4.2 `src/gaussia/llm/judge.py`

**Acción 1 — agregar imports en el bloque de imports existente (líneas 1-14):**

```python
import math  # ya puede estar; si no, agregar

from gaussia.core.exceptions import LogprobsExtractionError, LogprobsNotSupportedError
```

**Acción 2 — agregar constantes a nivel de módulo, justo después de los imports y antes de la línea 16 (`T = TypeVar(...)`):**

```python
_LOGPROB_CAPABLE_PROVIDERS: frozenset[str] = frozenset({
    "ChatOpenAI",
    "AzureChatOpenAI",
    "BaseChatOpenAI",
    "ChatOllama",
    "ChatLiteLLM",
})

_LOGPROB_INCAPABLE_PROVIDERS: frozenset[str] = frozenset({
    "ChatAnthropic",
    "ChatGoogleGenerativeAI",
    "ChatBedrock",
    "ChatBedrockConverse",
})

_DEFAULT_POSITIVE_TOKENS: tuple[str, ...] = ("YES", "Yes", "yes", " YES", " Yes", " yes")
_DEFAULT_NEGATIVE_TOKENS: tuple[str, ...] = ("NO", "No", "no", " NO", " No", " no")
```

**Acción 3 — agregar método público y helpers al final de la clase `Judge` (después de `_extract_json` que termina en línea 200):**

```python
def check_logprob_binary(
    self,
    system_prompt: str,
    query: str,
    data: dict,
    positive_tokens: tuple[str, ...] = _DEFAULT_POSITIVE_TOKENS,
    negative_tokens: tuple[str, ...] = _DEFAULT_NEGATIVE_TOKENS,
    top_logprobs: int = 10,
) -> tuple[float, dict]:
    """Score a binary YES/NO judgment via first-token logprobs.

    Renders the system prompt, invokes the model with logprobs enabled at
    temperature=1.0, and returns P(positive) / (P(positive) + P(negative))
    aggregated across surface-form variants using log-sum-exp.

    Raises:
        LogprobsNotSupportedError: if self.model is a provider known not to
            expose logprobs (e.g., ChatAnthropic).
        LogprobsExtractionError: if neither positive nor negative tokens
            appear in the top_logprobs of the first generated token.

    Returns:
        Tuple (score, raw_top_logprobs) where score is in [0, 1] and
        raw_top_logprobs is the unfiltered list of {token, logprob} dicts
        from the first token position (useful for debugging / logging).
    """
    if not self._supports_logprobs(self.model):
        raise LogprobsNotSupportedError(
            f"Provider {type(self.model).__name__} does not support logprobs. "
            f"Use a provider in {sorted(_LOGPROB_CAPABLE_PROVIDERS)} or call "
            f"Judge.check() instead."
        )

    rendered_system = self._render_system_prompt(system_prompt, data)
    bound = self.model.bind(
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=1.0,
    )
    response = bound.invoke([
        ("system", rendered_system),
        ("human", query),
    ])

    top_lp = (
        response.response_metadata
        .get("logprobs", {})
        .get("content", [{}])[0]
        .get("top_logprobs", [])
    )

    log_p_pos = self._aggregate_logprobs(top_lp, positive_tokens)
    log_p_neg = self._aggregate_logprobs(top_lp, negative_tokens)

    if log_p_pos == -math.inf and log_p_neg == -math.inf:
        raise LogprobsExtractionError(
            f"Neither positive {positive_tokens} nor negative {negative_tokens} "
            f"tokens appeared in top_{top_logprobs}. Observed tokens: "
            f"{[entry.get('token') for entry in top_lp]}"
        )

    score = 1.0 / (1.0 + math.exp(log_p_neg - log_p_pos))
    return score, {"top_logprobs": top_lp}

@staticmethod
def _supports_logprobs(model: BaseChatModel) -> bool:
    name = type(model).__name__
    if name in _LOGPROB_INCAPABLE_PROVIDERS:
        return False
    return name in _LOGPROB_CAPABLE_PROVIDERS

@staticmethod
def _aggregate_logprobs(top_logprobs: list[dict], target_tokens: tuple[str, ...]) -> float:
    matches = [
        entry["logprob"]
        for entry in top_logprobs
        if entry.get("token") in target_tokens
    ]
    if not matches:
        return -math.inf
    max_lp = max(matches)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in matches))
```

**Nota sobre el sigmoid:** la fórmula `1 / (1 + exp(log_p_neg - log_p_pos))` es numéricamente estable salvo cuando `log_p_neg - log_p_pos` es muy grande (> ~700) → `exp` overflow. En la práctica logprobs son negativos y la diferencia rara vez supera 50. Si querés blindarte: usar `math.exp(min(log_p_neg - log_p_pos, 700))` o reemplazar por una sigmoid clamp. **No es prioridad** — el caso degenerado solo ocurre si un token tiene prob ~1.0 y el otro ~0, en cuyo caso el score colapsa a 0 o 1 correctamente.

### 4.3 `src/gaussia/llm/__init__.py`

**Acción:** decidir si exportar las excepciones. Mirar primero el patrón existente:

```python
# Estado actual:
from .judge import Judge
from .schemas import BestOfJudgeOutput, ContextJudgeOutput, ConversationalJudgeOutput, RoleAdherenceJudgeOutput

__all__ = [
    "BestOfJudgeOutput",
    "ContextJudgeOutput",
    "ConversationalJudgeOutput",
    "Judge",
    "RoleAdherenceJudgeOutput",
]
```

Las excepciones de Gaussia (`GaussiaError`, `RetrieverError`, etc.) **NO están re-exportadas** desde otros `__init__.py` — viven en `gaussia.core.exceptions`. Seguir ese patrón: **no exportar `LogprobsNotSupportedError` ni `LogprobsExtractionError` desde `gaussia.llm`.** Quien las quiera usar las importa desde `gaussia.core.exceptions`.

**No tocar `src/gaussia/llm/__init__.py`.**

### 4.4 `tests/llm/test_judge.py`

**Acción:** agregar al final del archivo (después de la clase `TestJudge` existente) una clase nueva `TestJudgeLogprob` con los siguientes tests. **No modificar nada arriba.**

Tests requeridos (todos con `MagicMock`, sin llamar a APIs reales):

1. `test_supports_logprobs_openai` — instancia mock con `type(m).__name__ == "ChatOpenAI"` (vía `MagicMock(spec=ChatOpenAI)` o subclasing `MagicMock` con `__class__` override). Verificar `Judge._supports_logprobs(m) is True`.

2. `test_supports_logprobs_anthropic_returns_false` — mock con `__class__.__name__ == "ChatAnthropic"`. Verificar `False`.

3. `test_supports_logprobs_unknown_returns_false` — mock con name custom. Verificar `False` (default seguro).

4. `test_check_logprob_binary_raises_when_unsupported` — Judge con mock Anthropic. Verificar que `check_logprob_binary("p", "q", {})` levanta `LogprobsNotSupportedError`. Verificar que **no se llama a `model.bind()` ni a `model.invoke()`** (raise antes).

5. `test_check_logprob_binary_extracts_yes_score` — mock OpenAI. `bind()` retorna un mock cuyo `invoke()` retorna un response con esta estructura:
   ```python
   response.response_metadata = {
       "logprobs": {
           "content": [{
               "top_logprobs": [
                   {"token": "Yes", "logprob": -0.1},
                   {"token": "No", "logprob": -2.3},
                   {"token": "Maybe", "logprob": -5.0},
               ]
           }]
       }
   }
   ```
   Score esperado: `1 / (1 + exp(-2.3 - (-0.1)))` = `1 / (1 + exp(-2.2))` ≈ `0.9002`. Tolerancia `abs(score - 0.9002) < 1e-3`.

6. `test_check_logprob_binary_aggregates_variants` — top_logprobs con `"YES"` y `"Yes"` ambos. Verificar que el log-sum-exp se aplica (score más cercano a 1 que si se considerara solo uno).

7. `test_check_logprob_binary_no_tokens_present_raises_extraction_error` — top_logprobs sin YES ni NO. Verificar `LogprobsExtractionError`.

8. `test_check_logprob_binary_one_side_missing` — solo aparece NO en top_logprobs (YES ausente). Score esperado: 0.0 (sin error). Esto verifica que `-inf` en uno solo no rompe.

9. `test_check_logprob_binary_binds_correct_params` — capturar el call a `model.bind()` y verificar que se llamó con `logprobs=True, top_logprobs=10, temperature=1.0`.

10. `test_aggregate_logprobs_empty_returns_neg_inf` — unit test del helper.

11. `test_aggregate_logprobs_logsumexp_correctness` — pasar lista conocida `[{"token": "A", "logprob": -1.0}, {"token": "A", "logprob": -2.0}]` con target `("A",)`. Esperado: `log(exp(-1) + exp(-2))` ≈ `-0.6867`. Tolerancia `1e-4`.

**Patrón de mock para `type(m).__name__`:**

```python
class _FakeChatOpenAI(MagicMock):
    pass
_FakeChatOpenAI.__name__ = "ChatOpenAI"
mock_model = _FakeChatOpenAI()
```

O más limpio con `spec`:

```python
from langchain_openai import ChatOpenAI  # solo si está disponible en el entorno de test
mock_model = MagicMock(spec=ChatOpenAI)
# verificar que type(mock_model).__name__ devuelve "ChatOpenAI" — sino fallback a la técnica anterior
```

Verificar empíricamente cuál funciona; si `langchain_openai` no está en deps de test, usar la técnica de subclasing `MagicMock`.

---

## 5. Risk table

| # | Riesgo | Mitigación |
|---|--------|------------|
| 1 | Provider no listado en registry (LangChain agrega clases nuevas o renombra) | Default a `False` ante desconocido (`_supports_logprobs` retorna `True` solo si está en `_LOGPROB_CAPABLE_PROVIDERS`). El usuario que sepa que su provider soporta logprobs puede subclasear `Judge` o monkeypatch el frozenset. |
| 2 | `model.bind(temperature=1.0)` override silencioso de temperature ya bindeada | Aceptable. Documentar en docstring que el método siempre invoca a `temp=1` para preservar la distribución, sin importar la config del model. |
| 3 | top_logprobs=10 insuficiente para algún modelo (ni YES ni NO en top10) | Raise `LogprobsExtractionError` con detalle de tokens observados — error accionable para el usuario. Parámetro `top_logprobs` configurable para que pueda subirlo. |
| 4 | Listas default de tokens no cubren un caso edge (ej. modelo emite `"yEs"`) | Aceptar `positive_tokens`/`negative_tokens` como parámetros con default sensato. Caller customiza. |
| 5 | Overflow en `math.exp(log_p_neg - log_p_pos)` con diferencias absurdamente grandes (>700) | En la práctica logprobs negativos cerca de 0 hacen que la diff rara vez supere 50. Si pasa, levantar `OverflowError` natural. No agregar guarda preventiva — YAGNI. |
| 6 | Estructura de `response_metadata["logprobs"]["content"]` cambia entre versiones de LangChain | El `pyproject.toml` ya tiene LangChain pinneado. Si cambia en upgrade, tests fallan rápido. No es problema de hoy. |
| 7 | Tests existentes se rompen por side-effect | Path actual no se toca. Correr `uv run pytest tests/llm/test_judge.py` antes y después; los 19 tests de `TestJudge` deben pasar idénticos. |
| 8 | Modelo demasiado chico (Llama 8B) genera señal invertida (AUC=0.303 según paper) | No es bug, es capacidad del modelo. Documentar en el docstring del método: "Requires a sufficiently capable model — see paper for AUC results by model size." No es responsabilidad del SDK detectar esto. |
| 9 | `BaseChatOpenAI` (subclase usada por HuggingFace TGI con `base_url`) reporta `__name__` distinto en alguna versión | Incluida explícitamente en el frozenset `_LOGPROB_CAPABLE_PROVIDERS`. Si surge otra clase compatible, agregarla. |

---

## 6. Branching strategy

```
develop
   └── refactor/judge-logprobs        ← PR A (este plan)
          └── feature/role-adherence  ← PR B (siguiente sesión, NO en este PR)
```

**Comandos para arrancar:**

```bash
cd /Users/frino/Desktop/Alquimia/Gaussia/pygaussia
git fetch origin
git checkout develop
git pull origin develop
git checkout -b refactor/judge-logprobs
```

**Antes de empezar:** confirmar con el usuario que va a hacer la fast-mode toggle si quiere (lo mencionó en la conversación previa). No es bloqueante.

---

## 7. Verification

### 7.1 Lint y type check

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/gaussia
```

Todos deben pasar **sin agregar `# type: ignore`** (memoria del proyecto: prohibido). Si mypy se queja en el código nuevo:

- `response.response_metadata.get(...)` puede retornar `Any` → usar asserts o cast explícito.
- `model.bind(...)` retorna `Runnable`, no `BaseChatModel` → eso es OK, no asignar a una variable tipada como `BaseChatModel`.
- Ver `specs/002-address-review-feedback/plan.md` (mismo directorio que este) para patrones de cómo resolver mypy correctamente sin `type: ignore`.

### 7.2 Tests

```bash
# Solo el archivo modificado:
uv run pytest tests/llm/test_judge.py -v

# Suite completa (debe seguir pasando los 441 tests existentes):
uv run pytest
```

Métrica de éxito:
- 19 tests `TestJudge` siguen verdes (no se tocaron).
- ~11 tests nuevos en `TestJudgeLogprob` verdes.
- 441+ tests de la suite completa siguen verdes.

### 7.3 Smoke test manual (opcional)

Con `OPENAI_API_KEY` exportada:

```python
from langchain_openai import ChatOpenAI
from gaussia.llm import Judge

model = ChatOpenAI(model="gpt-4o-mini")
judge = Judge(model=model)
score, raw = judge.check_logprob_binary(
    system_prompt="You are an evaluator. Given a question, answer ONLY with YES or NO.",
    query="Is 'hello' a greeting?",
    data={},
)
print(score, raw["top_logprobs"][:5])
assert 0.5 < score <= 1.0, f"Expected high YES score, got {score}"
```

Negative test (debe levantar antes de llamar al modelo):

```python
from langchain_anthropic import ChatAnthropic
from gaussia.llm import Judge
from gaussia.core.exceptions import LogprobsNotSupportedError

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
judge = Judge(model=model)
try:
    judge.check_logprob_binary("Answer YES or NO.", "Is water wet?", {})
    raise AssertionError("Should have raised")
except LogprobsNotSupportedError as e:
    print("OK:", e)
```

---

## 8. Abrir el PR

Cuando todo verde:

```bash
git push -u origin refactor/judge-logprobs

gh pr create --repo gaussia-labs/pygaussia \
  --base develop \
  --title "refactor(judge): add logprob-based scoring path" \
  --body "$(cat <<'EOF'
## Summary

- Agrega `Judge.check_logprob_binary()` siguiendo la fórmula del paper `papers/role-adherence/logprob_judge_proposal.md` (branch `feat/role_adherence` en `gaussia-labs/papers`).
- Agrega registry de providers compatibles (`ChatOpenAI`, `AzureChatOpenAI`, `BaseChatOpenAI`, `ChatOllama`, `ChatLiteLLM`).
- Agrega `LogprobsNotSupportedError` y `LogprobsExtractionError` en `gaussia.core.exceptions`.
- **No modifica** el path existente (`Judge.check()`, `_check_structured()`, `_check_regex()`).
- **No migra** ninguna métrica. La migración de `RoleAdherence` va en un PR siguiente apilado sobre este.

## Decisión: raise vs fallback

Confirmado con maintainer: cuando el provider no soporta logprobs (ej. `ChatAnthropic`), `check_logprob_binary()` levanta `LogprobsNotSupportedError` **antes** de invocar al modelo. No hay fallback automático al path clásico de `check()`.

## Referencia al paper

Fórmula: `score = P(YES) / (P(YES) + P(NO))` con log-sum-exp sobre variantes de tokenización, `temperature=1.0`, `top_logprobs=10`. Ver sección "Method" en `logprob_judge_proposal.md`.

## Test plan

- [ ] `TestJudge` (19 tests existentes) sigue verde sin cambios
- [ ] `TestJudgeLogprob` (tests nuevos) verde
- [ ] `uv run pytest` completo verde (441+ tests)
- [ ] `uv run mypy src/gaussia` clean (sin `type: ignore`)
- [ ] `uv run ruff check .` clean
- [ ] Smoke test manual con `ChatOpenAI` retorna score plausible
- [ ] Negative test con `ChatAnthropic` levanta `LogprobsNotSupportedError`

## Próximo PR (no este)

Tras merge de este, `feature/role-adherence-metric` se rebasea sobre `develop`, se elimina `LLMJudgeStrategy` local y se consume `Judge.check_logprob_binary()`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Solicitar review de `@alexFiorenza`.

---

## 9. Checklist final antes de cerrar la sesión

- [ ] Branch `refactor/judge-logprobs` creada desde `develop` actualizada
- [ ] `src/gaussia/core/exceptions.py` con 2 excepciones nuevas
- [ ] `src/gaussia/llm/judge.py` con método nuevo + helpers + constantes
- [ ] `tests/llm/test_judge.py` con `TestJudgeLogprob`
- [ ] `uv run ruff check .` sin errores
- [ ] `uv run mypy src/gaussia` sin errores (cero `type: ignore`)
- [ ] `uv run pytest` 100% verde
- [ ] Commit con mensaje conventional commits (`refactor(judge): ...`)
- [ ] PR abierto contra `develop`
- [ ] Review solicitado a `@alexFiorenza`

---

## 10. Notas finales para el ejecutor

1. **No agregues docstrings largos.** El proyecto tiene memoria de "minimal comments". Una línea de docstring por método público alcanza. El `check_logprob_binary` puede tener docstring más largo porque es API pública nueva — pero no inventes Args/Returns/Raises si son obvios desde la firma. Ver el `Judge.check()` existente como referencia de tono.

2. **No hagas `# type: ignore`.** Está prohibido en el proyecto (ver memoria `feedback_type_ignores.md`). Si mypy se queja, arreglá la causa raíz con `assert isinstance(...)`, `cast(...)`, o ajustando la anotación.

3. **No toques nada fuera de los 3 archivos listados** (`exceptions.py`, `judge.py`, `test_judge.py`). Cualquier otro cambio es scope creep. Si encontrás un bug en otro archivo, anotalo y reportalo al usuario al final — no lo arregles en este PR.

4. **No abras el PR sin pasar todos los checks locales.** mypy + ruff + pytest deben estar verdes antes de `gh pr create`.

5. **Si tenés dudas sobre la estructura de `response_metadata["logprobs"]` en LangChain** — testealo con un script chico contra OpenAI y verificá la forma exacta antes de escribir los mocks. La estructura documentada arriba es la que dice la docs de LangChain pero puede haber sutilezas por versión.

6. **El PR #5 actual no se toca desde acá.** El usuario decide qué hacer con él (cerrar / draft / dejar abierto) cuando este PR esté mergeado.
