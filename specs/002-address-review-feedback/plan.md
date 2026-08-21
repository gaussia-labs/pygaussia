# Plan: Address PR Review Feedback — PR #3

**Branch:** `fix/pygaussia`
**PR:** gaussia-labs/pygaussia#3
**Reviewer:** alexFiorenza
**Review verdict:** Changes requested

---

## Problema

El PR fue rechazado por dos razones:

1. Los `# type: ignore` no son una solución — son una forma de callarle la boca a mypy sin arreglar nada.
2. Los problemas de Guardian Models no están resueltos en términos de tipos.

---

## Causa raíz

Había tres categorías de errores tapados con `# type: ignore`:

**Categoría A — Librerías sin stubs de tipos**
`torch`, `langchain` y `scipy` no estaban declaradas en el bloque `[[tool.mypy.overrides]]` de `pyproject.toml`. Ese bloque ya existe y ya tiene `transformers`, `nltk`, etc. Solo faltaba agregar las que faltaban.

**Categoría B — Tipo declarado incorrecto en código propio**
`LLMGuardianProvider.__init__` declaraba `tokenizer: AutoTokenizer`. Eso está mal — `AutoTokenizer` es una fábrica (crea tokenizers), no el tokenizer en sí. El tipo real del objeto que devuelve es `PreTrainedTokenizerBase`. Ese error de anotación generaba una cascada de `arg-type` en todos los lugares que le pasaban un tokenizer.

**Categoría C — mypy no puede garantizar que un objeto no es `None` después de la inicialización lazy**
`QwenEmbedder` y `QwenReranker` cargan el modelo solo cuando lo necesitan. mypy ve que el campo puede ser `None` y no puede garantizar que después de la asignación ya no lo es. La solución es `assert is not None`, que además actúa como guardia de runtime.

**Categoría D — `json.loads()` devuelve `Any`**
En `judge.py`, `json.loads()` devuelve `Any` y mypy se queja cuando eso se usa como `dict`. La solución es un `assert isinstance(result, dict)` que es correcto en contexto (la regex garantiza que el JSON es un objeto).

---

## Alcance de cambios

### Lo que ya está correcto — no tocar

- `schemas/bias.py`: `overrides` usa `Field(default_factory=dict)`.
- `prompt_optimizer/schemas.py`: `history`, `demos`, `trials` usan `Field(default_factory=list)`.
- `providers.py`: null guard en `_parse_guardian_response`.
- `providers.py` `infer()`: usa `next(model.parameters()).device`.
- `providers.py` `HuggingFaceGuardianProvider.__init__`: pasa `AutoTokenizer.from_pretrained(model)` a super.
- `generators.py` y `judge.py` (dos de tres ignores): se van solos con el fix de `pyproject.toml`.

---

## Fixes en detalle

### Fix 1 — `pyproject.toml`: completar el bloque de librerías sin stubs

El bloque ya existe. Solo se extiende la lista:

```toml
[[tool.mypy.overrides]]
module = [
    "nltk.*",
    "transformers.*",
    "sentence_transformers.*",
    "hdbscan.*",
    "umap.*",
    "interpreto.*",
    "torch.*",
    "langchain.*",
    "langchain_core.*",
    "langchain_community.*",
    "scipy.*",
]
ignore_missing_imports = true
```

Esto elimina todos los `# type: ignore[import-not-found]` e `# type: ignore[import-untyped]` del código. Sin tocar ningún archivo de lógica.

**Efecto colateral positivo:** los dos `# type: ignore` de `generators.py` y dos de `judge.py` desaparecen solos porque venían de langchain.

---

### Fix 2 — `schemas/bias.py`: corregir el tipo del tokenizer

```python
# Antes
from transformers import AutoTokenizer

class LLMGuardianProvider(ABC):
    def __init__(self, ..., tokenizer: AutoTokenizer, ...):

# Después
from transformers import PreTrainedTokenizerBase

class LLMGuardianProvider(ABC):
    def __init__(self, ..., tokenizer: PreTrainedTokenizerBase, ...):
```

Con esto desaparecen los `# type: ignore[arg-type]` en `HuggingFaceGuardianProvider.__init__`, `IBMGranite.__init__`, y `LLamaGuard.__init__`. También desaparece el `# type: ignore[attr-defined]` en `IBMGranite.is_biased` porque `PreTrainedTokenizerBase` tiene `apply_chat_template` definido.

---

### Fix 3 — `embedders/qwen.py`: tipos correctos + assert

Cambiar las anotaciones de los campos privados y propiedades, y reemplazar los `# type: ignore` con `assert`:

```python
# Antes
_tokenizer: AutoTokenizer | None
_model: AutoModel | None

@property
def tokenizer(self) -> AutoTokenizer:
    if self._tokenizer is None:
        self._tokenizer = AutoTokenizer.from_pretrained(...)  # type: ignore[assignment]
    return self._tokenizer  # type: ignore[return-value]

@property
def model(self) -> AutoModel:
    if self._model is None:
        self._model = AutoModel.from_pretrained(...)
        self._model.eval()  # type: ignore[union-attr]
    return self._model

def _encode_batch(self, texts):
    batch = self.tokenizer(...)  # type: ignore[operator]
    batch = {k: v.to(self.model.device) ...}  # type: ignore[attr-defined]
    ...
    return embeddings.cpu().numpy()  # type: ignore[no-any-return]

# Después
_tokenizer: PreTrainedTokenizerBase | None
_model: PreTrainedModel | None

@property
def tokenizer(self) -> PreTrainedTokenizerBase:
    if self._tokenizer is None:
        self._tokenizer = AutoTokenizer.from_pretrained(...)
    assert self._tokenizer is not None
    return self._tokenizer

@property
def model(self) -> PreTrainedModel:
    if self._model is None:
        self._model = AutoModel.from_pretrained(...)
        self._model.eval()
    assert self._model is not None
    return self._model

def _encode_batch(self, texts):
    # Con transformers en ignore_missing_imports, tokenizer y model son Any
    # Sus llamadas no generan errores. No se necesita nada extra.
    ...
```

Con `transformers.*` ya en `ignore_missing_imports`, `PreTrainedTokenizerBase` y `PreTrainedModel` son tratados como `Any` por mypy, lo que significa que todas sus llamadas (`.device`, `__call__`, `.eval()`, etc.) se aceptan sin errores y sin `# type: ignore`.

---

### Fix 4 — `rerankers/qwen.py`: mismo patrón que embedder

Mismo cambio: `AutoTokenizer | None` → `PreTrainedTokenizerBase | None`, `AutoModelForCausalLM | None` (este ya tiene mejor tipado que `AutoModel` en transformers, puede quedarse), agregar `assert is not None` en las propiedades.

---

### Fix 5 — `llm/judge.py`: el único ignore que queda

```python
# Antes
return json.loads(match.group(1).strip())  # type: ignore[no-any-return]

# Después
result = json.loads(match.group(1).strip())
assert isinstance(result, dict)
return result
```

El `assert` es correcto: la regex que llega hasta acá busca `{...}`, garantizando que el JSON es un objeto. Si alguna vez no lo fuera, el assert lo detecta en runtime con un mensaje claro.

---

## Tests a escribir (TDD: escribir antes de implementar)

Los tests verifican que los fixes funcionan en runtime, no solo que mypy pasa.

### `tests/guardians/test_huggingface_provider.py` (ampliar)

Ya existen tests de init y device. Agregar:

- `_parse_output` con mock de `output.scores` y `output.sequences` — verifica que `is_bias` y `prob_of_bias` se calculan correctamente.
- `_parse_output` lanza `ValueError` cuando `prob_of_bias` no se puede calcular (scores vacíos).
- `_get_probabilities` con logprobs mock — verifica que safe + unsafe suman a ~1.

### `tests/guardians/test_ibm_granite.py` (nuevo)

- `is_biased` con `provider.infer` mockeado → construye `GuardianBias` con el atributo correcto.
- Los mensajes enviados al tokenizer tienen la estructura `{"role": "user", ...}` esperada.
- El `guardian_config` incluye `risk_name` y `risk_definition`.

### `tests/guardians/test_llama_guard.py` (nuevo)

- `is_biased` con `provider.infer` mockeado → construye `GuardianBias` correctamente.
- `categories` se construye combinando `attribute.value` y `attribute.description`.

### `tests/embedders/__init__.py` + `tests/embedders/test_qwen_embedder.py` (nuevo)

- Lazy init del tokenizer: no se llama `from_pretrained` al construir, se llama en el primer acceso, y se cachea (segundo acceso no vuelve a llamarlo).
- Lazy init del model: mismo comportamiento.
- `_last_token_pool` con left padding: devuelve el último token.
- `_last_token_pool` con right padding: devuelve el token en la posición correcta según `attention_mask`.
- `encode()` con tokenizer y model mockeados: verifica que se hace vstack de múltiples batches.
- `encode()` con lista vacía: devuelve array con shape `(0, 0)`.

### `tests/rerankers/__init__.py` + `tests/rerankers/test_qwen_reranker.py` (nuevo)

- Lazy init del tokenizer y model.
- `score()` con tokenizer y model mockeados: devuelve una lista de floats de la misma longitud que `documents`.
- `_format_pair` construye el string con los delimitadores correctos.

---

## Estrategia de commits

Dos commits separados para mantener historial limpio:

**Commit 1** — `fix(types): complete mypy overrides and remove type suppressions`
Archivos: `pyproject.toml`, `src/gaussia/schemas/bias.py`, `src/gaussia/guardians/__init__.py`, `src/gaussia/guardians/llms/providers.py`, `src/gaussia/embedders/qwen.py`, `src/gaussia/rerankers/qwen.py`, `src/gaussia/llm/judge.py`

**Commit 2** — `test(guardians): add runtime coverage for HuggingFace provider, IBMGranite, LLamaGuard, QwenEmbedder, QwenReranker`
Archivos: todos los archivos de test nuevos o ampliados.

---

## Verificación final

```bash
uv run mypy src/gaussia        # Debe mostrar: Success: no issues found in 88 source files
uv run pytest -m "not slow"   # Debe pasar (ignorar test_humanity.py y test_toxicity.py localmente)
uv run ruff check .            # Sin errores
```

---

## Lo que este plan NO cambia

- Ningún comportamiento en runtime — todos los cambios son de anotaciones de tipo.
- Ningún schema — los modelos Pydantic ya están correctos.
- Ninguna abstracción nueva — no hay clases ni módulos nuevos de lógica.
