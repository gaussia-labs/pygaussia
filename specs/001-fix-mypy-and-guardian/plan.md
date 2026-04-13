# Plan: Fix mypy errors and Guardian provider overrides
**Branch:** `fix/pygaussia` → `develop`
**Issues:** gaussia-labs/pygaussia#1 · gaussia-labs/pygaussia#2

---

## Phase 0 — Documentation Discovery (DONE)

All relevant files were read before planning. Findings:

### Allowed APIs / confirmed signatures

**`Gaussia.batch()` base signature** (`src/gaussia/core/base.py` lines 66–73):
```python
def batch(self, session_id: str, context: str, assistant_id: str,
          batch: list[Batch], language: str | None): ...
```

**`Gaussia.run()` base signature** (`src/gaussia/core/base.py` lines 92–95):
```python
@classmethod
def run(cls, retriever: type[Retriever], **kwargs) -> list: ...
```

**`StatisticalMode` method return types** (`src/gaussia/statistical/base.py`):
- `distribution_divergence() -> float | dict[str, Any]`
- `rate_estimation() -> float | dict[str, Any]`
- `aggregate_metrics() -> float | dict[str, Any]`
- `dispersion_metric() -> float | dict[str, Any]`

**`GuardianLLMConfig`** (`src/gaussia/schemas/bias.py` line 103): Pydantic model, currently no `overrides` field.

**`LLMGuardianProvider.__init__`** (`src/gaussia/schemas/bias.py` line 75): accepts `model, tokenizer, api_key, url, temperature, safe_token, unsafe_token, max_tokens, logprobs, **kwargs`.

**`OpenAIGuardianProvider._parse_guardian_response`** (`src/gaussia/guardians/llms/providers.py` line 103): line 109 accesses `choice["message"]["content"]` without null guard.

**`BaseContextLoader.load()`** (`src/gaussia/schemas/generators.py` line 126): currently `load(self, source: str) -> list[Chunk]`.

**`SentenceTransformerEmbedder`** (`src/gaussia/embedders/sentence_transformer.py`): constructor does not accept `model=` kwarg — needs verification before fix.

### Anti-patterns confirmed
- Do NOT use `ignore_errors = true` — that is what we are removing.
- Do NOT change logic/behavior in mypy-fix phases, only type annotations.
- Do NOT widen `FrequentistMode` return values — only widen the declared type signature.

---

## Phase 1 — Issue #2: Guardian overrides + null content fix (TDD)

**Goal:** Add `overrides` support to `GuardianLLMConfig` and fix null content crash.

### Step 1.1 — Write failing tests first

File to create: `tests/metrics/test_guardian_overrides.py`

Tests to write (all must FAIL before implementation):
1. `test_guardian_llm_config_accepts_overrides` — instantiate `GuardianLLMConfig` with `overrides={"provider": {"ignore": ["DeepInfra"]}}` and assert `config.overrides == {...}`
2. `test_openai_provider_spreads_overrides_in_chat_completions` — mock `requests.post`, instantiate `OpenAIGuardianProvider` with overrides, call `_with_chat_completions`, assert the posted JSON body contains the override keys
3. `test_openai_provider_spreads_overrides_in_completions` — same for `_with_completions`
4. `test_parse_guardian_response_handles_null_content` — call `_parse_guardian_response` with `{"choices": [{"message": {"content": None}}]}` and assert it returns `(False, 1.0)` instead of raising `TypeError`

Copy fixture pattern from: `tests/metrics/test_bias.py` lines 13–24 (MockGuardian pattern).

**Verify:** `uv run pytest tests/metrics/test_guardian_overrides.py` → all 4 tests FAIL (not error, fail).

### Step 1.2 — Implement: `src/gaussia/schemas/bias.py`

Add to `GuardianLLMConfig` (after line 111):
```python
overrides: dict[str, Any] = {}
```
Also add `from typing import Any` if not already present.

**Verify:** test 1 passes.

### Step 1.3 — Implement: `src/gaussia/schemas/bias.py` → `LLMGuardianProvider`

Add `overrides: dict[str, Any] = {}` parameter to `LLMGuardianProvider.__init__` and store `self._overrides = overrides`.

### Step 1.4 — Implement: `src/gaussia/guardians/llms/providers.py`

In `OpenAIGuardianProvider.__init__`: accept and forward `overrides` to `super().__init__`.

In `_with_chat_completions` and `_with_completions`: spread overrides into request body:
```python
json={
    "model": self.model,
    ...
    **self._overrides,
}
```

In `_parse_guardian_response` line 109: add null guard:
```python
message_content = choice["message"]["content"]
if message_content is None:
    return False, 1.0
```

### Step 1.5 — Implement: `src/gaussia/guardians/__init__.py`

In `LLamaGuard.__init__` and `IBMGranite.__init__`: add `overrides=config.overrides` to the provider constructor call.

**Verify:** `uv run pytest tests/metrics/test_guardian_overrides.py` → all 4 tests PASS.

---

## Phase 2 — Issue #1, Group A: Missing stubs (torch + scipy)

**Goal:** Suppress import-not-found errors for optional deps without breaking runtime.

### Files and exact changes

**`src/gaussia/rerankers/qwen.py` line 3:**
```python
import torch  # type: ignore[import-not-found]
```

**`src/gaussia/embedders/qwen.py` lines 4–5:**
```python
import torch  # type: ignore[import-not-found]
from torch.nn import functional as torch_functional  # type: ignore[import-not-found]
```

**`src/gaussia/metrics/humanity.py` line 9:**
```python
from scipy.stats import spearmanr  # type: ignore[import-not-found]
```

**Verify:** `uv run mypy src/gaussia/rerankers/qwen.py src/gaussia/embedders/qwen.py src/gaussia/metrics/humanity.py --ignore-missing-imports` → 0 errors.

---

## Phase 3 — Issue #1, Group B: `batch()` signature mismatches

**Goal:** Align subclass signatures with base class contract.

### `src/gaussia/metrics/toxicity.py` line 332

Current (wrong order):
```python
def batch(self, session_id, assistant_id, batch, language="english", context=""):
```
Correct order (match base):
```python
def batch(self, session_id: str, context: str, assistant_id: str,
          batch: list[Batch], language: str | None = "english"):
```
Internal body: update any references to `context` and `language` that may have relied on the old positional order.

### `src/gaussia/metrics/bias.py` line 122

Change:
```python
language: str = "en",
```
To:
```python
language: str | None = "english",
```

**Verify:** `uv run mypy src/gaussia/metrics/toxicity.py src/gaussia/metrics/bias.py` → 0 errors on these files.

---

## Phase 4 — Issue #1, Group C: FrequentistMode return type signatures

**Goal:** Widen declared return types to match `StatisticalMode` base. No logic changes.

### `src/gaussia/statistical/frequentist.py`

Change all 4 method signatures from `-> float` to `-> float | dict[str, Any]`:
- `distribution_divergence(...) -> float | dict[str, Any]`
- `rate_estimation(...) -> float | dict[str, Any]`
- `aggregate_metrics(...) -> float | dict[str, Any]`
- `dispersion_metric(...) -> float | dict[str, Any]`

Add `from typing import Any` if not present.

**Verify:** `uv run mypy src/gaussia/statistical/frequentist.py` → 0 errors.

---

## Phase 5 — Issue #1, Group D: Agentic.run() override

**Goal:** Make `Agentic.run()` signature compatible with base class.

### `src/gaussia/metrics/agentic.py` line 157

Current:
```python
@classmethod
def run(cls, retriever: type[Retriever], k: int, **kwargs) -> list[AgenticMetric]:
    return cls(retriever, k=k, **kwargs)._process()
```

Change to (k moved into kwargs, callers pass `k=3` as kwarg):
```python
@classmethod
def run(cls, retriever: type[Retriever], **kwargs) -> list[AgenticMetric]:
    return cls(retriever, **kwargs)._process()
```

`k` is already in `__init__` as a required parameter — callers already pass `Agentic.run(MyRetriever, model=model, k=3)` which goes through `**kwargs` into `__init__`.

**Verify:** `uv run mypy src/gaussia/metrics/agentic.py` → 0 errors on signature lines.

---

## Phase 6 — Issue #1, Group E: Targeted `# type: ignore` for third-party libs

**Goal:** Silence errors that come from LangChain's complex overloads and dynamic attributes — not fixable by us.

### `src/gaussia/llm/judge.py` line 128

```python
result = agent.invoke({"messages": messages})  # type: ignore[call-overload]
```

### `src/gaussia/prompt_optimizer/mipro/mipro.py` lines 92–94

Add `# type: ignore[attr-defined]` to each of the 3 lines accessing unverifiable attributes.

**Verify:** `uv run mypy src/gaussia/llm/judge.py src/gaussia/prompt_optimizer/mipro/mipro.py` → 0 errors.

---

## Phase 7 — Issue #1, Remaining files

### 7.1 `src/gaussia/metrics/best_of.py`

**Problem 1** (line 59): `_session_kings` needs type annotation.
```python
self._session_kings: dict[tuple[str, ...], tuple[str, str, list[BestOfContest]]] = {}
```

**Problem 2** (lines 112–120): `result` is typed as `BestOfJudgeOutput | dict[Any, Any]` but the if/else branches access attributes on the wrong branch. Fix: replace `if self.use_structured_output:` with `if isinstance(result, BestOfJudgeOutput):` — this narrows the type correctly for mypy.

### 7.2 `src/gaussia/schemas/generators.py`

**Problem** (lines 277, 280, 307, 310): `chain.invoke({})` returns `dict[str, Any] | BaseModel`, not the specific output type. Fix: cast the return value.

In `_call_llm` (line 277):
```python
return cast(GeneratedQueriesOutput, chain.invoke({}))
```

In `_call_llm` (line 280) and `_call_llm_conversation` (line 310): the `response.content` access. Fix: cast `response` before accessing `.content`:
```python
response = chain.invoke({})
content = str(cast(Any, response).content)
```

In `_call_llm_conversation` (line 307):
```python
return cast(GeneratedConversationOutput, chain.invoke({}))
```

Add `from typing import cast` if not present.

Also fix `load()` base signature in this file (line 126):
```python
def load(self, source: str | list[str]) -> list[Chunk]:
```

### 7.3 `src/gaussia/guardians/__init__.py`

**Problem 1** (lines 34, 88): `AutoTokenizer.from_pretrained()` returns a union type, not `AutoTokenizer`. The `LLMGuardianProvider.__init__` expects `AutoTokenizer`.
Fix: change `LLMGuardianProvider.__init__` tokenizer param type from `AutoTokenizer` to `Any` in `src/gaussia/schemas/bias.py`.

**Problem 2** (line 51): `self.provider.tokenizer.apply_chat_template` — mypy doesn't know `AutoTokenizer` has this.
Fix: `# type: ignore[attr-defined]`

**Problem 3** (line 107): conversation argument type mismatch.
Fix: `# type: ignore[arg-type]`

### 7.4 `src/gaussia/metrics/regulatory.py` line 228

**Problem:** `session_verdict` is `str` but `RegulatoryMetric` expects `Literal['COMPLIANT', 'NON_COMPLIANT', 'IRRELEVANT']`.
Fix: add `from typing import Literal` and annotate `session_verdict`:
```python
session_verdict: Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"]
```
(The variable is already assigned one of those 3 string literals in the if/elif/else above, so mypy will narrow correctly.)

### 7.5 `src/gaussia/metrics/vision.py` line 24

**Problem:** `SentenceTransformerEmbedder(model=_DEFAULT_MODEL)` — `model` is not a valid kwarg.
Fix: Read `src/gaussia/embedders/sentence_transformer.py` to find the correct parameter name, then update the call. If the class uses `model_name`, change to `SentenceTransformerEmbedder(model_name=_DEFAULT_MODEL)`.

**Verify:** `uv run mypy src/gaussia/metrics/best_of.py src/gaussia/schemas/generators.py src/gaussia/guardians/__init__.py src/gaussia/metrics/regulatory.py src/gaussia/metrics/vision.py` → 0 errors.

---

## Phase 8 — Remove suppression block + final verification

### Step 8.1 — Remove `ignore_errors = true` block from `pyproject.toml`

Delete the entire `[[tool.mypy.overrides]]` block (lines 270–289):
```toml
[[tool.mypy.overrides]]
module = [
    "gaussia.embedders.qwen",
    ...
]
ignore_errors = true
```

### Step 8.2 — Run full mypy
```bash
uv run mypy src/gaussia
```
Expected: `Success: no issues found in 88 source files`

If any errors remain: fix them before proceeding.

### Step 8.3 — Run test suite
```bash
uv run pytest -m "not slow"
```
Expected: all tests pass, coverage ≥ 50%.

### Step 8.4 — Run linter
```bash
uv run ruff check .
```
Expected: no errors.

---

## Phase 9 — Commit and PR

### Commits (Conventional Commits format)

```
fix(guardians): add overrides support and null content guard in provider
fix(mypy): resolve all pre-existing type errors and remove ignore_errors suppression
```

### PR
- Title: `fix: resolve mypy errors and add guardian provider overrides`
- Base branch: `develop`
- References: closes gaussia-labs/pygaussia#1, closes gaussia-labs/pygaussia#2
