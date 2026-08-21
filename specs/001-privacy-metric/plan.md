# Implementation Plan: Domain-Adjusted Privacy Detection Metric and Ranker

**Branch**: `001-privacy-metric` | **Date**: 2026-05-28 | **Spec**: `specs/001-privacy-metric/spec.md`

## Summary

Translate the **Domain-Adjusted Privacy Detection Score** and **Risk Index** from `papers/2026-05-privacy/` into two reusable `Gaussia` subclasses (`Privacy`, `PrivacyRanker`) that consume a user-supplied labelled corpus and one or more injected `PIIDetector` adapters. The PR #10 sandbox is the numerical reference; its monolithic structure is intentionally not ported. The library remains pure compute: it owns the formulas, the schemas, and the detector abstraction, and delegates corpus parsing (to the user's `Retriever`) and visualisation (to the existing `panel_chatbot_v3.py` external tool).

## Technical Context

**Language/Version**: Python 3.11+ (matches existing pygaussia stack).
**Primary Dependencies**: No new mandatory runtime deps. Two new optional extras: `[project.optional-dependencies].privacy-presidio` (pins `presidio-analyzer`, `spacy`) and `.privacy-huggingface` (pins `transformers`, `torch`). Per FR-017 these are declared in `pyproject.toml` and surface a clean `ImportError` from the adapter module's import site — no `try/except ImportError`, no dynamic imports, no runtime `pip install`.
**Testing**: `uv run pytest` (existing convention). Correctness is verified by hand-computed `StubDetector` tests. The PR #10 sandbox artifact (`tests/privacy/chatbot_v3_results.json`, generated over the 500-turn corpus) is reproduced only by an optional, opt-in integration test (`RUN_SANDBOX=1` + extras), not by the default suite.
**Project Type**: Library extension (additive to `src/gaussia/`).

## Constitution Check

*Filled honestly per `/Users/frino/.../pygaussia/.specify/memory/constitution.md` and `references/constitution-base.md`.*

### SOLID Gate

- [x] **Single Responsibility**: `Privacy` evaluates one detector. `PrivacyRanker` orchestrates N detectors and ranks. Each `PIIDetector` adapter translates one backend (Presidio, HF) into the shared protocol. Each Pydantic schema describes one shape. No class straddles two responsibilities.
- [x] **Open/Closed**: Adding a third detector backend (e.g., GLiNER, OpenAI Moderation) requires only a new file under `src/gaussia/detectors/` implementing `PIIDetector`. `Privacy`, `PrivacyRanker`, and every schema remain untouched. SC-005 enforces this.
- [x] **Liskov**: `PrivacyBatch` is a strict subclass of `Batch` — every `Batch` field stays valid, `spans` is the only addition with a safe default `[]`. Any code that receives `list[Batch]` accepts `list[PrivacyBatch]` without modification. Concrete `PIIDetector` subclasses honour the same `predict` / `supported_classes` contract.
- [x] **Interface Segregation**: `PIIDetector` exposes only `predict(text: str) -> list[Span]`, the `supported_classes: frozenset[str]` property, and the two contextual scalars (`domain_fit`, `regulatory_fit`). No `train`, no `fine_tune`, no `serialize` — those belong elsewhere if ever needed.
- [x] **Dependency Inversion**: `Privacy` and `PrivacyRanker` import `PIIDetector` (abstraction), never `PresidioDetector` or `HuggingFacePIIDetector` directly. Concrete adapters are injected by the caller at construction.

### Pattern Gate

- [x] **No string-driven branching**: Selection between Presidio and HF is by class identity (the user instantiates `PresidioDetector(...)` or `HuggingFacePIIDetector(...)`). No `if backend == "presidio"`. The sandbox's `paradigm: "huggingface" | "presidio_baseline"` string switch is deliberately removed.
- [x] **No conditional chains for behaviour selection**: Per-detector behaviour is dispatched by polymorphism through `PIIDetector.predict`. The sandbox's `if paradigm == "huggingface": ... elif paradigm == "presidio_baseline": ...` block does not survive translation.
- [x] **Appropriate pattern identified**:
  - **Template Method**: `Privacy(Gaussia)` and `PrivacyRanker(Gaussia)` inherit `_process` from `Gaussia` and implement `batch`.
  - **Strategy / Adapter**: `PIIDetector` is the strategy interface; each concrete adapter wraps a third-party backend.
  - **Composition**: `PrivacyRanker` composes a `list[PIIDetector]` at construction.
- [x] **Composition over inheritance where applicable**: Detectors are composed into `Privacy` / `PrivacyRanker`, never inherited from. Schemas use composition (e.g., `PrivacyMetric` contains a `dict[str, ClassMetrics]`).

### Simplicity Gate

- [x] **No speculative features**: The plan implements exactly what FR-001…FR-020 require. No "infrastructure score for the future", no plug-in registry, no caching layer, no async detector execution. The paper's `InfraScore` is explicitly excluded by FR-019 (informational latency fields only).
- [x] **No premature abstractions**: `PIIDetector` exists because there are already two concrete backends in scope (Presidio + HF). It is not abstraction-for-its-own-sake. `PrivacyBatch` exists because `Batch` cannot type-check `spans` (FR-006 mandates IoU matching over typed spans).
- [x] **No over-engineered error handling**: Detector failures use the simplest reasonable policy (fail-fast in single-detector, fail-soft per-detector in the ranker). No retries, no backoff, no fallback chains.

### Pipeline Gate

- [x] **Respects the SDK's data flow**: `Retriever.load_dataset() → list[Dataset] → Gaussia._process() → Metric.batch() → self.metrics`. `Privacy` and `PrivacyRanker` implement `batch`; the user provides a `Retriever` that yields `Dataset`s whose `conversation` is `list[PrivacyBatch]`.
- [x] **New schemas follow SDK conventions**: All new entities are Pydantic `BaseModel`s. `PrivacyMetric` and `PrivacyRanking` inherit from `BaseMetric` (matching `RegulatoryMetric`, `ToxicityMetric`, …). `PrivacyBatch` subclasses `Batch`.
- [x] **Module boundaries respected**: `core/` holds the abstract `PIIDetector` only. `detectors/` holds concrete adapters. `metrics/` holds the two `Gaussia` subclasses. `schemas/privacy.py` holds Pydantic models. No upward or sideways imports between concrete modules.

## Architecture Decisions

### Why a new `core/detector.py` abstraction

`PIIDetector` cannot reuse existing `Guardian`, `Reranker`, or `Embedder` interfaces because none of them speak the language of token-level span prediction (`text → list[(label, start, end)]`). Creating a sibling abstraction in `core/` follows the same convention these existing interfaces use and keeps the dependency direction clean (`metrics` → `core`, never the reverse). This addresses DIP and ISP simultaneously.

### Why a new `detectors/` module

The repo already segregates concrete strategy implementations by domain: `guardians/`, `rerankers/`, `embedders/`. `detectors/` mirrors this convention so that future contributors find adapters where they expect them. Putting Presidio and HuggingFace adapters in `metrics/` would conflate "what is computed" with "how predictions are obtained" — a Feature Envy smell.

### Why `PrivacyRanker` is a `Gaussia` and not a free helper

Per decision B.2 (user-confirmed): every public metric in pygaussia is a `Gaussia` subclass so that uniform tooling (CLI runners, dashboards, `BaseRunner`) can drive it. Forcing the ranker through the same lifecycle is the cost of consistency. The axis-flip cost — that the "fixed model under test" abstraction does not naturally describe "N models under test" — is paid inside `PrivacyRanker.batch`, which iterates over the injected detectors per turn. `session_id` and `assistant_id` are populated by the user's `Retriever` and are semantically vestigial here; they remain in the signature to honour LSP with `Gaussia.batch`.

### Why no `StatisticalMode`

`FrequentistMode` and `BayesianMode` exist to aggregate per-batch sample distributions into session-level estimates with confidence/credible intervals. The Privacy Score is a deterministic function of corpus-wide counts (`TP`, `FP`, `FN` summed across all batches). There is no per-batch random variable to bootstrap. FR-020 forbids accepting a `statistical_mode` parameter to avoid the illusion that one is supported.

### Why latency fields stay (FR-019)

The `panel_chatbot_v3.py` diagnostic panel reads `load_time` and `inference_latency` from the result JSON. Removing them silently would break that downstream consumer. They are explicitly walled off from any Score / Risk computation; their only purpose is operational diagnostic continuity.

### Verification strategy: hand-computed `StubDetector` tests, not sandbox reproduction

Correctness is verified with a deterministic `StubDetector` whose predictions are fixed by the test, so every component (per-class F2, DetectionScore, Coverage, Penalty_FN, R1/R2/R_final, Score) is asserted against values computed by hand from the paper's formulas (SC-001, SC-004). These tests carry no third-party dependency, run in milliseconds, and are the authoritative proof that the formulas are implemented correctly. A small multi-class fixture reproduces the paper's worked example (`Score_100 = 28.36`) at reduced scale.

Reproducing the PR #10 sandbox artifact (`chatbot_v3_results.json`) is kept as an **optional integration check**, NOT a unit test. That JSON was generated by `Analisis_Score_Chatbot_v3.py` over the **500-turn** corpus (`chatbot_conversations_500.txt` / `chatbot_conversations_tagged_500.txt`; the script reports `corpus_samples = 4935`), running six real heavy detectors (Presidio + five OpenMed models). The earlier assumption that the `eval_*_100.txt` files produced this JSON was incorrect — they did not, so a 100-turn run can never match it. Faithful reproduction therefore requires the 500-turn corpus AND the heavy backends; it is gated behind the optional extras plus a `RUN_SANDBOX=1` flag and excluded from the default `uv run pytest` run.

## New Files

| File | Purpose | Pattern |
|------|---------|---------|
| `src/gaussia/core/detector.py` | Abstract `PIIDetector` interface (`predict`, `supported_classes`, `domain_fit`, `regulatory_fit`) | Strategy / Adapter base |
| `src/gaussia/detectors/__init__.py` | Module init re-exporting concrete adapters | Module facade |
| `src/gaussia/detectors/presidio.py` | `PresidioDetector` wrapping `presidio_analyzer.AnalyzerEngine` | Adapter |
| `src/gaussia/detectors/huggingface.py` | `HuggingFacePIIDetector` wrapping a `transformers` token-classification pipeline | Adapter |
| `src/gaussia/schemas/privacy.py` | `Span`, `PrivacyBatch`, `PrivacyDomainConfig`, `ClassMetrics`, `DetectionScoreContribution`, `CriticalFNContribution`, `PrivacyMetric`, `PrivacyRanking` | Pydantic models |
| `src/gaussia/metrics/privacy.py` | `Privacy(Gaussia)` and `PrivacyRanker(Gaussia)` | Template Method + Strategy |
| `tests/metrics/test_privacy.py` | Unit + integration tests for `Privacy` and `PrivacyRanker`, including numerical regression vs sandbox JSON | Pytest |
| `tests/detectors/__init__.py` | Empty marker | — |
| `tests/detectors/test_presidio.py` | Adapter contract + integration tests (skipped if extra not installed via a `pytest.importorskip` at file top) | Pytest |
| `tests/detectors/test_huggingface.py` | Same for HF adapter | Pytest |
| `tests/integration/test_privacy_sandbox.py` | **Optional** integration test reproducing the sandbox JSON over the 500-turn corpus; `@pytest.mark.slow`, skipped unless extras installed and `RUN_SANDBOX=1` set. Not part of the default suite. | Pytest (opt-in) |
| `tests/fixtures/privacy/__init__.py` | Empty marker | — |
| `tests/fixtures/privacy/stub_detector.py` | Deterministic `PIIDetector` for arithmetic tests (no external deps) — the primary verification double | Test double |
| `tests/fixtures/privacy/sandbox_corpus.py` | Loader that parses the **500-turn** sandbox corpus (`chatbot_conversations_tagged_500.txt` / `chatbot_conversations_500.txt`) into `list[Dataset[PrivacyBatch]]`; used only by the optional integration test | Test fixture |
| `tests/fixtures/privacy/sandbox_results.json` | Copy of `tests/privacy/chatbot_v3_results.json` (the 500-turn baseline) pinned for the optional integration test | Test fixture |

## Modified Files

| File | Change | Justification |
|------|--------|---------------|
| `src/gaussia/core/__init__.py` | Add `from .detector import PIIDetector` and append `"PIIDetector"` to `__all__` | Public API surface convention — every abstract in `core/` is re-exported here. |
| `src/gaussia/schemas/__init__.py` | Add the module-level import comment block for `from gaussia.schemas.privacy import ...` (matching the convention noted in the file's docstring for metric-specific schemas) | The repo's pattern is to NOT eagerly import metric-specific schemas (per the existing comment in `schemas/__init__.py`) but to document them in the file docstring. We honour this. |
| `src/gaussia/metrics/__init__.py` | Add `from .privacy import Privacy, PrivacyRanker` and extend `__all__` | Matches existing pattern for `Toxicity`, `Bias`, `Regulatory`, etc. |
| `pyproject.toml` | Add `[project.optional-dependencies] privacy-presidio = ["presidio-analyzer>=2.2", "spacy>=3.7"]` and `privacy-huggingface = ["transformers>=4.40", "torch>=2.1"]` | FR-017 requires extras instead of `try/except ImportError`. Version pins follow the sandbox `_ensure_dependencies()` behaviour but are explicit. |
| `CHANGELOG.md` | Append entry under "Unreleased" describing the new metric | Repo convention for user-visible changes. |

## Complexity Tracking

The Constitution Check passes without violations. The single tension — that `PrivacyRanker` runs many models against one corpus, inverting the natural "one model, many conversations" axis of `Gaussia.batch` — is acknowledged in the spec (Decision B.2) and absorbed inside `PrivacyRanker.batch` by iterating detectors per turn. This is an instance of "forced fit", not an architectural violation. No table entry needed.
