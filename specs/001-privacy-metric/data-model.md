# Data Model: Privacy Metric

**Branch**: `001-privacy-metric` | **Companion to**: `plan.md`

This document specifies the Pydantic schemas, their fields, constraints, validators, and inter-entity relationships. All schemas live in `src/gaussia/schemas/privacy.py` except `PIIDetector` (in `src/gaussia/core/detector.py`).

## Entity Map

```
PrivacyDomainConfig ──┐
                      │
PIIDetector ──────────┼──> Privacy / PrivacyRanker ──> PrivacyMetric / PrivacyRanking
                      │                                       │
PrivacyBatch ─────────┘                                       └── ClassMetrics
   │                                                              DetectionScoreContribution
   └── Span                                                       CriticalFNContribution
```

## `Span`

Atomic unit of PII annotation. Used in both ground-truth (`PrivacyBatch.spans`) and prediction outputs (`PIIDetector.predict` return value).

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `label` | `str` | `min_length=1` | Domain class identifier (e.g., `"email_address"`). Must be a member of `PrivacyDomainConfig.classes` to count in the score; otherwise filtered (FR-015). |
| `start` | `int` | `ge=0` | Inclusive character offset. |
| `end` | `int` | `gt=0` | Exclusive character offset. |
| `text` | `str` | — | Surface form. May be derived (`source_text[start:end]`) but stored explicitly for auditability. |
| `score` | `float \| None` | `ge=0.0, le=1.0` | Detector confidence. `None` when the ground-truth annotation has no confidence (typical). Used by greedy NMS in overlapping-prediction resolution. |

**Model validator**: `start < end` must hold; otherwise `ValueError("span start must be strictly less than end")`.

**Why no `source_text` reference**: spans live inside a `PrivacyBatch` whose `query` is the source. Carrying the parent text per span would duplicate storage and risk drift.

## `PrivacyBatch`

Strict subclass of `gaussia.schemas.common.Batch`. Adds the ground-truth annotations.

Inherits all `Batch` fields. Adds:

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `spans` | `list[Span]` | default `[]` | Ground-truth PII spans for `query`. May be empty (turn contains no PII). |

**Usage convention** (consistent with C.2 decision):
- `query` = text shown to the detector.
- `assistant` = `""` (no conversational counterpart in this evaluation).
- `ground_truth_assistant` = `""` (same).
- `spans` carries the semantic payload.

## `PrivacyDomainConfig`

Frozen configuration object describing the evaluation domain. One instance is built by the user and shared by all detectors in a run.

| Field | Type | Constraints | Notes |
|-------|------|-------------|-------|
| `classes` | `frozenset[str]` | `min_length=1` | The domain taxonomy `C_d`. |
| `criticality_weights` | `dict[str, float]` | each value in `[0, 1]` | The `w_{c,d}` map. |
| `fn_severity_weights` | `dict[str, float]` | each value in `[0, 1]` | The `ρ_{c,d}` map. |
| `iou_threshold` | `float` | `gt=0.0, le=1.0`, default `0.50` | Single-scalar IoU cutoff for span matching (FR-006). |
| `regulatory_framework` | `str \| None` | default `None` | Free-form label (`"HIPAA"`, `"GDPR"`, `"Law 25.326"`, …). Informational; does not change computation. |

**Model validators**:

1. `criticality_weights` keys MUST equal `classes` exactly (no extras, no missing). Same for `fn_severity_weights`.
2. `sum(criticality_weights.values()) == 1.0 ± 1e-9` (FR-011). Same constraint for `fn_severity_weights`.
3. `model_config = ConfigDict(frozen=True)` — once constructed, the config is immutable.

**Why frozen**: a single run reuses the config across all detectors and all batches. Mutation mid-run would silently invalidate comparability (the paper's comparability requirement in Implementation Considerations).

**Why no `domain_fit` / `regulatory_fit` here**: those are per-detector judgements (FR-005), not per-domain constants. They live on the `PIIDetector` instance.

## `PIIDetector` (abstract, in `core/detector.py`)

Adapter / Strategy base for all PII detection backends.

```python
class PIIDetector(ABC):
    name: str
    domain_fit: float                # in [0, 1]
    regulatory_fit: float            # in [0, 1]

    @property
    @abstractmethod
    def supported_classes(self) -> frozenset[str]: ...

    @abstractmethod
    def predict(self, text: str) -> list[Span]: ...

    def setup(self) -> None:
        """Optional hook for one-time initialisation (model loading).
        Called by Privacy/PrivacyRanker exactly once, before any predict() call.
        Default: no-op. Subclasses override if they need lazy heavy initialisation."""
```

**Type note**: `supported_classes` returns `frozenset[str]` (immutable, hashable). FR-003 in `spec.md` writes `set[str]` informally; the canonical type across the implementation is `frozenset[str]`, and FR-003 is aligned to it. `PrivacyMetric.supported_classes` is a sorted `list[str]` because it is a serialised snapshot, not the live contract.

**Concrete subclasses**:

- `PresidioDetector(name, domain_fit, regulatory_fit, *, language="en", supported_languages=None)`
  - Wraps `presidio_analyzer.AnalyzerEngine` with a blank spaCy pipeline (matches sandbox `_build_blank_presidio_engine`).
  - `supported_classes` = the domain projection of Presidio's standard recognizers (the sandbox map `_PRESIDIO_TO_DOMAIN.values()` is the seed; the actual classes returned must be derivable from the engine's loaded recognizers, not a hardcoded global).

- `HuggingFacePIIDetector(name, model_path, domain_fit, regulatory_fit, *, device=-1, aggregation_strategy="simple")`
  - Wraps `transformers.pipeline("token-classification", model=model_path, ...)`.
  - `supported_classes` derived from `model.config.id2label` after applying a label-canonicaliser (the sandbox `LABEL_ALIASES` + `canonicalize` logic, translated into a small pure module under `detectors/_label_canonicaliser.py`).

**Constructor contract**: both `domain_fit` and `regulatory_fit` are required positional/keyword arguments. No defaults (FR-005).

## `ClassMetrics`

Per-class confusion + derived rates. Embedded in `PrivacyMetric.class_metrics`.

| Field | Type | Constraints |
|-------|------|-------------|
| `tp` | `int` | `ge=0` |
| `fp` | `int` | `ge=0` |
| `fn` | `int` | `ge=0` |
| `precision` | `float` | `ge=0.0, le=1.0` |
| `recall` | `float` | `ge=0.0, le=1.0` |
| `f2` | `float` | `ge=0.0, le=1.0` |
| `fn_rate` | `float` | `ge=0.0, le=1.0` |
| `criticality_weight` | `float` | `ge=0.0, le=1.0` |
| `fn_severity_weight` | `float` | `ge=0.0, le=1.0` |

Frozen.

## `DetectionScoreContribution`

Per-class component of `DetectionScore = Σ_c w_c · f2_c`. Embedded in `PrivacyMetric.detection_score_contributions`.

| Field | Type | Constraints |
|-------|------|-------------|
| `weight` | `float` | `ge=0.0, le=1.0` |
| `f2` | `float` | `ge=0.0, le=1.0` |
| `contribution` | `float` | `ge=0.0, le=1.0` (== `weight * f2`) |

Frozen. Model validator: `abs(contribution - weight * f2) < 1e-9`.

## `CriticalFNContribution`

Per-class component of `CriticalFN = Σ_c ρ_c · fn_rate_c`. Embedded in `PrivacyMetric.critical_fn_contributions`.

| Field | Type | Constraints |
|-------|------|-------------|
| `severity_weight` | `float` | `ge=0.0, le=1.0` |
| `fn_rate` | `float` | `ge=0.0, le=1.0` |
| `contribution` | `float` | `ge=0.0, le=1.0` (== `severity_weight * fn_rate`) |

Frozen. Same product-equality validator.

## `PrivacyMetric` (`extends BaseMetric`)

Per-detector evaluation result. One instance per detector per run.

### Identity & status

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str` | Detector identifier (== `detector.name`). |
| `success` | `bool` | `True` for completed runs, `False` when caught by `PrivacyRanker`'s per-detector exception handler. |
| `error` | `str \| None` | Exception message when `success=False`; `None` otherwise. |

### Latency (informational only — FR-019)

| Field | Type | Constraints |
|-------|------|-------------|
| `load_time` | `float` | `ge=0.0` — seconds spent in `detector.setup()`. |
| `inference_latency` | `float` | `ge=0.0` — total seconds spent inside `detector.predict` across the corpus. |

### Corpus diagnostics

| Field | Type | Constraints |
|-------|------|-------------|
| `corpus_samples` | `int` | `ge=0` — total `PrivacyBatch`es processed. |
| `ground_truth_count` | `int` | `ge=0` — total ground-truth spans. |
| `raw_prediction_count` | `int` | `ge=0` — total predictions returned by `detector.predict` before filtering. |
| `prediction_count` | `int` | `ge=0` — predictions remaining after out-of-domain filter + NMS. |
| `out_of_domain_filtered` | `int` | `ge=0` — count of predictions dropped because their label is not in `C_d` (FR-015). |

### Detection score

| Field | Type | Constraints |
|-------|------|-------------|
| `class_metrics` | `dict[str, ClassMetrics]` | One entry per `c ∈ C_d`. |
| `detection_score` | `float` | `ge=0.0, le=1.0` — `Σ_c w_c · f2_c`. |
| `detection_score_contributions` | `dict[str, DetectionScoreContribution]` | One entry per `c ∈ C_d`. |

### Coverage

| Field | Type | Constraints |
|-------|------|-------------|
| `coverage` | `float` | `ge=0.0, le=1.0` |
| `supported_classes` | `list[str]` | Output of `detector.supported_classes`, sorted. |
| `covered_domain_classes` | `list[str]` | `C_M ∩ C_d`, sorted. |
| `missing_domain_classes` | `list[str]` | `C_d \ C_M`, sorted. |

### Context fits (passthrough from detector)

| Field | Type | Constraints |
|-------|------|-------------|
| `domain_fit` | `float` | `ge=0.0, le=1.0` |
| `regulatory_fit` | `float` | `ge=0.0, le=1.0` |

### Penalty / Risk

| Field | Type | Constraints |
|-------|------|-------------|
| `critical_fn` | `float` | `ge=0.0, le=1.0` |
| `penalty_fn` | `float` | `ge=0.0, le=1.0` |
| `critical_fn_contributions` | `dict[str, CriticalFNContribution]` | One entry per `c ∈ C_d`. |
| `r1_weakest_class_risk` | `float` | `ge=0.0, le=1.0` |
| `r1_weakest_class` | `str \| None` | Class achieving the maximum; `None` only if all `fn_rate = 0`. |
| `r2_systemic_risk` | `float` | `ge=0.0, le=1.0` |
| `r_final` | `float` | `ge=0.0, le=1.0` |

### Final score

| Field | Type | Constraints |
|-------|------|-------------|
| `score` | `float` | `ge=0.0, le=1.0` |
| `score_100` | `float` | `ge=0.0, le=100.0` (== `score * 100`) |
| `interpretation` | `Literal[...]` | One of `"Not suitable"`, `"Baseline / complement only"`, `"Suitable with mitigations"`, `"Operationally suitable"`, `"Recommended"`, `"Recommended with strong local evidence"`. Derived from `score_100` per the paper's interpretation table. |

### Computed fields (`@computed_field`, Pydantic v2 — not stored)

These are emitted in `model_dump(mode="json")` for sandbox-JSON compatibility but are derived, not stored, to avoid duplicating data in memory.

| Field | Derivation |
|-------|------------|
| `r1_weakest_class_risk_100` | `r1_weakest_class_risk * 100` |
| `r2_systemic_risk_100` | `r2_systemic_risk * 100` |
| `r_final_100` | `r_final * 100` |

**Model validators on `PrivacyMetric`**:

1. `abs(score_100 - score * 100) < 1e-6`.
2. `abs(score - detection_score * coverage * domain_fit * regulatory_fit * penalty_fn) < 1e-9`.
3. `abs(r_final - (1 - (1 - r1_weakest_class_risk) * (1 - r2_systemic_risk))) < 1e-9`.
4. `class_metrics.keys() == covered_domain_classes ∪ missing_domain_classes == set(C_d)` (from the run's `PrivacyDomainConfig.classes`).

These guard against silent inconsistency between components and aggregate fields — Pydantic catches drift at construction even if the metric class miscomputes.

**Failure path (`success=False`)**: validators 1–4 are **skipped** when `success is False`. They encode the relationship between the score components, which only hold for a completed evaluation. A failed entry (produced by `PrivacyRanker`'s per-detector exception handler) is constructed with `success=False`, a populated `error`, `score=0.0`, `score_100=0.0`, and the remaining numeric fields left at their neutral defaults (`detection_score=0.0`, `coverage=0.0`, `penalty_fn=1.0`, empty `class_metrics`/contribution dicts, `r_final` and the `r*` fields at `0.0`). The four cross-field validators short-circuit via `if self.success is False: return self` at the top of the `model_validator`, so a failed entry never needs to fabricate an internally-consistent all-zero metric just to pass construction.

## `PrivacyRanking` (`extends BaseMetric`)

Aggregate result emitted by `PrivacyRanker`. One instance per run.

| Field | Type | Constraints |
|-------|------|-------------|
| `results` | `list[PrivacyMetric]` | Ordered by `score_100` descending; failed entries (`success=False`) sorted to the tail. |
| `winning_detector` | `str \| None` | `results[0].name` when at least one entry has `success=True`; otherwise `None`. |
| `iou_threshold` | `float` | Echo of the run's `PrivacyDomainConfig.iou_threshold` for traceability. |
| `domain_classes` | `list[str]` | Sorted echo of `PrivacyDomainConfig.classes`. |
| `regulatory_framework` | `str \| None` | Echo. |

**Model validators**:

1. `results` ordering: for any two adjacent successful entries `a, b`, `a.score_100 >= b.score_100`.
2. Failed entries (`success=False`) appear strictly after all successful entries.
3. `winning_detector` consistent with `results[0].name` (or `None`).

## Field naming compatibility with sandbox JSON

`PrivacyMetric` field names match the sandbox JSON keys (`score_100`, `r_final_100`, `r1_weakest_class_risk`, `critical_fn_contributions`, etc.) so that `panel_chatbot_v3.py` works on the metric's `model_dump(mode="json")` output without modification. The sandbox's redundant `*_100` fields (`r1_weakest_class_risk_100`, `r2_systemic_risk_100`, `r_final_100`) are not stored as separate Pydantic fields — they are emitted by a `@computed_field` (Pydantic v2) so that consumers see the same JSON shape without duplicating data in memory.

## Failure modes summarised

| Where | Error | Source |
|-------|-------|--------|
| `Span(start=10, end=5)` | `ValidationError`: start ≥ end | Span validator |
| `PrivacyDomainConfig(criticality_weights={"a": 0.5, "b": 0.4})` | `ValidationError`: weights do not sum to 1.0 | Config validator |
| `PrivacyDomainConfig(classes={"a","b"}, criticality_weights={"a": 1.0})` | `ValidationError`: weight keys ≠ classes | Config validator |
| `PresidioDetector(name="p", domain_fit=1.5, regulatory_fit=0.8)` | `ValidationError`: domain_fit > 1 | Field constraint |
| Detector raises in `Privacy.run` | Propagated | Fail-fast policy |
| Detector raises in `PrivacyRanker.run` | Caught; `PrivacyMetric(success=False, error=...)` appended to `results` | Fail-soft policy |
| `PrivacyMetric` constructed with `score_100 = score * 99` | `ValidationError`: aggregate consistency | Metric validator |
