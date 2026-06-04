# Feature Specification: Domain-Adjusted Privacy Detection Metric and Ranker

**Feature Branch**: `001-privacy-metric`
**Created**: 2026-05-28
**Status**: Planned

## Revision Note (post-approval corrections)

The spec was approved and merged via PR #11. The changes in the plan-gate PR are **writing-error corrections, not scope changes** — as a rule we do not re-open approved specs, but factual mistakes are fixed and recorded here for traceability:

1. **Corpus fact (substantive):** SC-001, SC-002 and US2 acceptance scenario #2 originally claimed the 100-turn files (`eval_*_100.txt`) reproduce `chatbot_v3_results.json`. That artifact was actually generated over the **500-turn** corpus (`corpus_samples = 4935`). Correctness verification is now defined via hand-computed `StubDetector` tests; the sandbox reproduction is an optional, opt-in integration check (SC-002a).
2. **Housekeeping:** `Status` Draft → Planned; the `[NEEDS CLARIFICATION]` on the Implementation Issue resolved to N/A; FR-003 `set[str]` → `frozenset[str]` to match the data model.

## Paper Reference

- **Paper**: `gaussia-labs/papers/papers/2026-05-privacy/`
- **Paper PR**: gaussia-labs/papers#16
- **Implementation Issue**: N/A — this metric is tracked through the SDD gate PRs on branch `001-privacy-metric` (spec → plan → tasks → code); no separate tracking issue is opened.

### Extracted from Paper

- **Formal Definition**: The paper proposes the **Domain-Adjusted Privacy Detection Score**, a multiplicative composite metric in `[0, 1]` (presented on a 0–100 scale) that integrates five normalised dimensions for evaluating PII / PHI detection models in regulated domains:

  ```
  Score(M, d) = DetectionScore(M, d)
              · Coverage(M, d)
              · DomainFit(M, d)
              · RegulatoryFit(M, d, r)
              · Penalty_FN(M, d)
  ```

  where, for a domain `d` with class set `C_d`, model `M`, criticality weights `w_{c,d}` (summing to 1), severity weights `ρ_{c,d}` (summing to 1), and class-level confusion counts `TP_{i,c} / FP_{i,c} / FN_{i,c}`:

  - `DetectionScore = Σ_c w_{c,d} · F2_{i,c}` — recall-weighted F-score aggregated by criticality.
  - `Coverage = |C_{M_i} ∩ C_d| / |C_d|` — taxonomic coverage of the model against the domain.
  - `DomainFit, RegulatoryFit ∈ [0, 1]` — expert-provided contextual scalars per model.
  - `Penalty_FN = 1 − Σ_c ρ_{c,d} · FNrate_c` — severity-weighted penalty on critical false negatives.

  The paper additionally defines a **Risk Index**, complementary to the Score:

  ```
  R1       = max_c ( ρ_{c,d} · FNrate_c )        // weakest-link risk
  R2       = 1 − ( Penalty_FN · Coverage )       // systemic residual risk
  R_final  = 1 − (1 − R1) · (1 − R2)             // probabilistic union
  ```

- **Algorithm**: The paper's `Domain-adjusted model selection` algorithm iterates each candidate model over a validated dataset, computes per-class precision / recall / F2, then DetectionScore / Coverage / DomainFit / RegulatoryFit / Penalty_FN / Score, and finally returns the `argmax`-ranked model. The Risk Index is computed alongside but does not enter the ranking objective.

- **Implementation Considerations** (per Section "Implementation Considerations" of the paper):
  - The metric is **model-agnostic**: each model is treated as a black box. The user controls inputs, outputs, taxonomy, weights, and the set of evaluated models.
  - Required output fields per evaluated model: identifier, application domain, required taxonomy, classes covered by the model, class-level metrics, detection score, coverage, domain fit, regulatory fit, false negative penalty, normalised final score, and traceable justification.
  - Comparability requirement: all models must be evaluated under the same dataset, taxonomy, normalisation policy, decision thresholds, and severity criteria.
  - The multiplicative form requires that non-applicable dimensions be set to neutral (`= 1`), never to zero.

- **Evaluation Baseline**: The paper presents an illustrative computation for a Presidio configuration in the healthcare domain, yielding `Score_100 = 28.36` from `DetectionScore = 0.8231`, `Coverage = 0.7273`, `DomainFit = 0.85`, `RegulatoryFit = 0.80`, `Penalty_FN = 0.81`. The pygaussia PR #10 sandbox extends this with a 500-turn chatbot corpus and six concrete detectors (Presidio Baseline + five OpenMed models), producing a `chatbot_v3_results.json` artifact that this implementation should reproduce numerically as a regression baseline.

## User Scenarios & Testing

### User Story 1 — Evaluate a single PII detector against a labelled corpus (Priority: P1)

A privacy engineer has a labelled corpus of texts containing PII (each turn carries ground-truth spans with class labels) and one PII detector under consideration. They need a single, auditable score that reflects how suitable that detector is for their domain, plus a risk index that flags whether the detector has critical blind spots.

**Why this priority**: This is the irreducible unit of the methodology. Every higher-level use case (multi-model ranking, longitudinal tracking, regulatory reporting) is built on top of "evaluate one detector against one corpus". Without P1 nothing else exists.

**Independent Test**: Provide a small fixture corpus with hand-labelled spans, a stub detector with deterministic predictions, and verify that the emitted `PrivacyMetric` reproduces the numerical components (DetectionScore, Coverage, Penalty_FN, R1, R2, R_final, Score_100) computed by hand from the paper's formulas.

**Acceptance Scenarios**:

1. **Given** a corpus with labelled `PrivacyBatch`es and a `PIIDetector` covering all domain classes with perfect predictions, **When** `Privacy.run` is invoked, **Then** the emitted metric has `DetectionScore = 1.0`, `Coverage = 1.0`, `Penalty_FN = 1.0`, `Score = DomainFit · RegulatoryFit`, and `R_final = 0` (no risk).
2. **Given** a corpus and a detector that systematically misses all instances of one critical class (FNrate = 1.0 for that class), **When** the metric is computed, **Then** `R1 = ρ_c` for that class and is non-zero even if other classes are perfect, exposing the blind spot that the Score alone might dilute.
3. **Given** class criticality weights that do not sum to 1.0, **When** the user constructs the `PrivacyDomainConfig`, **Then** Pydantic raises a `ValidationError` at construction time, before any computation runs.
4. **Given** a domain where `RegulatoryFit` does not apply (no regulation in scope), **When** the user sets `regulatory_fit = 1.0` (neutral), **Then** the Score is not artificially nullified and the omission is traceable from the result.

---

### User Story 2 — Rank multiple detectors against the same corpus (Priority: P2)

The same privacy engineer has several candidate detectors (Presidio, OpenMed, custom regex, etc.) and needs a single artifact that orders them by overall suitability while also surfacing each one's strengths and weaknesses per class — so they can decide which to deploy, which to ensemble, and which to discard.

**Why this priority**: Single-detector evaluation answers "is this one any good?" but the operational decision is "which one do I ship?". The ranker is the deliverable a stakeholder reads. It also gates the downstream HTML diagnostic panel, which consumes the ranking JSON.

**Independent Test**: Provide three stub detectors with deterministic predictions ordered such that one has the highest Score, one has the highest Coverage but lower Score, and one has a critical FN gap; verify that `PrivacyRanker.run` emits a `PrivacyRanking` with the correct Score-descending order and that each entry contains the full per-class breakdown.

**Acceptance Scenarios**:

1. **Given** three detectors with computable Scores `S1 > S2 > S3`, **When** `PrivacyRanker.run` is invoked over a shared corpus, **Then** the resulting `PrivacyRanking.results` list is ordered `[S1, S2, S3]` by `score_100` descending.
2. **Given** the sandbox PR #10 500-turn corpus (`chatbot_conversations_500.txt` / `chatbot_conversations_tagged_500.txt`) and the same six real detectors, **When** the ranker is run under the optional integration test (`RUN_SANDBOX=1` + extras), **Then** the resulting per-detector numerical components match `chatbot_v3_results.json` within a 1e-6 relative tolerance for `score_100`, `r_final_100`, and per-class `f2`. (This is an opt-in check, not part of the default suite — see SC-002a.)
3. **Given** a ranking output, **When** the user serialises it to JSON, **Then** the field names and structure match the contract consumed by the existing `panel_chatbot_v3.py` so that no changes to the diagnostic panel are required.

---

### Edge Cases

- A detector returns predictions whose class labels are outside the domain taxonomy `C_d`. Resolution: predictions are filtered before TP/FP/FN accounting (they do not become false positives) and the total number of dropped predictions is exposed in a diagnostic `out_of_domain_filtered` field on `PrivacyMetric`. This preserves the paper's model-agnostic semantics while keeping detector noisiness observable.
- A detector returns overlapping predictions for the same text region. Resolution: keep the highest-confidence prediction, drop the rest (greedy non-maximum suppression as in the sandbox).
- A predicted span partially overlaps a ground-truth span. Resolution: IoU ≥ `iou_threshold` counts as TP, below the threshold counts as FP + FN. The threshold is a single per-domain scalar exposed on `PrivacyDomainConfig` with default `0.50`; it is not per-class.
- A domain class has zero ground-truth instances in the corpus. Resolution: `recall` and `f2` for that class are `0.0` via the `0/0` guard, and the class still consumes its full weight `w_c` in the `DetectionScore` sum. This is faithful to the paper's formula `Σ_c w_c · f2_c` over the complete `C_d` and surfaces the cost of an under-representative corpus rather than hiding it through weight renormalisation. Users who do not want a class to penalise the score must remove it from the `PrivacyDomainConfig` at construction time.
- A detector raises at predict time on a given input. Resolution: `Privacy.run` (single detector) propagates the exception — a failed detector aborts that evaluation. `PrivacyRanker.run` (multi-detector) catches per-detector exceptions, records a failed `PrivacyMetric` entry with `success = False`, the exception message, `score_100 = 0.0`, and continues with the remaining detectors. This matches the sandbox's `evaluate_all_models` behaviour and lets a ranking complete even when one backend is broken.

## Requirements

### Functional Requirements

- **FR-001**: A `Privacy` metric MUST be implemented as a subclass of `Gaussia`, evaluating exactly one injected `PIIDetector` against the corpus delivered by the configured `Retriever` and emitting one `PrivacyMetric` in `self.metrics` per evaluated dataset.

- **FR-002**: A `PrivacyRanker` metric MUST be implemented as a subclass of `Gaussia`, accepting a list of `PIIDetector` instances and emitting one `PrivacyRanking` in `self.metrics` containing the ordered per-detector `PrivacyMetric` results.

- **FR-003**: A `PIIDetector` abstract base class MUST define `predict(text: str) -> list[Span]` and `supported_classes -> frozenset[str]`. Concrete adapters for Microsoft Presidio and HuggingFace token-classification pipelines MUST be provided.

- **FR-004**: A `PrivacyDomainConfig` Pydantic model MUST capture the domain taxonomy `C_d`, the criticality weights `w_{c,d}`, and the severity weights `ρ_{c,d}`, with validators enforcing that both weight maps sum to `1.0 ± 1e-9` and that every class appearing in the weights also appears in the taxonomy.

- **FR-005**: Each `PIIDetector` instance MUST carry its own `domain_fit ∈ [0, 1]` and `regulatory_fit ∈ [0, 1]` scalars, supplied by the user at construction. Defaults are not provided by the library; omission MUST raise a `ValidationError`.

- **FR-006**: The metric MUST compute, per class `c`: `TP_c`, `FP_c`, `FN_c`, `precision_c`, `recall_c`, `f2_c`, `fn_rate_c`, using IoU-based span matching. The IoU threshold is read from `PrivacyDomainConfig.iou_threshold` (default `0.50`, range `(0, 1]`) and applies uniformly across all classes.

- **FR-007**: The metric MUST compute `DetectionScore = Σ_c w_{c,d} · f2_c` across the full domain taxonomy (including classes with zero predictions and zero ground truth, which contribute 0 to the sum).

- **FR-008**: The metric MUST compute `Coverage = |C_M ∩ C_d| / |C_d|` where `C_M` is the set returned by `detector.supported_classes`.

- **FR-009**: The metric MUST compute `Penalty_FN = max(0, 1 − Σ_c ρ_{c,d} · fn_rate_c)`.

- **FR-010**: The metric MUST compute `Score = DetectionScore · Coverage · DomainFit · RegulatoryFit · Penalty_FN` and expose both the raw `score ∈ [0, 1]` and `score_100 = 100 · score`.

- **FR-011**: The metric MUST compute `R1 = max_c (ρ_{c,d} · fn_rate_c)` and report the class achieving the maximum.

- **FR-012**: The metric MUST compute `R2 = 1 − (Penalty_FN · Coverage)` and `R_final = 1 − (1 − R1) · (1 − R2)`.

- **FR-013**: The `PrivacyMetric` schema MUST include the full breakdown required by the paper's Section "Implementation Considerations": detector identifier, applied domain config, supported classes, covered classes, missing classes, per-class metrics, per-class DetectionScore contributions, per-class CriticalFN contributions, R1 / R2 / R_final, Score / Score_100, and qualitative interpretation per the paper's Table 3.

- **FR-014**: The `PrivacyRanking` schema MUST contain the ordered list of `PrivacyMetric` results and expose the winning detector identifier, with field names compatible with the existing `chatbot_v3_results.json` shape consumed by `panel_chatbot_v3.py` so that the diagnostic panel works without modification.

- **FR-015**: Predictions whose labels fall outside `C_d` MUST be filtered before TP/FP/FN accounting and counted in a diagnostic `out_of_domain_filtered` integer field on `PrivacyMetric` — they MUST NOT contribute to FP.

- **FR-015a**: For each domain class `c ∈ C_d` with zero ground-truth instances and zero predictions in the corpus, `recall_c`, `precision_c`, and `f2_c` MUST evaluate to `0.0` via a `0/0` guard, and the class MUST retain its full weight `w_c` in the `DetectionScore` aggregation. Renormalising weights over present-only classes is forbidden.

- **FR-016**: Overlapping predictions over the same text span MUST be resolved by greedy non-maximum suppression on confidence score before matching against ground truth.

- **FR-017**: The library MUST NOT auto-install any runtime dependency, MUST NOT use `try/except ImportError`, and MUST NOT perform dynamic imports. Optional detector backends (Presidio, transformers) MUST be declared as extras in `pyproject.toml` and surface a clear `ImportError` at the import site of the corresponding adapter module.

- **FR-018**: HTML or other visualisation generation MUST NOT live inside `src/gaussia/`. The existing `panel_chatbot_v3.py` and `chatbot_v3_results_panel_v3.html` remain external tools that consume the metric's JSON output.

- **FR-019**: The metric MUST NOT incorporate latency or infrastructure measurements into the Score (the paper's v3 explicitly removed `InfraScore`). However, `Privacy` MUST measure and expose `load_time` (seconds, measured around the detector's initialisation hook or construction) and `inference_latency` (total seconds spent inside `detector.predict` across the corpus) as informational fields on `PrivacyMetric`. These fields preserve continuity with the sandbox JSON contract consumed by `panel_chatbot_v3.py` and MUST NOT enter any Score / Risk computation.

- **FR-020**: The Frequentist / Bayesian `StatisticalMode` strategy used by other pygaussia metrics is **not applicable** to this metric: Score and Risk are deterministic functions of class-level counts, with no per-batch distribution to aggregate. The metric MUST document this exclusion explicitly and MUST NOT accept a `statistical_mode` parameter.

### Key Entities

- **Span**: A region of text with a class label, character offsets `[start, end)`, surface text, and optional confidence score. Used in both ground truth and predictions.

- **PIIDetector** (abstract): A black-box adapter over a concrete detection backend. Carries the `supported_classes` advertised by the model card and the user-supplied `domain_fit` / `regulatory_fit` scalars. Concrete implementations: `PresidioDetector`, `HuggingFacePIIDetector`.

- **PrivacyBatch**: A subclass of the standard `Batch` adding `spans: list[Span]` for ground-truth annotations. The `query` field carries the input text shown to the detector; `assistant` and `ground_truth_assistant` remain empty strings (no conversational counterpart in this evaluation context).

- **PrivacyDomainConfig**: Frozen configuration object describing the evaluation domain: taxonomy, criticality weights, severity weights, IoU threshold, and the optional regulatory framework label.

- **PrivacyMetric**: Per-detector evaluation result containing every component required by the paper's output contract.

- **PrivacyRanking**: Aggregated ordering of `PrivacyMetric` results plus the winning detector identifier.

### SDK Pipeline Fit

- **New base class needed?**: Yes — `core/detector.py` introducing the `PIIDetector` abstraction (Adapter / Strategy). This is analogous to existing `Reranker`, `Embedder`, `Guardian` abstractions and is required by FR-003 so that the metric can run multiple detector backends without string branching.

- **New metric/component?**: Yes — `metrics/privacy.py` containing two `Gaussia` subclasses: `Privacy` (single-detector) and `PrivacyRanker` (multi-detector orchestration).

- **New schema?**: Yes — `schemas/privacy.py` containing `Span`, `PrivacyBatch`, `PrivacyDomainConfig`, `PrivacyMetric`, `PrivacyRanking`.

- **New strategy?**: Yes — `PIIDetector` is the strategy interface; `PresidioDetector` and `HuggingFacePIIDetector` are the initial concrete strategies, housed under a new `detectors/` module mirroring the existing `guardians/`, `rerankers/`, `embedders/` convention.

- **Existing patterns affected?**: None. All additions are open-closed extensions. The existing `Batch` schema is extended via subclassing (`PrivacyBatch`), not modified.

## Success Criteria

- **SC-001**: Every formula component is verified with a deterministic `StubDetector` whose predictions are fixed by the test. For a hand-constructed corpus and detector, the emitted `PrivacyMetric` reproduces values computed by hand from the paper's formulas — per-class `f2`, `detection_score`, `coverage`, `penalty_fn`, `r1_weakest_class_risk`, `r2_systemic_risk`, `r_final`, and `score_100` — within `1e-9`. A multi-class fixture reproduces the paper's worked example (`Score_100 = 28.36`) at reduced scale.

- **SC-002**: `PrivacyRanker.run` with multiple `StubDetector`s whose component scores are predetermined emits a `PrivacyRanking` ordered by `score_100` descending, with failed detectors at the tail and `winning_detector == results[0].name`.

- **SC-002a** (optional, opt-in): Reproducing the PR #10 sandbox artifact `chatbot_v3_results.json` — which was generated over the **500-turn** corpus (`chatbot_conversations_500.txt` / `chatbot_conversations_tagged_500.txt`, `corpus_samples = 4935`) with the six real detectors — is covered by an integration test gated behind `RUN_SANDBOX=1` and the optional extras. It is NOT part of the default `uv run pytest` run. The 100-turn `eval_*_100.txt` files do not correspond to this JSON and are not used as a baseline.

- **SC-003**: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src/gaussia` all pass without errors on the resulting branch.

- **SC-004**: The Pydantic schemas reject malformed input at construction time, demonstrated by at least one negative test per validation rule (weights summing to ≠ 1, fit scalars outside `[0, 1]`, spans with `start ≥ end`, unknown class in weight map).

- **SC-005**: Adding a hypothetical third detector backend requires creating a new adapter file under `detectors/` and zero modifications to `Privacy`, `PrivacyRanker`, or any existing schema (Open/Closed verification).

## Assumptions

- The user is responsible for producing the labelled corpus and for choosing the parsing format (inline `<TAG>value</TAG>`, JSON Lines, CoNLL, etc.) by implementing a `Retriever` subclass that yields `PrivacyBatch` instances. The library does not own any specific parser.

- `DomainFit` and `RegulatoryFit` are expert judgements per the paper; the library treats them as opaque scalars supplied by the user at detector construction and does not attempt to compute them.

- The HTML diagnostic panel (`panel_chatbot_v3.py`) is treated as a downstream consumer of the metric's JSON output, not as part of the library. It remains in `tests/privacy/` or moves to a future `tools/` location, but it is out of scope for this spec.

- The PR #10 sandbox script `Analisis_Score_Chatbot_v3.py` is the reference implementation of the formulas and the source of regression fixtures, but is not itself ported into `src/gaussia/`. Its taxonomy mappings, model registries, file parsers, and auto-installation logic are explicitly discarded.

- The paper's Section "Risk Index" contains LaTeX typos (`Risk*{WC}`, `\max*{c}`) that this spec interprets as the subscript forms `Risk_{WC}` and `\max_c`. The semantics are reconstructed from surrounding prose and verified against the sandbox implementation in `compute_score()` (lines 971–981 of `Analisis_Score_Chatbot_v3.py`).

- The InfraScore dimension present in earlier versions of the paper is permanently out of scope for this metric.

- `StatisticalMode` (Frequentist / Bayesian) is not relevant: the metric is a deterministic function of class-level counts and does not aggregate per-batch distributions.
