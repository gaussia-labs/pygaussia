# Tasks: Domain-Adjusted Privacy Detection Metric and Ranker

**Input**: `specs/001-privacy-metric/plan.md`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story — US1 (evaluate a single detector), US2 (rank multiple detectors)
- **TDD**: Phase 2 tests are written and **verified FAILING** before any Phase 3+ implementation. Traceability tags (FR-xxx / SC-xxx) point back to `spec.md`.

## Phase 1: Schema & Contracts

- [ ] T001 [P] Add optional extras to `pyproject.toml`: `privacy-presidio = ["presidio-analyzer>=2.2", "spacy>=3.7"]`, `privacy-huggingface = ["transformers>=4.40", "torch>=2.1"]`; no new mandatory runtime deps. (FR-017)
- [ ] T002 [P] Create package markers: `tests/detectors/__init__.py`, `tests/fixtures/privacy/__init__.py`.
- [ ] T003 [P] [US1] Implement all Pydantic schemas in `src/gaussia/schemas/privacy.py`: `Span`, `PrivacyBatch(Batch)`, `PrivacyDomainConfig`, `ClassMetrics`, `DetectionScoreContribution`, `CriticalFNContribution`, `PrivacyMetric(BaseMetric)`, `PrivacyRanking(BaseMetric)` — including validators, `@computed_field` `*_100` derivations, `interpretation` Literal, and the `success=False` short-circuit. (FR-004, FR-013, FR-014)
- [ ] T004 [P] [US1] Implement abstract `PIIDetector` in `src/gaussia/core/detector.py` (`predict`, `supported_classes -> frozenset[str]`, required `domain_fit`/`regulatory_fit`, no-op `setup()`); re-export from `core/__init__.py` and add to `__all__`. (FR-003, FR-005)
- [ ] T005 [US1] Document `from gaussia.schemas.privacy import ...` in `schemas/__init__.py` docstring (no eager import, matching repo convention). (plan §Modified Files)

## Phase 2: Tests (Red Phase)

- [ ] T006 [P] [US1] Schema validation tests in `tests/metrics/test_privacy_schemas.py`: `Span` (`start<end`, score/offset bounds); `PrivacyDomainConfig` (weights sum `1.0±1e-9`, keys == classes, iou range, frozen); contribution product-equality; `PrivacyMetric` cross-field validators + `success=False` skip path; `PrivacyRanking` ordering. (SC-004, FR-004, FR-011, FR-013, FR-014)
- [ ] T007 [P] [US1] `PIIDetector` contract test in `tests/detectors/test_detector_contract.py`: subclass must implement `predict`/`supported_classes`; missing `domain_fit`/`regulatory_fit` raises; `setup()` default no-op. (FR-003, FR-005)
- [ ] T008 [P] [US1] Implement deterministic `StubDetector` in `tests/fixtures/privacy/stub_detector.py` (predictions fixed by test, no external deps) — primary verification double. (SC-001)
- [ ] T009 [P] [US1] `Privacy` accounting tests in `tests/metrics/test_privacy.py` (StubDetector): IoU TP/FP/FN at config threshold; precision/recall/f2/fn_rate; `0/0` guard keeps full weight, no renormalisation; out-of-domain filter (counted, never FP); greedy NMS by confidence. (FR-006, FR-007, FR-015, FR-015a, FR-016, SC-001)
- [ ] T010 [P] [US1] `Privacy` score-pipeline tests vs hand-computed values: DetectionScore, Coverage, Penalty_FN, R1 (+weakest class), R2, R_final, Score/Score_100, interpretation band; multi-class fixture reproducing `Score_100 = 28.36`. (FR-007–FR-013, SC-001)
- [ ] T011 [P] [US1] `Privacy` latency + exclusion tests: `load_time`/`inference_latency` populated and `ge=0`, never enter Score/Risk; no `statistical_mode` parameter accepted; fail-fast on detector exception. (FR-019, FR-020)
- [ ] T012 [P] [US2] `PrivacyRanker` tests in `tests/metrics/test_privacy_ranker.py` (multiple StubDetectors): ordered `score_100` desc, `winning_detector == results[0].name`; fail-soft records `PrivacyMetric(success=False, error=...)` at tail; all-fail → `winning_detector is None`. (FR-002, FR-014, SC-002)
- [ ] T013 [P] [US1] Adapter contract tests with `pytest.importorskip`: `tests/detectors/test_presidio.py` (domain-projected spans, classes from loaded recognizers) and `tests/detectors/test_huggingface.py` (id2label via canonicaliser). (FR-003)
- [ ] T014 [P] [US2] Open/Closed test (SC-005): a new dummy adapter under `detectors/` plugs into `Privacy`/`PrivacyRanker` with zero edits to existing classes/schemas. (SC-005)

**Checkpoint**: All Phase 2 tests written and FAILING for the right reason.

## Phase 3: Implementation (Green Phase) — User Story 1 (P1)

- [ ] T015 [US1] Implement `Privacy(Gaussia)` in `src/gaussia/metrics/privacy.py`: `batch` via Template Method, injected `PIIDetector`, IoU matcher + NMS + OOD filter, full score/risk computation, latency capture around `setup()`/`predict`. (FR-001, FR-006–FR-013, FR-015, FR-015a, FR-016, FR-019)
- [ ] T016 [P] [US1] Implement pure label-canonicaliser in `src/gaussia/detectors/_label_canonicaliser.py` (translate sandbox `LABEL_ALIASES` + `canonicalize`, no I/O). (data-model)
- [ ] T017 [US1] Implement `PresidioDetector` in `src/gaussia/detectors/presidio.py` (AnalyzerEngine + blank spaCy; clean `ImportError` at import site, no try/except). (FR-003, FR-017)
- [ ] T018 [US1] Implement `HuggingFacePIIDetector` in `src/gaussia/detectors/huggingface.py` (token-classification pipeline + T016 canonicaliser; clean `ImportError` at import site). (FR-003, FR-017)
- [ ] T019 [US1] Re-export adapters from `src/gaussia/detectors/__init__.py`; re-export `Privacy` from `metrics/__init__.py` and extend `__all__`.

**Checkpoint**: User Story 1 independently functional — T006–T011, T013 pass.

## Phase 4: Implementation (Green Phase) — User Story 2 (P2)

- [ ] T020 [US2] Implement `PrivacyRanker(Gaussia)` in `src/gaussia/metrics/privacy.py`: composes `list[PIIDetector]`, per-detector fail-soft handler, ranking; re-export `PrivacyRanker` from `metrics/__init__.py` and extend `__all__`. (FR-002, FR-014)

**Checkpoint**: US1 + US2 independently functional — T012, T014 pass.

## Phase 5: Optional sandbox integration (opt-in, excluded from default suite)

- [ ] T021 [P] [US2] `tests/fixtures/privacy/sandbox_corpus.py` — parser for the **500-turn** corpus into `list[Dataset[PrivacyBatch]]`. (plan §New Files)
- [ ] T022 [P] [US2] Pin `tests/fixtures/privacy/sandbox_results.json` (copy of the 500-turn `chatbot_v3_results.json`). (plan §New Files)
- [ ] T023 [US2] `tests/integration/test_privacy_sandbox.py` reproducing the sandbox JSON: `@pytest.mark.slow`, skipped unless extras installed AND `RUN_SANDBOX=1`; confirm excluded from default `uv run pytest`. (SC-002a)

## Phase 6: Polish

- [ ] T024 `uv run ruff check .` passes.
- [ ] T025 `uv run ruff format --check .` passes.
- [ ] T026 `uv run mypy src/gaussia` passes.
- [ ] T027 `uv run pytest` (default suite) green. (SC-003)
- [ ] T028 Append the new metric under "Unreleased" in `CHANGELOG.md`. (plan §Modified Files)
- [ ] T029 Confirm no visualisation/HTML generation lives under `src/gaussia/`; the diagnostic panel stays an external consumer of the metric's JSON. (FR-018)

## Dependencies

- Phase 1 (Schema & Contracts) before Phase 2 (Tests).
- Phase 2 tests written and FAILING before Phase 3+ implementation (TDD).
- US1 (Phase 3) is independent and ships first; US2 (Phase 4) depends only on the shared schemas/contracts (Phase 1) and the `PrivacyMetric` impl path exercised by US1.
- Phase 5 (sandbox) depends on US1 + US2; it is opt-in and never gates the default suite.
- Phase 6 (Polish) depends on all stories complete.
