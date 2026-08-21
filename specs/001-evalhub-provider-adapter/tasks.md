# Tasks: EvalHub Provider Adapter

**Input**: `specs/001-evalhub-provider-adapter/spec.md`
**Plan**: `specs/001-evalhub-provider-adapter/plan.md`

## Phase 1: Adapter Integration

- [x] T001 Add `gaussia.integrations.evalhub` package.
- [x] T002 Add executable `GaussiaEvalHubAdapter`.
- [x] T003 Add benchmark dispatch for `humanity`, `context`, `conversational`, `agentic`, `bias`, and `toxicity`.
- [x] T004 Add OCI artifact writer through EvalHub callbacks.

## Phase 2: Payload Contract

- [x] T005 Support preferred `parameters.dataset` plus `parameters.metadata`.
- [x] T006 Support legacy `parameters.context_persistance`.
- [x] T007 Prefer `dataset` when both payload shapes are present.
- [x] T008 Add tests for preferred, legacy, invalid, and benchmark-selection payloads.

## Phase 3: Configuration

- [x] T009 Move judge, guardian, toxicity, and benchmark settings into `ProviderConfig`.
- [x] T010 Remove hard dependency on `langchain-openai`.
- [x] T011 Support injected judge connectors.
- [x] T012 Support connector class configuration through `GAUSSIA_JUDGE_CONNECTOR_CLASS`.
- [x] T013 Support LangChain provider registry configuration through `GAUSSIA_JUDGE_MODEL_PROVIDER`.

## Phase 4: Run Logging

- [x] T014 Introduce run logging protocol for adapter telemetry.
- [x] T015 Keep MLflow as the default implementation when `MLFLOW_*` variables are present.
- [x] T016 Support custom run logger factories through `GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY`.
- [x] T017 Add tests for run logger selection and MLflow request construction.

## Phase 5: Packaging and Documentation

- [x] T018 Add `evalhub` optional extra.
- [x] T019 Keep EvalHub imports out of core `gaussia`.
- [x] T020 Document EvalHub install without provider-specific LangChain dependencies.
- [x] T021 Add spec, plan, and tasks markdowns for SDD traceability.

## Phase 6: Validation

- [x] T022 Run EvalHub integration unit tests.
- [x] T023 Run focused lint checks.
