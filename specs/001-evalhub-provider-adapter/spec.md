# Feature Specification: EvalHub Provider Adapter

**Feature Branch**: `feat/evalhub-provider-adapter`
**Created**: 2026-05-11
**Status**: Draft
**Input**: Move the EvalHub BYOF provider adapter into `pygaussia` as an optional integration.

## User Scenarios & Testing

### Primary User Story

As an EvalHub operator, I can install `gaussia[evalhub]`, register the Gaussia provider adapter, submit an EvalHub job with a Gaussia-compatible dataset, and receive benchmark results without depending on a bridge-specific provider module.

### Acceptance Scenarios

1. Given a job with `parameters.dataset` and `parameters.metadata`, when EvalHub runs `python -m gaussia.integrations.evalhub.adapter`, then the adapter evaluates the requested benchmark and returns `JobResults`.
2. Given a legacy job with `parameters.context_persistance`, when the adapter runs, then it converts the payload into a Gaussia dataset and marks the payload source as legacy.
3. Given a judge-backed benchmark, when the user injects a LangChain connector or configures one by environment, then the adapter uses that connector without requiring `langchain-openai`.
4. Given run logging configuration, when the adapter completes, then it logs metrics through the configured run logger; if no logger is configured, evaluation still succeeds.
5. Given MLflow environment variables, when no custom run logger is configured, then MLflow is used as the default run logger implementation.

## Requirements

### Functional Requirements

- **FR-001**: The integration MUST expose `GaussiaEvalHubAdapter` under `gaussia.integrations.evalhub`.
- **FR-002**: The integration MUST be executable with `python -m gaussia.integrations.evalhub.adapter`.
- **FR-003**: The integration MUST support `humanity`, `context`, `conversational`, `agentic`, `bias`, and `toxicity`.
- **FR-004**: The `evalhub` extra MUST include EvalHub adapter dependencies and MUST NOT require a provider-specific LangChain package.
- **FR-005**: Judge-backed benchmarks MUST accept any LangChain-compatible chat connector supplied by the caller or configured through environment.
- **FR-006**: The preferred payload contract MUST be `JobSpec.parameters.dataset` plus `JobSpec.parameters.metadata`.
- **FR-007**: The legacy `JobSpec.parameters.context_persistance` payload MUST remain supported as fallback.
- **FR-008**: The adapter MUST support OCI artifact creation through EvalHub callbacks.
- **FR-009**: Run logging MUST depend on an interface/protocol; MLflow MUST be an implementation, not a direct adapter dependency.
- **FR-010**: When `MLFLOW_*` variables are set and no custom run logger factory is configured, the default run logger SHOULD write MLflow runs.

### Non-Functional Requirements

- **NFR-001**: Core `gaussia` imports MUST NOT import EvalHub.
- **NFR-002**: Provider-specific LLM packages MUST remain user-selected optional dependencies.
- **NFR-003**: The adapter MUST preserve existing `GAUSSIA_*` and `MLFLOW_*` environment variable compatibility where possible.
- **NFR-004**: The implementation MUST follow Gaussia's existing `Retriever -> Gaussia._process -> Metric.batch -> metrics` pipeline.

## Entities

- **EvalHub JobSpec**: External job contract with benchmark, model, payload, callbacks, and exports.
- **BenchmarkInput**: Normalized adapter input containing a Gaussia dataset, metadata, and payload source.
- **ProviderConfig**: Environment and injected runtime configuration for judge, guardian, toxicity, and benchmark options.
- **RunLogger**: Protocol for optional run telemetry logging.
- **MLflowRunLogger**: Default RunLogger implementation for MLflow-compatible tracking servers.

## Assumptions

- EvalHub SDK remains an optional dependency under `gaussia[evalhub]`.
- Users install their chosen LangChain provider package separately when judge-backed benchmarks need it.
- The misspelled legacy key `context_persistance` remains supported for compatibility only.
- Infrastructure extension loading is scoped to EvalHub integration configuration and does not affect core metric execution.
