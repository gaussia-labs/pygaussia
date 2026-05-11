# Implementation Plan: EvalHub Provider Adapter

**Branch**: `feat/evalhub-provider-adapter`
**Spec**: `specs/001-evalhub-provider-adapter/spec.md`

## Summary

Add EvalHub BYOF support as an optional `pygaussia` integration. The adapter normalizes EvalHub payloads into Gaussia datasets, dispatches supported benchmarks through existing Gaussia metrics, returns EvalHub `JobResults`, writes optional artifacts, and logs runs through a pluggable logging protocol.

## Technical Context

- **Language**: Python 3.11+
- **Package**: `gaussia`
- **Optional extra**: `gaussia[evalhub]`
- **External SDK**: `eval-hub-sdk[adapter]`
- **Primary modules**:
  - `src/gaussia/integrations/evalhub/adapter.py`
  - `src/gaussia/integrations/evalhub/benchmarks.py`
  - `src/gaussia/integrations/evalhub/config.py`
  - `src/gaussia/integrations/evalhub/payloads.py`
  - `src/gaussia/integrations/evalhub/run_logging.py`
  - `src/gaussia/integrations/evalhub/mlflow_runs.py`

## Constitution Check

- **Pipeline**: Benchmarks call existing metric `run()` methods with a generated `Retriever` class, preserving `Retriever.load_dataset() -> Gaussia._process() -> Metric.batch() -> self.metrics`.
- **Module boundaries**: EvalHub-specific code stays under `integrations/evalhub`; core modules do not import EvalHub.
- **Dependency inversion**: Judge models and run logging are injected/configured through abstractions instead of hardcoded provider implementations.
- **Optional dependencies**: EvalHub dependencies stay behind the `evalhub` extra; provider-specific LangChain packages are user-selected.

## Design

### Payload Normalization

`payloads.py` converts either:

- preferred `parameters.dataset` plus `parameters.metadata`, or
- legacy `parameters.context_persistance`

into `BenchmarkInput`.

### Benchmark Dispatch

`benchmarks.py` maps EvalHub benchmark IDs to Gaussia metric runners. Each runner returns a `BenchmarkExecution` containing EvalHub metric results plus an artifact payload.

### Judge Configuration

`ProviderConfig.require_judge_model()` resolves judge-backed benchmark connectors in this order:

1. injected `judge_connector`
2. `GAUSSIA_JUDGE_CONNECTOR_CLASS` plus optional JSON kwargs
3. LangChain `init_chat_model()` using `GAUSSIA_JUDGE_MODEL` and optional `GAUSSIA_JUDGE_MODEL_PROVIDER`

This keeps OpenAI, Groq, Anthropic, local serving, and other LangChain providers outside the Gaussia dependency graph.

### Run Logging

The adapter depends on `EvalHubRunLogger` / `EvalHubRunContext` protocols. A custom logger can be configured with `GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY`. If no custom logger is provided, `MLFLOW_*` variables enable the bundled `MLflowRunLogger`.

## Verification

- Unit tests cover supported benchmark dispatch, legacy payloads, preferred payloads, OCI artifacts, unsupported benchmarks, invalid payloads, judge connector configuration, run logger configuration, and MLflow request construction.
- Linting covers integration modules and tests.
