from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import TYPE_CHECKING, Any

from . import _warnings as _evalhub_warnings  # noqa: F401
from evalhub.adapter import EvaluationResult

from gaussia.guardians import IBMGranite
from gaussia.loaders import HurtlexLoader
from gaussia.metrics.bias import Bias
from gaussia.metrics.context import Context
from gaussia.metrics.conversational import Conversational
from gaussia.metrics.humanity import Humanity

from .payloads import BenchmarkInput, build_static_retriever

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import ProviderConfig


SUPPORTED_BENCHMARK_IDS = (
    "humanity",
    "context",
    "conversational",
    "bias",
    "toxicity",
)
MIN_INTERACTIONS_FOR_BIAS = 5
MIN_INTERACTIONS_FOR_TOXICITY = 5


@dataclass
class BenchmarkExecution:
    benchmark_id: str
    primary_metric_name: str
    primary_metric_value: float
    overall_score: float
    interaction_count: int
    metric_count: int
    evaluation_results: list[EvaluationResult]
    artifact_payload: dict[str, object]


@dataclass(frozen=True)
class BenchmarkContext:
    benchmark_id: str
    benchmark_input: BenchmarkInput
    provider_id: str
    config: ProviderConfig
    retriever_cls: Any
    interaction_count: int


def run_gaussia_benchmark(
    *,
    benchmark_id: str,
    benchmark_input: BenchmarkInput,
    provider_id: str,
    config: ProviderConfig,
) -> BenchmarkExecution:
    runner = _BENCHMARK_RUNNERS.get(benchmark_id)
    if runner is None:
        raise ValueError(f"Unsupported Gaussia benchmark: {benchmark_id}")

    dataset = benchmark_input.dataset
    context = BenchmarkContext(
        benchmark_id=benchmark_id,
        benchmark_input=benchmark_input,
        provider_id=provider_id,
        config=config,
        retriever_cls=build_static_retriever(dataset),
        interaction_count=len(dataset.conversation),
    )
    return runner(context)


def _run_humanity(context: BenchmarkContext) -> BenchmarkExecution:
    metrics = Humanity.run(context.retriever_cls)
    if not metrics:
        raise ValueError("Gaussia returned no Humanity metrics")
    primary = round(fmean(metric.humanity_assistant_emotional_entropy for metric in metrics), 6)
    return _build_execution(
        context=context,
        primary_metric_name="humanity_assistant_emotional_entropy",
        primary_metric_value=primary,
        metric_count=len(metrics),
        evaluation_results=[_float_result("humanity_assistant_emotional_entropy", primary)],
        metrics=metrics,
        summary={"emotional_entropy_mean": primary},
    )


def _run_context(context: BenchmarkContext) -> BenchmarkExecution:
    metric = _single_metric(
        Context.run(context.retriever_cls, **_judge_metric_kwargs(context.config)),
        context.benchmark_id,
    )
    primary = round(metric.context_awareness, 6)
    return _build_execution(
        context=context,
        primary_metric_name="context_awareness",
        primary_metric_value=primary,
        metric_count=1,
        evaluation_results=[_float_result("context_awareness", primary)],
        metrics=[metric],
        summary={
            "n_interactions": metric.n_interactions,
            "context_awareness": primary,
            "context_awareness_ci_low": metric.context_awareness_ci_low,
            "context_awareness_ci_high": metric.context_awareness_ci_high,
        },
    )


def _run_conversational(context: BenchmarkContext) -> BenchmarkExecution:
    metric = _single_metric(
        Conversational.run(context.retriever_cls, **_judge_metric_kwargs(context.config)),
        context.benchmark_id,
    )
    primary = round(metric.conversational_sensibleness.mean, 6)
    return _build_execution(
        context=context,
        primary_metric_name="conversational_sensibleness",
        primary_metric_value=primary,
        metric_count=1,
        evaluation_results=[
            _float_result("conversational_memory", metric.conversational_memory.mean),
            _float_result("conversational_language", metric.conversational_language.mean),
            _float_result("conversational_quality_maxim", metric.conversational_quality_maxim.mean),
            _float_result("conversational_quantity_maxim", metric.conversational_quantity_maxim.mean),
            _float_result("conversational_relation_maxim", metric.conversational_relation_maxim.mean),
            _float_result("conversational_manner_maxim", metric.conversational_manner_maxim.mean),
            _float_result("conversational_sensibleness", primary),
        ],
        metrics=[metric],
        summary={
            "n_interactions": metric.n_interactions,
            "conversational_memory": metric.conversational_memory.mean,
            "conversational_language": metric.conversational_language.mean,
            "conversational_quality_maxim": metric.conversational_quality_maxim.mean,
            "conversational_quantity_maxim": metric.conversational_quantity_maxim.mean,
            "conversational_relation_maxim": metric.conversational_relation_maxim.mean,
            "conversational_manner_maxim": metric.conversational_manner_maxim.mean,
            "conversational_sensibleness": primary,
        },
    )


def _run_bias(context: BenchmarkContext) -> BenchmarkExecution:
    if context.interaction_count < MIN_INTERACTIONS_FOR_BIAS:
        raise ValueError(f"Bias benchmark requires at least {MIN_INTERACTIONS_FOR_BIAS} interactions")
    metric = _single_metric(
        Bias.run(
            context.retriever_cls,
            guardian=IBMGranite,
            protected_attributes=context.config.build_bias_protected_attributes(),
            config=context.config.require_guardian_config(),
        ),
        context.benchmark_id,
    )
    attribute_rates = {rate.protected_attribute: round(rate.rate, 6) for rate in metric.attribute_rates}
    primary = context.config.build_bias_primary_score(list(attribute_rates.values()))
    return _build_execution(
        context=context,
        primary_metric_name="bias_score",
        primary_metric_value=primary,
        metric_count=1,
        evaluation_results=[_float_result("bias_score", primary)]
        + [_float_result(f"bias_rate_{name}", value) for name, value in sorted(attribute_rates.items())],
        metrics=[metric],
        summary={
            "attribute_rates": attribute_rates,
            "attribute_count": len(attribute_rates),
            "bias_score": primary,
        },
    )


def _run_toxicity(context: BenchmarkContext) -> BenchmarkExecution:
    if context.interaction_count < MIN_INTERACTIONS_FOR_TOXICITY:
        raise ValueError(f"Toxicity benchmark requires at least {MIN_INTERACTIONS_FOR_TOXICITY} interactions")
    _disable_numba_caching_for_toxicity()
    from gaussia.metrics.toxicity import Toxicity

    metric = _single_metric(
        Toxicity.run(
            context.retriever_cls,
            embedder=context.config.build_toxicity_embedder(),
            toxicity_loader=HurtlexLoader,
            statistical_mode=context.config.build_toxicity_statistical_mode(),
            group_prototypes=context.config.toxicity_group_prototypes,
            group_default_threshold=context.config.toxicity_group_default_threshold,
            toxicity_min_cluster_size=min(context.config.toxicity_min_cluster_size, context.interaction_count),
            toxicity_cluster_selection_epsilon=context.config.toxicity_cluster_selection_epsilon,
            toxicity_cluster_selection_method=context.config.toxicity_cluster_selection_method,
            toxicity_cluster_use_latent_space=context.config.toxicity_cluster_use_latent_space,
            umap_n_neighbors=min(context.config.toxicity_umap_n_neighbors, context.interaction_count - 1),
            umap_n_components=context.config.toxicity_umap_n_components,
            umap_min_dist=context.config.toxicity_umap_min_dist,
            umap_metric=context.config.toxicity_umap_metric,
            w_DR=context.config.toxicity_w_dr,
            w_ASB=context.config.toxicity_w_asb,
            w_DTO=context.config.toxicity_w_dto,
        ),
        context.benchmark_id,
    )
    summary = _build_toxicity_summary(metric)
    return _build_execution(
        context=context,
        primary_metric_name="toxicity_didt",
        primary_metric_value=summary["toxicity_didt"],
        metric_count=1,
        evaluation_results=[
            _float_result("toxicity_didt", summary["toxicity_didt"]),
            _float_result("toxicity_dr", summary["toxicity_dr"]),
            _float_result("toxicity_asb", summary["toxicity_asb"]),
            _float_result("toxicity_dto", summary["toxicity_dto"]),
        ],
        metrics=[metric],
        summary=summary,
    )


def build_benchmark_artifact_payload(
    *,
    benchmark_input: BenchmarkInput,
    provider_id: str,
    benchmark_id: str,
    metrics: list[Any],
    metric_count: int,
    interaction_count: int,
    summary: dict[str, object],
    generated_at: datetime | None = None,
) -> dict[str, object]:
    timestamp = generated_at or datetime.now(UTC)
    metadata = benchmark_input.metadata
    return {
        "generated_at": timestamp.isoformat(),
        "provider_id": provider_id,
        "benchmark_id": benchmark_id,
        "assistant_id": metadata.get("assistant_id", benchmark_input.dataset.assistant_id),
        "session_id": metadata.get("session_id", benchmark_input.dataset.session_id),
        "stream_id": metadata.get("stream_id", ""),
        "control_id": metadata.get("control_id", ""),
        "agentspace_id": metadata.get("agentspace_id", ""),
        "payload_source": benchmark_input.payload_source,
        "metric_count": metric_count,
        "interaction_count": interaction_count,
        "summary": summary,
        "metadata": metadata,
        "metrics": [metric.model_dump(mode="json") for metric in metrics],
    }


def _judge_metric_kwargs(config: ProviderConfig) -> dict[str, object]:
    return {
        "model": config.require_judge_model(),
        "use_structured_output": config.judge_use_structured_output,
        "bos_json_clause": config.judge_bos_json_clause,
        "eos_json_clause": config.judge_eos_json_clause,
    }


def _build_execution(
    *,
    context: BenchmarkContext,
    primary_metric_name: str,
    primary_metric_value: float,
    metric_count: int,
    evaluation_results: list[EvaluationResult],
    metrics: list[Any],
    summary: dict[str, object],
) -> BenchmarkExecution:
    primary = round(float(primary_metric_value), 6)
    return BenchmarkExecution(
        benchmark_id=context.benchmark_id,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary,
        overall_score=primary,
        interaction_count=context.interaction_count,
        metric_count=metric_count,
        evaluation_results=evaluation_results,
        artifact_payload=build_benchmark_artifact_payload(
            benchmark_input=context.benchmark_input,
            provider_id=context.provider_id,
            benchmark_id=context.benchmark_id,
            metrics=metrics,
            metric_count=metric_count,
            interaction_count=context.interaction_count,
            summary=summary,
        ),
    )


def _single_metric(metrics: list[Any], benchmark_id: str) -> Any:
    if not metrics:
        raise ValueError(f"Gaussia returned no metrics for benchmark {benchmark_id}")
    return metrics[0]


def _float_result(metric_name: str, metric_value: float) -> EvaluationResult:
    return EvaluationResult(
        metric_name=metric_name,
        metric_value=round(float(metric_value), 6),
        metric_type="float",
    )


def _build_toxicity_summary(metric: Any) -> dict[str, object]:
    group_profiling = _jsonish(metric.group_profiling)
    mode = str(group_profiling.get("mode", "frequentist"))
    if mode == "bayesian":
        summary = _jsonish(group_profiling.get("bayesian")).get("summary", {})
        didt = _summary_mean(summary, "DIDT")
        dr = _summary_mean(summary, "DR")
        asb = _summary_mean(summary, "ASB")
        dto = _summary_mean(summary, "DTO")
    else:
        frequentist = _jsonish(group_profiling.get("frequentist"))
        didt = round(float(frequentist.get("DIDT", 0.0)), 6)
        dr = round(float(frequentist.get("DR", 0.0)), 6)
        asb = round(float(frequentist.get("ASB", 0.0)), 6)
        dto = round(float(frequentist.get("DTO", 0.0)), 6)

    return {
        "mode": mode,
        "group_count": len(group_profiling.get("groups", [])),
        "cluster_count": len(metric.cluster_profiling),
        "toxicity_didt": didt,
        "toxicity_dr": dr,
        "toxicity_asb": asb,
        "toxicity_dto": dto,
    }


def _summary_mean(summary: dict[str, Any], key: str) -> float:
    item = _jsonish(summary).get(key, {})
    item = _jsonish(item)
    return round(float(item.get("mean", 0.0)), 6)


def _jsonish(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _disable_numba_caching_for_toxicity() -> None:
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc.ufuncbuilder import UFuncDispatcher

    if getattr(Dispatcher.enable_caching, "__name__", "") != "_disable_cache":

        def _disable_cache(_self) -> None:
            return None

        Dispatcher.enable_caching = _disable_cache
        UFuncDispatcher.enable_caching = _disable_cache


_BENCHMARK_RUNNERS: dict[str, Callable[[BenchmarkContext], BenchmarkExecution]] = {
    "humanity": _run_humanity,
    "context": _run_context,
    "conversational": _run_conversational,
    "bias": _run_bias,
    "toxicity": _run_toxicity,
}
