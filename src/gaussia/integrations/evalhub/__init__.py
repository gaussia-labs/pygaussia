"""EvalHub provider integration for Gaussia."""

__all__ = [
    "BenchmarkExecution",
    "BenchmarkInput",
    "GaussiaEvalHubAdapter",
    "SUPPORTED_BENCHMARK_IDS",
    "load_benchmark_input",
    "run_gaussia_benchmark",
]


def __getattr__(name: str):
    if name == "GaussiaEvalHubAdapter":
        from .adapter import GaussiaEvalHubAdapter

        return GaussiaEvalHubAdapter
    if name in {"BenchmarkExecution", "SUPPORTED_BENCHMARK_IDS", "run_gaussia_benchmark"}:
        from . import benchmarks

        return getattr(benchmarks, name)
    if name in {"BenchmarkInput", "load_benchmark_input"}:
        from . import payloads

        return getattr(payloads, name)
    raise AttributeError(name)

