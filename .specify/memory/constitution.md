# Gaussia Python SDK — Constitution Extension

This extends the base Gaussia constitution (shipped with the `speckit` skill in `gaussia-labs/skills`). Read both together.

## Architecture: The Gaussia Pipeline

This is the canonical data flow. All metrics must respect it:

```
Retriever.load_dataset() -> list[Dataset] | Iterator[Dataset]
    |
Gaussia._process() iterates datasets
    |
Metric.batch() processes each conversation batch
    |
Results appended to self.metrics
```

### Patterns in Use

- **Template Method**: `Gaussia` base class defines `_process()` flow; subclasses implement `batch()`.
- **Strategy**: `FrequentistMode` / `BayesianMode` for statistical analysis; `SelectionStrategy` for generators.
- **Adapter**: Guardian implementations adapt different models behind a common `Guardian` interface.
- **Factory Method**: `MyMetric.run(RetrieverClass)` instantiates and orchestrates the pipeline.

## Module Boundaries

Each module has a clear responsibility. Cross-module dependencies flow downward:

- **`core/`**: Abstract base classes only. No concrete implementations.
- **`metrics/`**: Concrete metric implementations. Depend on `core/` and `schemas/`.
- **`schemas/`**: Pydantic models. No business logic.
- **`statistical/`**: Statistical analysis strategies. Depend only on `schemas/`.
- **`guardians/`**: Bias detection adapters. Depend on `core/`.
- **`loaders/`**: Dataset loaders. Depend on `core/`.
- **`llm/`**: LLM integration. Self-contained with its own schemas and prompts.
- **`extractors/`**: Group extraction. Depend on `core/`.
- **`generators/`**: Synthetic data generation. Depend on `core/` and `schemas/`.
- **`utils/`**: Utilities. No dependencies on other gaussia modules.

## Quality Tools

```bash
uv run pytest                  # Tests
uv run ruff check .            # Linter
uv run ruff format .           # Formatter
uv run mypy src/gaussia      # Type checker
```

## Python-Specific Rules

- `snake_case` for functions/variables, `PascalCase` for classes
- Type hints required on all function signatures
- Google-style docstrings
- Imports organized with isort (stdlib, third-party, local)
- No `try/except ImportError`, no `if TYPE_CHECKING` tricks for runtime, no dynamic imports

**Version**: 1.0.0 | **Ratified**: 2026-04-09
