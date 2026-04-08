---
name: metric-creator
description: Create new Gaussia metrics with proper structure, schema, tests, and fixtures. Use when adding a new evaluation metric to pygaussia.
argument-hint: [metric-name] [optional description]
---

# Gaussia Metric Creator

Create new metrics for the Gaussia AI evaluation library. This skill generates all required files following the established patterns.

## Usage

```
/metric-creator [metric-name] [optional description]
```

Examples:
```
/metric-creator safety "Evaluate AI response safety and harmlessness"
/metric-creator coherence "Measure logical coherence in multi-turn conversations"
/metric-creator factuality
```

## Files to Create

For a new metric called `{MetricName}`:

| File | Purpose |
|------|---------|
| `src/pygaussia/metrics/{metric_name}.py` | Metric implementation |
| `src/pygaussia/schemas/{metric_name}.py` | Pydantic schema for results |
| `tests/metrics/test_{metric_name}.py` | Unit tests |
| `tests/fixtures/mock_data.py` | Add `create_{metric_name}_dataset()` |
| `tests/fixtures/mock_retriever.py` | Add `{MetricName}DatasetRetriever` |
| `pyproject.toml` | Add optional dependency group |

### For LLM-Judge Metrics (additional files)

| File | Purpose |
|------|---------|
| `src/pygaussia/llm/schemas.py` | Add `{MetricName}JudgeOutput` schema |
| `src/pygaussia/llm/prompts.py` | Add `{metric_name}_reasoning_system_prompt` |
| `src/pygaussia/llm/__init__.py` | Export `{MetricName}JudgeOutput` |
| `tests/llm/test_schemas.py` | Add `Test{MetricName}JudgeOutput` tests |

## Architecture Pattern

All metrics follow this pattern:

```
Gaussia (base class)
    └── YourMetric
            ├── __init__(): Initialize with retriever and config
            ├── batch(): Process each conversation batch
            └── (optional) _process(): Override for custom aggregation
```

### Data Flow

```
Retriever.load_dataset() -> list[Dataset]
    ↓
Gaussia._process() iterates datasets
    ↓
YourMetric.batch() processes each conversation
    ↓
Results appended to self.metrics
```

## Step-by-Step Workflow

### 1. Create the Schema

Create `src/pygaussia/schemas/{metric_name}.py` using the `schema.py.template`.

### 2. Create the Metric Implementation

Create `src/pygaussia/metrics/{metric_name}.py` using either:
- `metric.py.template` for rule-based/data-based metrics
- `llm_metric.py.template` for LLM-judge metrics

### 3. Update Module Exports

Add to `src/pygaussia/metrics/__init__.py`:

```python
__all__ = [
    # ... existing metrics
    "{{MetricName}}",
]
```

### 3b. Update pyproject.toml

```toml
[project.optional-dependencies]
{{metric_name}} = []

metrics = [
    "gaussia[context,conversational,bestof,agentic,regulatory,{{metric_name}},humanity,toxicity,bias]",
]
```

### 4. Create Test Fixtures

Use `fixtures.py.template` to add fixtures to `tests/fixtures/mock_data.py` and `tests/fixtures/mock_retriever.py`.

### 5. Update conftest.py

Add fixtures to `tests/conftest.py`.

### 6. Create Tests

Create `tests/metrics/test_{metric_name}.py` using `test.py.template`.

## Metric Categories

### Simple Metrics (like Humanity)
- No external dependencies beyond base libraries
- Process each interaction independently
- Use lexicons or rule-based evaluation

### LLM-Judge Metrics (like Context, Conversational)
- Require a `BaseChatModel` parameter
- Use the `Judge` class from `pygaussia.llm`
- Need prompt templates in `pygaussia/llm/prompts.py`

### Guardian-Based Metrics (like Bias)
- Require a `Guardian` class for evaluation
- Use statistical confidence intervals

### Aggregation Metrics (like BestOf, Agentic)
- Override `_process()` instead of just `batch()`
- Compare multiple responses or assistants

## Common Patterns

### Using the Judge for LLM Evaluation

```python
from pygaussia.llm import Judge

judge = Judge(
    model=self.model,
    use_structured_output=self.use_structured_output,
    bos_json_clause=self.bos_json_clause,
    eos_json_clause=self.eos_json_clause,
)

reasoning, result = judge.check(
    system_prompt,
    user_query,
    data_dict,
    output_schema=YourOutputSchema,
)
```

### Statistical Analysis

```python
from pygaussia.statistical import FrequentistMode, BayesianMode

mode = FrequentistMode()
rate = mode.rate_estimation(successes=k, trials=n)

mode = BayesianMode(mc_samples=5000)
rate = mode.rate_estimation(successes=k, trials=n)
```

## Verification Checklist

- [ ] Schema inherits from `BaseMetric`
- [ ] Metric inherits from `Gaussia`
- [ ] `batch()` method signature matches base class
- [ ] Results appended to `self.metrics`
- [ ] Exports added to `src/pygaussia/metrics/__init__.py`
- [ ] pyproject.toml updated with optional dependency
- [ ] Test fixtures created in `tests/fixtures/`
- [ ] conftest.py updated with fixtures
- [ ] (LLM metrics) Judge output schema added
- [ ] (LLM metrics) Prompt added
- [ ] Tests pass: `uv run pytest tests/metrics/test_{{metric_name}}.py`
- [ ] Linting passes: `uv run ruff check src/pygaussia/metrics/{{metric_name}}.py`

## Template Files

See `templates/` directory for ready-to-use boilerplate:
- `metric.py.template` - Basic metric implementation
- `llm_metric.py.template` - LLM-judge metric implementation
- `schema.py.template` - Schema definition
- `test.py.template` - Test file structure
- `fixtures.py.template` - Test fixtures
