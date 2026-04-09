# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gaussia (pygaussia) is the Python implementation of the Gaussia AI evaluation framework. It provides metrics for measuring fairness, toxicity, bias, conversational quality, and more for AI models and assistants.

- **PyPI package name**: `gaussia`
- **Import name**: `pygaussia`
- **Base class**: `Gaussia` (in `pygaussia.core.base`)

## Development Commands

```bash
# Install dependencies
uv sync

# Run scripts in development
uv run python your_script.py

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/metrics/test_toxicity.py

# Run a specific test
uv run pytest tests/metrics/test_toxicity.py::test_function_name

# Run tests in parallel
uv run pytest -n auto

# Skip slow tests
uv run pytest -m "not slow"

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/pygaussia

# Build package
uv build
```

## Design Principles

This project demands clean, well-architected object-oriented code. Every piece of code must reflect intentional design decisions. Think in terms of **SOLID principles** and **design patterns** before writing a single line.

### SOLID Principles — Non-Negotiable

- **S — Single Responsibility Principle**: Each class has exactly one reason to change.
- **O — Open/Closed Principle**: Adding a new metric, guardian, or strategy must NEVER require changing existing classes.
- **L — Liskov Substitution Principle**: Any subclass must be usable wherever its parent class is expected.
- **I — Interface Segregation Principle**: Prefer small, focused interfaces over large monolithic ones.
- **D — Dependency Inversion Principle**: High-level modules depend on abstractions. Concrete classes are injected.

### Design Patterns — Mandatory Approach

- **No string-driven branching**: NEVER pass strings as parameters to select behavior. Use Strategy, Command, or Template Method instead.
- **No long if/elif/else chains**: Replace with polymorphism, a registry of classes, or the Strategy pattern.
- **Favor composition over inheritance**: Use Decorator, Adapter, and Strategy to compose behavior.
- **Program to interfaces, not implementations**: Depend on abstract base classes or Protocols.

### Patterns Already Used

- **Template Method**: `Gaussia` base class defines the `_process()` flow, subclasses implement `batch()`.
- **Strategy**: `FrequentistMode` / `BayesianMode` for statistical analysis, `SelectionStrategy` for generators.
- **Adapter**: Guardian implementations adapt different models behind a common `Guardian` interface.
- **Factory Method**: `MyMetric.run(RetrieverClass)` instantiates and orchestrates the pipeline.

## Code Style

- **Minimal comments**: Only comment *why* something is done when the reasoning is non-obvious.
- **No redundant docstrings**: Do not add docstrings to private methods or simple functions where the signature conveys intent.
- **No commented-out code**: Never leave commented-out code blocks.
- **No section dividers**: Do not add decorative comment blocks.

## Code Smells — Zero Tolerance

Reference: https://refactoring.guru/refactoring/smells

Every code smell is a defect. Do not introduce any. If you encounter one while working on nearby code, refactor it. This includes: Long Methods, Large Classes, Primitive Obsession, Switch Statements, Duplicate Code, Dead Code, Speculative Generality, Feature Envy, and Inappropriate Intimacy.

## Architecture

### Core Pattern: Gaussia Base Class

All metrics inherit from `Gaussia` (in `src/pygaussia/core/base.py`):
1. Subclass `Gaussia`
2. Implement `batch()` method to process conversation batches
3. Append results to `self.metrics`
4. Use via `MyMetric.run(RetrieverClass, **kwargs)`

### Data Flow

```
Retriever.load_dataset() -> list[Dataset]
    ↓
Gaussia._process() iterates datasets
    ↓
Metric.batch() processes each conversation
    ↓
Results in self.metrics
```

### Key Modules

- **`src/pygaussia/metrics/`**: Metric implementations (Toxicity, Bias, Context, Conversational, Humanity, BestOf, Agentic, Vision, Regulatory)
- **`src/pygaussia/core/`**: Base classes — `Gaussia`, `Retriever`, `Guardian`, `ToxicityLoader`, `SentimentAnalyzer`
- **`src/pygaussia/schemas/`**: Pydantic models for data validation (`Dataset`, `Batch`, metric-specific schemas)
- **`src/pygaussia/statistical/`**: Statistical modes (`FrequentistMode`, `BayesianMode`)
- **`src/pygaussia/guardians/`**: Bias detection implementations
- **`src/pygaussia/loaders/`**: Dataset loaders (e.g., `HurtlexLoader` for toxicity lexicons)
- **`src/pygaussia/llm/`**: LLM integration (`Judge`, prompts, schemas for structured outputs)
- **`src/pygaussia/extractors/`**: Group extraction implementations (`EmbeddingGroupExtractor`)
- **`src/pygaussia/generators/`**: Synthetic dataset generation
- **`src/pygaussia/prompt_optimizer/`**: Prompt optimization algorithms (GEPA, MIPROv2)
- **`src/pygaussia/explainability/`**: Token attribution analysis
- **`src/pygaussia/utils/`**: Utilities (logging configuration)

### Custom Retriever Pattern

Users implement a `Retriever` subclass to load their data:

```python
from pygaussia.core.retriever import Retriever
from pygaussia.schemas.common import Dataset

class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Load and return datasets
        pass
```

### Test Fixtures

Shared fixtures in `tests/conftest.py` provide mock retrievers and datasets for each metric type.

## Key Data Structures

- **`Dataset`**: A conversation session with `session_id`, `assistant_id`, `language`, `context`, and `conversation` (list of Batch)
- **`Batch`**: Single Q&A interaction with `query`, `assistant`, `ground_truth_assistant`, `qa_id`

## Paper-Driven Development

This project follows a paper-first workflow. Metrics and significant features originate from accepted papers in [gaussia-labs/papers](https://github.com/gaussia-labs/papers), then get translated into SDK implementations via Specification-Driven Development.

### Full Lifecycle

```
Paper (LaTeX)  →  Accepted  →  Implementation Issue  →  SDK Spec  →  Plan  →  Tasks  →  Code
```

### SDD Workflow

The `speckit` skill (installed via `npx skills add @gaussia-labs/skills`) guides the full workflow. Tell Claude to:

- **Specify**: Create a spec from a paper or feature description
- **Clarify**: Resolve gaps between paper methodology and SDK implementation
- **Plan**: Map paper algorithms to Gaussia architecture
- **Tasks**: Generate executable TDD task list
- **Implement**: Execute tasks following test-first

### Key Files

- **`.specify/memory/constitution.md`**: SDK-specific principles (pipeline, module boundaries, Python tools)
- **`specs/[###-feature-name]/`**: Feature specifications, plans, and task lists

### Rules

- Papers are the upstream source of truth for methodology. Specs translate papers into implementable requirements.
- Every implementation traces back: code → plan → spec → paper.
- `[NEEDS CLARIFICATION]` markers must be resolved before planning.
- Tests are written and verified to FAIL before implementation (TDD).
- Constitution gates (SOLID, Pattern, Simplicity, Pipeline) must pass before implementation.
- For infrastructure/tooling work without a paper, the workflow starts at Spec directly.
