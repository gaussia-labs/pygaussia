# Contributing to pygaussia

## How Features Are Built

Gaussia follows a **paper-driven development** model. New metrics and significant features originate from research papers, then get translated into SDK implementations through a structured specification workflow.

```
Paper (LaTeX)  →  Accepted  →  Spec  →  Plan  →  Tasks  →  Code
```

### Step 1: Paper

A LaTeX paper is proposed, reviewed, and merged in [gaussia-labs/papers](https://github.com/gaussia-labs/papers). The paper defines the methodology formally: definitions, algorithms, evaluation criteria, and implementation considerations.

If you want to propose a new metric, start there.

### Step 2: Spec

Once the paper is accepted, an implementation issue is created in this repo. A contributor picks it up and creates a **feature specification** that translates the paper into implementable requirements.

The spec lives in `specs/[###-feature-name]/spec.md` and answers:

- What does this feature do? (derived from the paper's methodology)
- What are the user stories and acceptance criteria?
- How does it fit into the Gaussia pipeline? (which base class, which patterns)
- What is unclear? (marked with `[NEEDS CLARIFICATION]`)

The spec focuses on **what** and **why**, never on **how**.

### Step 3: Plan

A technical implementation plan maps the paper's algorithms to Gaussia's architecture:

- Which files to create and where
- Which design patterns to use
- Constitution gates (SOLID, patterns, simplicity, pipeline fit)
- Data model and contracts

The plan lives in `specs/[###-feature-name]/plan.md`.

### Step 4: Tasks

An executable task list is generated from the plan, organized by user story and following TDD:

1. Define schemas and contracts
2. Write tests (Red phase — tests must fail)
3. Implement (Green phase — make tests pass)
4. Polish (ruff, mypy, full test suite)

The task list lives in `specs/[###-feature-name]/tasks.md`.

### Step 5: Implement

Tasks are executed in order. Tests are written before implementation. Each user story is independently testable and deliverable.

### Work Without a Paper

Infrastructure, tooling, and refactoring work that doesn't originate from a paper starts directly at Step 2 (Spec). The spec must still define what and why before any code is written.

## Development Workflow

### Using Claude Code

If you use [Claude Code](https://claude.ai/code), the spec workflow is automated via slash commands:

```
/speckit/specify   Create a spec from an accepted paper (or feature description)
/speckit/clarify   Resolve ambiguities between paper and SDK implementation
/speckit/plan      Create the technical implementation plan
/speckit/tasks     Generate the TDD task list
/speckit/implement Execute tasks
```

### Manual Workflow

If you prefer to work without Claude Code:

1. Copy templates from `.specify/templates/` into `specs/[###-feature-name]/`
2. Fill in `spec.md` referencing the paper
3. Fill in `plan.md` with architecture decisions
4. Fill in `tasks.md` with concrete tasks
5. Implement following TDD

## Project Structure

```
pygaussia/
├── .specify/
│   ├── memory/
│   │   └── constitution.md       # Project principles (SOLID, patterns, TDD, architecture)
│   └── templates/
│       ├── spec-template.md      # Feature specification template
│       ├── plan-template.md      # Implementation plan template
│       └── tasks-template.md     # Task list template
├── specs/
│   └── [###-feature-name]/       # One folder per feature
│       ├── spec.md               # What and why (from paper)
│       ├── plan.md               # How (architecture decisions)
│       ├── data-model.md         # Pydantic schemas (if needed)
│       ├── research.md           # Technical research (if needed)
│       └── tasks.md              # Executable task list
├── src/pygaussia/                # Source code
└── tests/                        # Tests (mirrors src/ structure)
```

## Constitution

The project constitution (`.specify/memory/constitution.md`) defines non-negotiable principles:

| Article | Principle |
|---------|-----------|
| I | SOLID principles |
| II | Design patterns (no string branching, no if/elif chains, composition over inheritance) |
| III | Test-first development (Red-Green-Refactor) |
| IV | Code purity (zero tolerance for code smells) |
| V | Architecture integrity (pipeline, module boundaries) |
| VI | Simplicity (YAGNI, no premature abstractions) |
| VII | Paper-driven development lifecycle |

## Traceability

Every implementation traces back to its origin:

```
Code  →  tasks.md  →  plan.md  →  spec.md  →  Paper (main.tex)
```

This chain ensures that every line of code can be justified by a requirement, which traces to a design decision, which traces to a specification, which traces to peer-reviewed research.

## Development Setup

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/pygaussia
```

## Code Standards

- **Python**: `snake_case` for functions/variables, `PascalCase` for classes
- **Type hints**: Required on all function signatures
- **Docstrings**: Google-style, only when the signature doesn't convey intent
- **Comments**: Only explain *why*, never *what*
- **Tests**: pytest, in `tests/` mirroring `src/` structure
- **Imports**: Organized with isort (stdlib, third-party, local)

## Pull Requests

- Reference the spec (`specs/[###-feature-name]/`) in the PR description
- Reference the paper if applicable
- All tests must pass (`uv run pytest`)
- All linting must pass (`uv run ruff check .`)
- Type checking must pass (`uv run mypy src/pygaussia`)
