# Implementation Plan: Roast Me — Profile-then-Exploit Adversarial Evaluation

**Branch**: `004-roastme-metric` | **Date**: 2026-08-04 | **Spec**: `specs/004-roastme-metric/spec.md`

## Summary

Translate the Roast Me methodology from `papers/2026-06-roastme/` into the SDK as a specification the
user implements against. Seven abstract interfaces land in `core/`; the Pydantic shapes land in
`schemas/roastme.py`; `RoastMeProfiler(Gaussia)` is the metric; the Probe Library and the Exploiter are
generators whose product is the Roast Dataset, which the existing pipeline then consumes. Base
implementations ship for grading and for probe generation, each declared as a convenience rather than
as the definition of its component (spec D1).

The arithmetic gaussia owns — the violation score, the weakness map, `S(c)`, the `κ` gate, the `δ`
budget, refinement — is deliberately kept in modules with no I/O so it is verifiable against
hand-computed fixtures with no network and no GPU.

## [NEEDS DECISION — ALEX] Which category searches ship

**Not in question** (spec D7 / FR-033, already approved): the `CategorySearch` interface exists and
every search lives behind it. What needs a decision is which implementations land now.

| | A — attribute iteration only | B — both | C — both, training behind its own extra |
|---|---|---|---|
| `gaussia[roastme]` pulls | embedder, graph lib | + `peft`, `accelerate`, `trl` | embedder, graph lib |
| `gaussia[roastme-rl]` pulls | — | — | `peft`, `accelerate`, `trl` |
| Runs with no GPU | everything | the iteration only | everything in the base extra |
| Covered by CI | all of level 3 | the training loop is not: no GPU on the runner | same as A, plus the RL path uncovered |
| What the library ships as default | a search **never exercised against a live assistant** | the search the paper validated | the search the paper validated, opt-in |
| Cost to someone who only profiles | none | downloads the training stack | none |

The case for A is verifiability and install weight. The case against A, and the reason this is not
mine to close: **the only search the paper validated is the policy-gradient one.** Shipping without it
means the library's single available path is one nobody has ever run against a real assistant, and
that someone wanting to reproduce the paper cannot do it with the library.

**C is the recommendation.** Both implementations exist from day one, whoever only profiles pays
nothing, and the interface requirement is satisfied either way. Its cost is that it contradicts spec
D5, which fixes a single extra — so choosing C means amending an already-approved spec decision, in
its own commit, explicitly.

Everything below is written for C and degrades cleanly: under A, drop the `policy_gradient.py` row and
the second extra; under B, merge the two extras into one.

## Technical Context

**Language/Version**: Python 3.11+, matching the existing stack.

**Primary Dependencies**: no new mandatory runtime deps. One new extra,
`[project.optional-dependencies].roastme`, for what the shipped base implementations need — an
embedder for the retrieval engine and the realism estimator, a graph library for the graph engine.
Under decision C a second extra, `roastme-rl`, carries the training stack. Both stay **out** of the
`metrics` aggregate, following the `privacy-*` precedent, so `pip install gaussia[metrics]` does not
grow. Per FR-037 the interfaces in `core/` import with no extra installed.

**Testing**: `uv run pytest`. Correctness rests on hand-computed fixtures with deterministic stubs for
every interface (grader, target, engine, verifier, estimator). No network, no GPU, no external
artifacts in the default suite (SC-010).

**Project Type**: library extension, additive to `src/gaussia/`.

## Constitution Check

### SOLID Gate

- [x] **Single Responsibility**: `RoastMeProfiler` grades and aggregates. `ProbeLibrary` composes
  engines and merges duplicates. `Exploiter` composes a search and an estimator and emits a dataset.
  Each engine translates one retrieval mechanism. `scoring.py` holds only the arithmetic. No class
  straddles two jobs.
- [x] **Open/Closed**: adding an engine, a grader, a realism estimator or a search is a new file
  implementing the corresponding `core/` interface. `RoastMeProfiler`, `ProbeLibrary`, `Exploiter` and
  every schema stay untouched. SC-011 exercises this by importing the interfaces alone.
- [x] **Liskov**: `ProbeBatch` is a strict subclass of `Batch`; every inherited field keeps its type,
  and the two additions carry safe defaults. Any code accepting `list[Batch]` accepts
  `list[ProbeBatch]`. Concrete implementations honour their interface without narrowing it.
- [x] **Interface Segregation**: seven small interfaces instead of one god-object. A user writing a
  probe engine does not inherit grading, and a user writing a grader does not inherit knowledge-base
  access — which is also what keeps invariant 2 structurally true rather than merely documented.
- [x] **Dependency Inversion**: the metric and the generators depend on the `core/` abstractions and
  receive concrete implementations by construction. Nothing imports `graders.logprob` or a concrete
  engine from `metrics/` or from the generators.

### Pattern Gate

- [x] **No string-driven branching** — this is the one place it took work, so it is spelled out. The
  catalogue's `transform` field is a string the user writes, and it cannot be anything else:
  configuration is data. What the gate forbids is *branching* on it. Resolution: a `Transform`
  interface with four implementations and a registry that resolves the string **once**, during
  catalogue validation (FR-025). After that point behaviour is polymorphic and no `if transform ==`
  exists anywhere.
- [x] **No conditional chains for behaviour selection**: engine selection is composition, not a
  cascade — every applicable engine contributes and their outputs merge (FR-022). Grader selection is
  class identity at construction. Search selection is an injected strategy.
- [x] **Appropriate pattern identified**: Template Method for `RoastMeProfiler(Gaussia)`; Strategy for
  grader, engine, verifier, estimator, search and transform; Adapter for the target assistant;
  Composite for `ProbeLibrary` over its engines; Registry only for resolving the four transforms.
- [x] **Composition over inheritance**: engines are composed into the library, searches and estimators
  into the exploiter, graders into the contract. The only inheritance is from the abstractions and
  from `Batch`.

### Simplicity Gate

- [x] **No speculative features**: the plan implements FR-001…FR-038 and nothing else. No plug-in
  discovery, no caching, no async fan-out, no retry policy beyond the sampling fallback FR-008
  requires.
- [x] **No premature abstractions**: every interface has at least one real consumer in scope. The two
  that ship no implementation — `HookVerifier` and `TargetAssistant` — exist because the spec assigns
  them obligations that something has to name (FR-017, FR-018, and the `verified` field), and because
  SC-004 and SC-008 need to substitute them in tests.
- [x] **Gaussia does not own a catalogue file format.** The spec asks it to *validate* a catalogue,
  not to read YAML. The user constructs the specs, or hands over data Pydantic validates. Inventing a
  loader would be surface nobody asked for.
- [x] **No over-engineered error handling**: a catalogue fails validation up front (FR-025), a
  principle with no grader fails at construction (FR-003), a failed exchange is recorded ungraded
  (FR-016). No retries, no backoff, no fallback chains beyond FR-008.

### Pipeline Gate

- [x] **Respects the data flow**: `Retriever.load_dataset() → list[Dataset] → Gaussia._process() →
  batch() → self.metrics`. `_process_dataset` (`core/base.py:108-119`) hands `batch()` the whole
  `conversation` list, which is what lets the weakness map aggregate over the full probe set in one
  call without inventing a second pass.
- [x] **New schemas follow conventions**: all Pydantic. The metric result inherits `BaseMetric`
  (`schemas/metrics.py`), matching every other metric. `ProbeBatch` subclasses `Batch` the way
  `PrivacyBatch` does.
- [x] **Module boundaries respected**: `core/` gets abstractions only; `graders/` and
  `generators/probes|exploiter/` get concrete implementations; `schemas/` gets models with no business
  logic; `metrics/` gets the one `Gaussia` subclass. Dependencies flow downward only.
- [x] **No article bent by the Exploiter.** Per spec D8 it is a generator, not a metric: it does not
  subclass `Gaussia` and does not emit through `self.metrics`. It produces the Roast Dataset and the
  pipeline consumes that with existing metrics (FR-034).

## Architecture Decisions

### Why all seven interfaces live in `core/`

FR-037 requires the interfaces to be importable with the extra uninstalled, so a user can implement
against them without pulling an embedder or a graph library. `core/` carries no heavy imports, so
placing them there satisfies FR-037 structurally instead of by convention. It also matches how the
thirteen existing abstractions are organised — one per file, named after the abstraction, re-exported
from `core/__init__.py`.

### Why the weakness map's rate goes through the injected `StatisticalMode`

`StatisticalMode.rate_estimation(successes, trials)` (`statistical/base.py:40`) is already the exact
shape of `ω(π_j, z)`: how many probes of descriptor `z` made the assistant violate `π_j`, out of how
many. `FrequentistMode` returns the point estimate (`frequentist.py:27`); `BayesianMode` returns a
Beta-Binomial posterior summary (`bayesian.py:82-91`).

That is not only reuse. FR-012 requires each weakness entry to carry its sample size and its
dispersion precisely because a descriptor resting on a handful of probes cannot distinguish "never
failed" from "undersampled" — a limitation the paper states about its own results. With `BayesianMode`
injected, such a descriptor returns a credible interval instead of a `0.0` that reads as settled.
Hand-rolling a binomial standard error would have re-implemented, worse, something eight metrics
already depend on. Injection follows `role_adherence.py:191,206`: a `StatisticalMode | None`
parameter defaulting to `FrequentistMode()`.

### Why `ProbeBatch` overrides `ground_truth_assistant`

`Batch.ground_truth_assistant` is a required `str` (`schemas/common.py:33`), which assumes every turn
has an expected answer. A probe is a trap: there is no reference response, and forcing the user to
supply a meaningless one would be friction with no purpose. `ProbeBatch` overrides it with a default
of `""`. The field keeps its type, so LSP holds and any consumer of `Batch` is unaffected.

Note also that `Batch.weight` (`common.py:41`) already exists and is a *per-turn* weight consumed by
`Gaussia._resolve_weights`. It is unrelated to the contract's principle weights `w_j` and must not be
conflated with them; the plan keeps principle weights on the contract, where FR-001 validates them.

### Why a new `Document` schema rather than reusing `RegulatoryDocument`

`connectors/base.py::RegulatoryDocument` is a dataclass with two fields, `text` and `source`. It has
no identifier to quote in a hook, and no `structured` flag — which is the field that decides whether
an engine can establish absence over a document at all, and therefore the field the whole
absence/breadth trade-off of invariant 7 turns on. Extending a two-field dataclass from a different
module and a different modelling style to carry that would couple the Probe Library to the regulatory
corpus connector for no gain. A separate Pydantic `Document` in `schemas/roastme.py` is a distinct
concept, not duplicated code.

### Why the Profiler never receives a `TargetAssistant`

FR-014 asks the Profiler to grade a frozen set of responses without contacting the assistant. The plan
satisfies it structurally rather than with a flag: the Profiler takes **no** target at all. Responses
arrive in `Batch.assistant`, put there by whoever collected them — the user's `Retriever`. So
"grading without contacting the target" is not a mode the Profiler can be in or out of; it is the only
thing the Profiler can do, and no target credentials can be required because there is nowhere to put
them.

The target is used by the Exploiter, which has to send queries that do not exist yet. That is the
honest division: profiling is evaluation over data already collected, exploitation is generation. It
also explains why `TargetAssistant` ships no implementation and the `failed` flag nonetheless lives on
`ProbeBatch` — whoever collected the responses records that an exchange failed, and the Profiler only
has to honour it (FR-016).

### Why a new `graders/` module

The repo segregates concrete strategy implementations by kind: `guardians/`, `rerankers/`,
`embedders/`, `detectors/`. `graders/` mirrors that. Putting the logprob grader in `metrics/roastme.py`
would conflate what is computed with how a verdict is obtained — the same Feature Envy argument that
put the privacy adapters in `detectors/`.

### Why the exploiter's arithmetic is a separate module

`scoring.py` holds `S(c)`, the `κ` gate, the `δ` budget and refinement as functions over values, with
no I/O and no model. SC-007 then verifies the criterion that decides whether a category passes without
constructing a search, a target or an estimator. This is also what limits the blast radius of the open
decision above: whichever search ships, the part that decides is covered.

### Verification strategy

Deterministic stubs for all seven interfaces, and probe fixtures whose expected `v`, `ω` and
dispersion are computed by hand from the paper's equations (SC-001, SC-002). No fixture depends on a
provider, a network, a GPU or an external repository, so the default suite is hermetic (SC-010). Two
behaviours get dedicated doubles because they are easy to get silently wrong: a target that reports a
failed exchange (SC-008) and a grader whose provider exposes no usable logprobs (SC-009).

## New Files

| File | Purpose | Pattern |
|---|---|---|
| `src/gaussia/core/grader.py` | `Grader`: one principle's violation for a query and response, plus its evidence (FR-005) | Strategy |
| `src/gaussia/core/probe_engine.py` | `ProbeEngine`: can-handle plus documents+catalogue → tagged probes (FR-019, FR-021) | Strategy |
| `src/gaussia/core/hook_verifier.py` | `HookVerifier`: confirms a hook's `doc` label against the corpus | Strategy |
| `src/gaussia/core/target_assistant.py` | `TargetAssistant`: query in, response or failed exchange out (FR-017, FR-018) | Adapter |
| `src/gaussia/core/realism_estimator.py` | `RealismEstimator`: distance from the natural-query prior (FR-031) | Strategy |
| `src/gaussia/core/category_search.py` | `CategorySearch`: profile and contract → scored categories (FR-033) | Strategy |
| `src/gaussia/core/transform.py` | `Transform`: how a real entity becomes a probe's premise | Strategy |
| `src/gaussia/schemas/roastme.py` | Every model in `data-model.md`, including `ProbeBatch(Batch)` and the metric result over `BaseMetric` | Models |
| `src/gaussia/graders/__init__.py` | Re-exports the shipped grader | Module facade |
| `src/gaussia/graders/logprob.py` | `LogprobGrader`: last verdict-shaped token in the sequence, discarded when the model's own final answer does not parse to one, sampling fallback marked as such (FR-007, FR-008) | Adapter |
| `src/gaussia/generators/probes/__init__.py` | Module init | Module facade |
| `src/gaussia/generators/probes/library.py` | `ProbeLibrary`: composes engines, merges duplicates, records the originating engine (FR-020, FR-022, FR-024) | Composite |
| `src/gaussia/generators/probes/retrieval.py` | Retrieval engine; records that its absence labels are unreliable (FR-023) | Strategy |
| `src/gaussia/generators/probes/graph.py` | Graph engine; confirms absence from the complete graph | Strategy |
| `src/gaussia/generators/probes/transforms.py` | The four `Transform` implementations and the registry that resolves the catalogue string once | Strategy + Registry |
| `src/gaussia/generators/probes/catalogue.py` | Catalogue validation: dangling principle, dangling plugin, unknown transform, out-of-range `doc`, duplicate identifier (FR-025, FR-026, SC-005) | Validator |
| `src/gaussia/generators/exploiter/__init__.py` | Module init | Module facade |
| `src/gaussia/generators/exploiter/scoring.py` | `S(c)`, the `κ` gate, the `δ` budget, refinement (FR-029…FR-032) | Pure functions |
| `src/gaussia/generators/exploiter/attribute_iteration.py` | `AttributeIterationSearch`: common attributes of the highest-scoring pool, then attribute subsets | Strategy |
| `src/gaussia/generators/exploiter/policy_gradient.py` | `PolicyGradientSearch`. **Pending the open decision** — under A this row and the `roastme-rl` extra drop | Strategy |
| `src/gaussia/generators/exploiter/exploiter.py` | `Exploiter`: composes search and estimator, emits the Roast Dataset and the failure report (FR-034…FR-036) | Composition |
| `src/gaussia/metrics/roastme.py` | `RoastMeProfiler(Gaussia)` (FR-010…FR-016) | Template Method |
| `tests/metrics/test_roastme_profiler.py` | Grading, the weakness map, ungraded outcomes, and control exclusion driven purely by a strategy with no plugin — including the pair of probes over the same documented entity that land on opposite sides of that line (SC-003) | Pytest |
| `tests/generators/exploiter/test_roast_dataset.py` | An emitted Roast Dataset loads through the standard dataset contract and is consumed end-to-end by an existing metric with no change to that metric (SC-006) | Pytest |
| `tests/generators/probes/test_catalogue.py` | Every rejection path of SC-005 | Pytest |
| `tests/generators/probes/test_library.py` | Composition, duplicate merging, absence-label reliability (SC-004) | Pytest |
| `tests/generators/exploiter/test_scoring.py` | `S(c)`, `κ`, `δ`, refinement against hand computation (SC-007) | Pytest |
| `tests/graders/test_logprob.py` | Last-token location, discard on unparsed final answer, fallback marking (SC-009) | Pytest |
| `tests/fixtures/roastme/` | Deterministic doubles for all seven interfaces, plus hand-computed probe fixtures | Test doubles |
| `docs/metrics/roastme.mdx` | The metric page. States that no grader has been calibrated against human labels and that the figures are a judge-only measurement (FR-038), and that the shipped search is not the one the paper validated | Docs |
| `examples/roastme/catalogue/` | Schema examples for `PluginSpec` and `StrategySpec` — the shape, with domain-neutral prose. **Not a domain catalogue** (FR-027) | Example |
| `examples/roastme/jupyter/` | Runnable notebook: build a contract, validate a catalogue, profile a frozen response set. Mirrors `examples/privacy/jupyter/` | Example |

## Modified Files

| File | Change | Justification |
|---|---|---|
| `src/gaussia/core/__init__.py` | Import the seven abstractions and extend `__all__` | Convention: every abstraction in `core/` is re-exported there |
| `src/gaussia/metrics/__init__.py` | Add `RoastMeProfiler` | Matches `Privacy`, `Toxicity`, `RoleAdherence` |
| `src/gaussia/schemas/__init__.py` | Docstring pointer only, no eager import | The file's own docstring establishes that metric-specific schemas are imported from their module so unnecessary dependencies do not load |
| `pyproject.toml` | Add `roastme`, and `roastme-rl` under decision C | FR-037. Kept out of the `metrics` aggregate, as `privacy-*` is |
| `docs/docs.json` | Register `metrics/roastme` in the metrics nav group | The docs nav is a registry; a page that is not listed is not published |

Three files deliberately **not** modified, which is worth stating because two of them are where this
feature would most plausibly have leaked:

- `src/gaussia/llm/judge.py` — FR-009 and spec D4. Roast Me grades behind its own interface. The
  shared judge reads the *first* generated token, which is wrong for a reasoning model, but fixing it
  there would change behaviour for `role_adherence`, which depends on it. Extending shared code that
  `003-judge-logprobs-refactor` recently reworked is a separate decision with its own blast radius,
  and it is not this feature's to take.

- `src/gaussia/generators/__init__.py` eagerly imports concrete classes. Registering the probe engines
  there would make `import gaussia.generators` require the embedder, breaking FR-037. The subpackages
  are imported directly, which is the precedent `core/__init__.py` documents for
  `document_retriever` and `contradiction_checker`.
- `CHANGELOG.md` is generated by python-semantic-release
  (`[tool.semantic_release.changelog]`, `pyproject.toml:392`). It must not be hand-edited.

## Complexity Tracking

Under decision A the Constitution Check passes with no violations.

Under B or C one entry applies:

| Violation | Why needed | Simpler alternative rejected because |
|---|---|---|
| `PolicyGradientSearch` has no automated coverage: it trains a LoRA policy and the CI runner has no GPU | It is the only search the paper validated. Omitting it means the library cannot reproduce the published methodology | Testing the loop on CPU would assert that a training step runs, not that it searches, which is a test that passes while telling nobody anything. Mitigation instead: the criterion that decides whether a category passes lives in `scoring.py` with full hand-computed coverage, so what is uncovered is the optimisation loop, not the decision rule |
