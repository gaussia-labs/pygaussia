# Implementation Plan: Roast Me — Profile-then-Exploit Adversarial Evaluation

**Branch**: `004-roastme-metric` | **Date**: 2026-08-06 | **Spec**: `specs/004-roastme-metric/spec.md`

## What changed since the version you reviewed

This file was substantially rewritten, so the diff is large. The changes, in one place:

- **Reversed**: an earlier draft argued for reusing the framework's statistical modes for the weakness
  map. They are no longer used at all — see "Why the framework's statistical modes are not used".
- **Reversed**: the Profiler had no target assistant and graded pre-collected responses. Both components
  now receive the target interface.
- **Resolved**: the open decision on which category searches ship. Both, with training behind its own
  extra.
- **Removed**: "Why the Profiler never receives a `TargetAssistant`", and the section re-litigating the
  extras decision.
- **Added**: the Testing Gate, which carries the one unchecked box in this document; "Why the query
  generator and the on-profile filter are separate interfaces"; "Why the enumeration engine ships but
  does not run by default"; "How optional dependencies fail"; the open decision below.
- **Grown**: interfaces from seven to ten, probe engines from two to four.

## Summary

Translate the Roast Me methodology from `papers/2026-06-roastme/` into the SDK as a specification the
user implements against. Ten abstract interfaces land in `core/`; the Pydantic shapes land in
`schemas/roastme.py`; the Probe Library, the Profiler and the Exploiter land in `generators/roastme/` as
a generator subsystem that drives an injected target assistant and produces the Roast Dataset, which the
framework's existing pipeline then consumes (spec D15).

Base implementations ship for grading and for probe generation, each declared as a convenience rather
than as the definition of its component (spec D1). The arithmetic gaussia owns — the violation score,
the weakness map with its standard errors, `S(c)`, the `κ` gate, the `δ` budget, refinement — is kept in
modules with no I/O so it is verifiable against hand-computed fixtures with no network and no GPU.

**One question for the reviewer**, and it is the only thing here that is not decided: the paper defines
`τ`, `λ`, `κ`, `δ` and `η` as parameters of the method and fixes no values. This plan makes them
**required** inputs with no defaults, on the grounds that a default would be gaussia deciding how hard a
category has to fail before it counts. If you would rather ship recommended values, say so and the
config models get defaults.

## Technical Context

**Language/Version**: Python 3.11+, matching the existing stack.

**Primary Dependencies**: no new mandatory runtime deps. Two new extras:

```toml
roastme    = ["sentence-transformers>=5.0.0", "torch>=2.0.0", "networkx>=3.0"]
roastme-rl = ["gaussia[roastme]", "peft>=0.10.0", "accelerate>=0.25.0", "trl>=0.8.0"]
```

`roastme` covers inference: an embedder for the retrieval engine and the realism estimator, a graph
library for the graph engine. `roastme-rl` adds the training stack for the policy-gradient search. Both
stay **out** of the `metrics` aggregate and out of `all`, following the `privacy-*` precedent, so no
existing install grows. Per FR-037 the interfaces in `core/` import with no extra installed.

**Testing**: `uv run pytest`. Correctness rests on hand-computed fixtures with deterministic stubs for
every interface. No network, no GPU, no external artifacts in the default suite (SC-010).

**Project Type**: library extension, additive to `src/gaussia/`.

## Constitution Check

### SOLID Gate

- [x] **Single Responsibility**: `Profiler` drives the target, grades and aggregates. `ProbeLibrary`
  composes engines and merges duplicates. `Exploiter` composes a search, a query generator, a filter and
  an estimator. Each engine translates one retrieval mechanism. `searches/scoring.py` holds only
  arithmetic. `dataset.py` only converts at the output boundary. No class straddles two jobs.
- [x] **Open/Closed**: adding an engine, a grader, an estimator, a search, a query generator, a filter or
  a target adapter is a new file implementing the corresponding `core/` interface, with nothing else
  touched. **One exception, stated rather than hidden**: the transform set is closed by FR-025 to four
  members, so a fifth transformation requires editing `transforms.py` *and* relaxing that requirement.
  It is closed because `transform` is the one field whose value changes what a probe *means* — an
  unrecognised transformation would silently produce probes whose `doc` label nobody can trust. If that
  trade is wrong, FR-025 is where to change it.
- [x] **Liskov**: nothing subclasses a framework class, so there is no substitution surface to break.
  A target adapter that replays recorded responses is substitutable for one that calls a live assistant,
  which is what lets FR-014's credential-free path use the same code as a live run.
- [x] **Interface Segregation**: ten small interfaces instead of one god-object. A user writing a probe
  engine does not inherit grading, and a grader does not inherit knowledge-base access — which is what
  keeps invariant 2 structurally true rather than merely documented.
- [x] **Dependency Inversion**: every component depends on the `core/` abstractions and receives concrete
  implementations by construction. Nothing imports `graders.logprob` or a concrete engine.

### Pattern Gate

- [x] **No string-driven branching** — the one place it took work. The catalogue's `transform` field is a
  string the user writes, and it cannot be anything else: configuration is data. What the gate forbids is
  *branching* on it. Resolution: a `Transform` interface with four implementations and a registry that
  resolves the string **once**, during catalogue validation (FR-025). After that point behaviour is
  polymorphic and no `if transform ==` exists anywhere. The string is still carried on the hook's `how`
  field for auditing, which is a record rather than a dispatch.
- [x] **No conditional chains for behaviour selection**: engine selection is composition, not a cascade —
  every applicable engine contributes and their outputs merge (FR-022). Grader, search, query generator,
  filter and estimator selection is class identity at construction.
- [x] **Appropriate pattern identified**: Strategy for grader, engine, enumerator, verifier, transform,
  query generator, filter, estimator and search; Adapter for the target assistant and for the
  output-boundary conversion; Composite for `ProbeLibrary` over its engines; Registry only for resolving
  the four transforms.
- [x] **Composition over inheritance**: engines compose into the library; search, generator, filter,
  estimator and target compose into the exploiter; graders compose into the contract. The only
  inheritance anywhere is from the abstractions themselves.

### Simplicity Gate

- [x] **No speculative features**: the plan implements FR-001…FR-038 and nothing else. No plug-in
  discovery, no caching, no async fan-out, no retry policy beyond the sampling fallback FR-008 requires.
- [x] **No premature abstractions**: every interface has a real consumer in scope. A *category generator*
  is deliberately **not** an interface: the training-free search has none — it intersects a pool — so
  requiring one would force a fake implementation.
- [x] **Gaussia does not own a catalogue file format.** The spec asks it to *validate* a catalogue, not to
  read YAML. The user constructs the specs, or hands over data Pydantic validates.
- [x] **No over-engineered error handling**: a catalogue fails validation up front (FR-025), a principle
  with no grader fails at construction (FR-003), a failed exchange is recorded ungraded (FR-016).

### Testing Gate

- [x] Tests precede implementation, and every success criterion has a task that asserts it.
- [ ] **`PolicyGradientSearch` ships with no automated coverage and needs your approval.** It trains a
  LoRA policy; the CI runner has no GPU, so nothing verifies it. Article III is non-negotiable, so this
  box is left unchecked deliberately rather than absorbed into a footnote. Testing a training step on CPU
  would assert that the step runs, not that the search searches — a test that passes while telling nobody
  anything. What bounds the risk: the criterion that decides whether a category passes lives in
  `searches/scoring.py` with full hand-computed coverage, so what is uncovered is the optimisation loop,
  not the decision rule. The class is marked `requires_gpu`, a marker the repo already defines.

### Pipeline Gate

- [x] **Roast Me sits upstream of the metric pipeline rather than inside it** (spec D15). It does not
  consume the dataset-loading contract, because there is no dataset to load. What enters the canonical
  flow is the output: the Roast Dataset, loaded the ordinary way for existing metrics to evaluate
  (FR-034). This is not a deviation — the framework's own generator base is a plain class rather than a
  metric, so producing datasets outside the metric pipeline is established.
- [x] **New schemas follow conventions**: all Pydantic, in `schemas/roastme.py`. The framework's `Batch`
  and `Dataset` appear only at the output boundary; `data-model.md` states what each of their required
  fields is filled with, so the conversion is buildable rather than assumed.
- [x] **Module boundaries respected**: `core/` gets abstractions only; `graders/` and
  `generators/roastme/` get concrete implementations; `schemas/` gets models with no business logic.
  Dependencies flow downward only, and `metrics/` is not touched.

## Architecture Decisions

### Why all ten interfaces live in `core/`

FR-037 requires them importable with the extra uninstalled, so a user can implement against them without
pulling an embedder or a graph library. `core/` carries no heavy imports, so placing them there satisfies
FR-037 structurally instead of by convention.

The existing organisation it follows, stated accurately: `core/` holds twelve abstractions today. Most are
one per file and re-exported from `core/__init__.py`, but two are not — `document_retriever.py` and
`contradiction_checker.py` hold several classes each and are **deliberately excluded** from the
re-export, because they require numpy. That file's own docstring says so. The rule is therefore
"abstractions with no heavy imports are re-exported; ones that would drag a dependency are imported
directly", and Roast Me's ten fall on the light side of it.

### [NEEDS DECISION — ALEX] Which pluggable pieces ship a working implementation

Not "what the run returns" — that is settled. This is about which of the ten interfaces come with an
implementation in the box, so a user can run without writing code.

Settled by the approved spec: the four probe engines (D14, FR-022) and the logprob grader (FR-007) ship.
The entity enumerator does not, because enumerating a domain's entities is domain knowledge rather than a
technique.

Open, and it is a product call rather than a technical one:

| Piece | Ships? | The case for | The case against |
|---|---|---|---|
| **Realism estimator** | probably yes | the paper gives the construction — expected cosine distance from a prior pool — and the framework already has the embedder it needs | none material |
| **Query generator** | unclear | the Exploiter cannot run at all without one | the paper names it as a role and gives no construction, so shipping one means inventing it |
| **On-profile filter** | unclear | same: without one, the `κ` gate has nothing to compare against | same: the paper deliberately leaves `f` abstract, so a shipped default is our invention that every user inherits |

The honest position: **the query generator and the filter are in the same situation**, and an earlier
draft of this plan treated them differently on the same premise, which was wrong. Either both ship as
declared inventions, or neither ships and the Exploiter refuses to run until both are supplied.

Whichever way it goes, spec D1 needs to agree: it currently says base implementations ship "only where
D14 and the requirements name them", and no requirement names any of these three. That is one line in
the spec once the decision is made.

The rest of this plan is written for **the estimator shipping and the other two not** — the only reading
the paper supports on its own. It degrades in one direction: if the query generator and the filter ship
too, two files and two tasks are added and D1 is widened.

### Why the query generator and the on-profile filter are separate interfaces

Both are things the paper names and an earlier draft folded into the search, which made two requirements
unimplementable.

Invariant 5 requires the query generator stay unmodified while the search is optimised. That is only
checkable if the two are distinct objects — if the search owns query generation internally, "it was not
modified" is an assertion about nothing.

The `κ` gate compares against a score of how on-profile and indirect a single query is. That is a
semantic judgement about one query, not arithmetic over values, so it cannot live in `scoring.py` with
the rest. Separating it lets `scoring.py` stay a pure-function module that takes the filter's output as a
number.

### Why the enumeration engine ships but does not run by default

It is the only engine that cannot work from a knowledge base alone: enumerating "every article that
exists" needs domain knowledge no general library has. So it takes an injected `EntityEnumerator`, which
gaussia specifies and ships none of.

Shipping the engine rather than omitting it keeps the strongest absence guarantee available to whoever
can afford to write that enumerator, while three engines still run out of the box. This is also what
gives `Document.structured` a consumer: it is the field the enumeration engine reads to know whether a
document's boundary can be enumerated at all.

### Why the framework's statistical modes are not used

`S(c) = Φ̂ₙ(c) − λ·seₙ(c)` needs a **standard error**, and the framework's dispersion utility returns a
mean absolute deviation. Those are different quantities, not differently-scaled ones.

Its rate estimator would have fit the weakness map, so the tempting design was to use the framework's
mode for `ω` and hand arithmetic for `S(c)`. That puts two statistical treatments inside one measurement,
where a reader cannot tell which number came from which. Roast Me computes its own rates and standard
errors, uniformly, in the module that also computes `S(c)`.

### Why there is no input-side `ProbeBatch`

With spec D15 there is no dataset on the way in, so there is nothing to subclass: the Profiler works on
`Probe` objects and the responses the target returns. A `Batch` subclass appears only on the **output**
side, where `data-model.md` specifies it and every field it fills.

One trap avoided: `Batch.weight` is the framework's own per-turn aggregation weight, unrelated to the
contract's principle weights `w_j`. Keeping principle weights on the contract, where FR-001 validates
them, keeps the two from ever being confused.

### Why the Profiler and the Exploiter both receive the target interface

The user implements one adapter for their transport and both components drive it. An assistant that only
exists as a page in a browser is still reachable: the user implements the interface, returns the response
as `TargetResponse`, and gaussia consumes nothing but that. Pushing the exchange onto the user instead
would mean every user writing their own probe-sending loop, which is the part gaussia is better placed to
own.

What it buys rather than costs: FR-014 asks that grading a recorded response set need no credentials.
That is now an *implementation* of the same interface — one returning recorded responses — rather than a
mode the Profiler can be in or out of. One code path serves both, and the credential-free case stops
being a special case that could rot.

The `failed` flag lives on the response model for the same reason: whoever writes the adapter is the only
party that can recognise a transport failure, and FR-016 obliges the Profiler to honour it.

### Why a new `Document` schema

The framework's corpus-connector document is a two-field dataclass with no identifier and no enumerability
flag, and lives in a different module with a different modelling style. `data-model.md` carries the
comparison.

### Why a new `graders/` module

The repo groups concrete implementations by the kind of thing they are, one directory per family:
`guardians/`, `rerankers/`, `embedders/`, `detectors/`. `graders/` follows that. To be explicit, since the
wording invited the opposite reading: **nothing selects a grader by a string.** `Grader` is an abstract
base class, users extend it freely, and the shipped logprob grader is one subclass among however many a
user writes.

### Why the exploiter's arithmetic is a separate module

`searches/scoring.py` holds `S(c)`, the `κ` comparison, the `δ` comparison, refinement and the weakness
map's rates and standard errors as functions over values, with no I/O and no model. SC-007 then verifies
the criterion that decides whether a category passes without constructing a search, a target or an
estimator. It is also what bounds the untested surface of the policy-gradient search.

### How optional dependencies fail

The constitution forbids `try/except ImportError` and dynamic imports, so a missing extra must surface as
a plain `ImportError` at the import site of the module that needs it. Concretely:

- The framework already defines `LogprobsNotSupportedError` and `LogprobsExtractionError` for a provider
  that returns no usable logprobs. The base grader reuses them rather than inventing its own, which is why
  `core/exceptions.py` is not in the Modified Files table.
- `core/` and `schemas/roastme.py` import nothing heavy, so the interfaces and the models always import.
- `generators/roastme/__init__.py` re-exports `ProbeLibrary`, `Profiler` and `Exploiter` — none of which
  imports an engine. `probes/__init__.py` **must not** re-export `retrieval.py`, `graph.py` or `grag.py`;
  those are imported directly by the user, the way the framework already treats its numpy-dependent
  abstractions. Without that rule, `from gaussia.generators.roastme import Profiler` would pull an
  embedder and break FR-037 one level below where the plan guards against it.
- SC-011's task asserts this for the subsystem, not only for `core/`.

### Verification strategy

Deterministic stubs for all ten interfaces, and probe fixtures whose expected `v`, rates and standard
errors are computed by hand from the paper's equations (SC-001, SC-002). No fixture depends on a provider,
a network, a GPU or an external repository (SC-010). Three behaviours get dedicated doubles because they
are easy to get silently wrong: a target that returns recorded responses, a target that reports a failed
exchange (SC-008), and a grader whose provider exposes no usable logprobs (SC-009).

## New Files

| File | Purpose | Pattern |
|---|---|---|
| `src/gaussia/core/grader.py` | `Grader` (FR-005) | Strategy |
| `src/gaussia/core/probe_engine.py` | `ProbeEngine`, including the entity kinds it declares it handles (FR-019, FR-021, FR-025) | Strategy |
| `src/gaussia/core/entity_enumerator.py` | `EntityEnumerator`: the entities of a kind that exist in the base. Ships no implementation | Strategy |
| `src/gaussia/core/hook_verifier.py` | `HookVerifier` | Strategy |
| `src/gaussia/core/transform.py` | `Transform` | Strategy |
| `src/gaussia/core/target_assistant.py` | `TargetAssistant` (FR-017, FR-018) | Adapter |
| `src/gaussia/core/query_generator.py` | `QueryGenerator`: category → concrete queries (FR-033) | Strategy |
| `src/gaussia/core/on_profile_filter.py` | `OnProfileFilter`: how on-profile one query is, for the `κ` gate (FR-030) | Strategy |
| `src/gaussia/core/realism_estimator.py` | `RealismEstimator` (FR-031) | Strategy |
| `src/gaussia/core/category_search.py` | `CategorySearch` (FR-033) | Strategy |
| `src/gaussia/schemas/roastme.py` | Every model in `data-model.md`, including `TargetResponse` and the three config models | Models |
| `src/gaussia/graders/__init__.py` | Re-exports the shipped grader | Module facade |
| `src/gaussia/graders/logprob.py` | `LogprobGrader` (FR-007, FR-008) | Adapter |
| `src/gaussia/generators/roastme/__init__.py` | Re-exports `ProbeLibrary`, `Profiler`, `Exploiter` — no engine imports | Module facade |
| `src/gaussia/generators/roastme/profiler.py` | Drives the target over a probe set, grades, aggregates (FR-010…FR-016) | Composition |
| `src/gaussia/generators/roastme/exploiter.py` | Composes search, query generator, filter, estimator and target; emits the dataset and the report (FR-034…FR-036) | Composition |
| `src/gaussia/generators/roastme/dataset.py` | The output-boundary conversion (FR-034) | Adapter |
| `src/gaussia/generators/roastme/probes/__init__.py` | Module init — **no engine re-exports** | Module facade |
| `src/gaussia/generators/roastme/probes/library.py` | `ProbeLibrary` (FR-020, FR-022, FR-024) | Composite |
| `src/gaussia/generators/roastme/probes/retrieval.py` | Retrieval engine; records that its absence labels are unreliable (FR-023) | Strategy |
| `src/gaussia/generators/roastme/probes/graph.py` | Graph engine; confirms absence from the complete graph | Strategy |
| `src/gaussia/generators/roastme/probes/grag.py` | Multi-hop engine; false premises spanning several entities | Strategy |
| `src/gaussia/generators/roastme/probes/enumeration.py` | Enumeration engine; opt-in, requires an injected `EntityEnumerator` (D14) | Strategy |
| `src/gaussia/generators/roastme/probes/transforms.py` | The four `Transform` implementations and the registry | Strategy + Registry |
| `src/gaussia/generators/roastme/probes/catalogue.py` | Catalogue validation, including `entity_kind` against the engines' declared kinds (FR-025, FR-026, SC-005) | Validator |
| `src/gaussia/generators/roastme/searches/__init__.py` | Module init | Module facade |
| `src/gaussia/generators/roastme/searches/scoring.py` | `S(c)`, the gates, refinement, rates and standard errors (FR-012, FR-029…FR-032) | Pure functions |
| `src/gaussia/generators/roastme/searches/attribute_iteration.py` | The training-free search; the default | Strategy |
| `src/gaussia/generators/roastme/searches/realism.py` | Base realism estimator: expected cosine distance from a prior pool, using an injected embedder — the instantiation the paper gives for `δ` | Adapter |
| `src/gaussia/generators/roastme/searches/policy_gradient.py` | The policy-gradient search, behind `roastme-rl`, marked `requires_gpu` | Strategy |
| `tests/generators/roastme/test_profiler.py` | Grade retention (FR-004), control exclusion by empty plugin including the pair over one documented entity (SC-003), ungraded outcomes (SC-008) | Pytest |
| `tests/generators/roastme/test_dataset.py` | The conversion and its field fills; an existing metric consumes the result unchanged (SC-006) | Pytest |
| `tests/generators/roastme/test_catalogue.py` | Every rejection path (SC-005) | Pytest |
| `tests/generators/roastme/test_library.py` | Composition, duplicate merging, absence reliability (SC-004), the enumeration engine refusing to run without an enumerator | Pytest |
| `tests/generators/roastme/test_scoring.py` | `v`, rates, standard errors, `S(c)`, the gates, refinement (SC-001, SC-002, SC-007) | Pytest |
| `tests/generators/roastme/test_exploiter.py` | The report ranked by `S(c)` with the queries at or above `τ` surfaced (FR-035), an auditable evaluation record (FR-036), and the refusal to run with no on-profile filter supplied | Pytest |
| `tests/graders/test_logprob.py` | Last-token location, discard on unparsed answer, fallback marking (SC-009) | Pytest |
| `tests/fixtures/roastme/` | Deterministic doubles for all ten interfaces plus hand-computed probe fixtures | Test doubles |
| `docs/advanced/roastme.mdx` | The subsystem page, alongside `generators` and `prompt-optimizer` (FR-038) | Docs |
| `examples/roastme/catalogue/` | Schema examples, **not** a domain catalogue (FR-027) | Example |
| `examples/roastme/jupyter/` | Runnable notebook: contract, catalogue, profile against a recorded target | Example |

Package `__init__.py` files for the new test directories are covered in `tasks.md`; every test directory
in this repo has one.

## Modified Files

| File | Change | Justification |
|---|---|---|
| `src/gaussia/core/__init__.py` | Import the ten abstractions and extend `__all__` | They carry no heavy imports, so they belong on the re-exported side of that file's own rule |
| `src/gaussia/schemas/__init__.py` | Docstring pointer only, no eager import | The file's docstring establishes that feature-specific schemas are imported from their module |
| `pyproject.toml` | Add the `roastme` and `roastme-rl` extras with the pins above | FR-037. Out of the `metrics` and `all` aggregates |
| `docs/docs.json` | Register `advanced/roastme` in the Advanced nav | An unlisted page is not published |
| `docs/docs-sync.json` | Register `sdks/python/advanced/roastme` in the Advanced group | A second navigation registry, the one that publishes to the docs site. Listing only the first would leave the page unpublished |

Five rows. `src/gaussia/metrics/__init__.py` is **not** among them, because nothing here is a metric: the
public surface is `from gaussia.generators.roastme import ProbeLibrary, Profiler, Exploiter`.

Three files deliberately **not** modified, worth stating because two are where this feature would most
plausibly have leaked:

- `src/gaussia/llm/judge.py` — FR-009 and spec D4. Roast Me grades behind its own interface. The shared
  judge reads the first generated token, which is wrong for a reasoning model, but fixing it there would
  change behaviour for `role_adherence`, which depends on it. That is a separate decision with its own
  blast radius. Note also what the shared judge does *not* do: it raises `LogprobsNotSupportedError` rather than
  falling back. The fallback that exists lives in `role_adherence`'s scoring strategy and goes to
  structured output, where FR-008 requires sampling — so the violation-rate denominator stays
  provider-independent.
- `src/gaussia/generators/__init__.py` eagerly imports concrete classes. Registering the engines there
  would make `import gaussia.generators` require an embedder, breaking FR-037.
- `CHANGELOG.md` is generated by python-semantic-release and must not be hand-edited.

## Complexity Tracking

Two items need the reviewer's explicit call: the unchecked box in the Testing Gate (the policy-gradient
search has no automated coverage) and the open decision on which pluggable pieces ship. Both are stated
where they arise rather than repeated here.
