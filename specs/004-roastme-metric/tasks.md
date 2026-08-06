# Tasks: Roast Me — Profile-then-Exploit Adversarial Evaluation

**Input**: `specs/004-roastme-metric/plan.md`
**Prerequisites**: `plan.md`, `data-model.md`, `spec.md`

## Format: `[ID] [P?] [Story] Description (FR / SC)`

- **[P]**: can run in parallel — different files, no dependency between them
- **[Story]**: which user story the task serves; unmarked tasks serve all of them
- Every task carries the requirement or success criterion it exists for. Every path comes from the plan's
  file tables. Tests are written and verified failing before the code that satisfies them.

---

## Phase 1: Schemas & Contracts

The ten abstractions carry no logic, so they can be written at once. None may import an embedder or a
graph library: FR-037 requires them importable with no extra installed.

- [ ] T001 [P] `Grader` in `src/gaussia/core/grader.py` — one principle's violation for a query and a
      response, with the evidence behind it (FR-005)
- [ ] T002 [P] `ProbeEngine` in `src/gaussia/core/probe_engine.py` — can-handle, the entity kinds it
      declares it handles, plus documents and catalogue to tagged probes (FR-019, FR-021, FR-025)
- [ ] T003 [P] `EntityEnumerator` in `src/gaussia/core/entity_enumerator.py` — the entities of a kind that
      exist in the base. Interface only; gaussia ships no implementation (FR-019, D14)
- [ ] T004 [P] `HookVerifier` in `src/gaussia/core/hook_verifier.py` — confirm a hook's `doc` label
      against the corpus (FR-019)
- [ ] T005 [P] `Transform` in `src/gaussia/core/transform.py` — a real entity to a probe's premise (FR-019)
- [ ] T006 [P] `TargetAssistant` in `src/gaussia/core/target_assistant.py` — a query in, a `TargetResponse`
      out, or the exchange marked failed (FR-017, FR-018)
- [ ] T007 [P] `QueryGenerator` in `src/gaussia/core/query_generator.py` — a category's attributes to
      concrete queries. Separate from the search so invariant 5 is checkable (FR-033)
- [ ] T008 [P] `OnProfileFilter` in `src/gaussia/core/on_profile_filter.py` — how on-profile one query is,
      which is what the `κ` gate compares against (FR-030)
- [ ] T009 [P] `RealismEstimator` in `src/gaussia/core/realism_estimator.py` — distance from the
      natural-query prior, without querying the assistant (FR-031)
- [ ] T010 [P] `CategorySearch` in `src/gaussia/core/category_search.py` — a profile to scored categories
      (FR-033)
- [ ] T011 Every model of `data-model.md` in `src/gaussia/schemas/roastme.py`, with the validators it
      specifies: contract weights summing to `1 ± 1e-9` and one grader per principle (FR-001, FR-003);
      `Probe.plugin` empty if and only if `hook.principle` is empty (FR-011); `TargetResponse` rejecting
      empty content that is not marked failed (FR-016); `doc` in `{0,1}`; `Category.provenance` the same
      length as `attributes` (FR-028); and the config models carrying every field
      `data-model.md` lists as required — the verdict surface forms, `reasoning_budget`,
      `fallback_samples`, `top_logprobs`, `tau`, `lambda_`, `kappa`, `delta`, `eta`,
      `queries_per_category` and `pool_size` — with no defaults (FR-006)
- [ ] T012 Re-export the ten abstractions from `src/gaussia/core/__init__.py` and extend `__all__` — they
      carry no heavy imports, so they belong on the re-exported side of that file's own rule (FR-037)
- [ ] T013 Add the schema pointer to the `src/gaussia/schemas/__init__.py` docstring — **no eager import** (FR-037)
- [ ] T014 Add the `roastme` extra to `pyproject.toml`:
      `["sentence-transformers>=5.0.0", "torch>=2.0.0", "networkx>=3.0"]`. Out of the `metrics` and `all`
      aggregates (FR-037)

**Checkpoint**: `uv run python -c "import gaussia.core"` succeeds with no extra installed.

---

## Phase 2: Tests (Red Phase)

Nothing here may pass. The package markers and the doubles come first because everything else needs them.

- [ ] T015 [P] Package markers: `tests/generators/roastme/__init__.py`, `tests/graders/__init__.py`,
      `tests/fixtures/roastme/__init__.py`. Every test directory in this repo has one
- [ ] T016 Deterministic doubles in `tests/fixtures/roastme/` — one per interface: a grader with
      prescribed per-principle scores, a grader whose provider exposes no usable logprobs (SC-009), a
      target returning recorded responses, a target reporting a failed exchange (SC-008), a probe engine,
      an entity enumerator, a hook verifier, a query generator, an on-profile filter with prescribed
      scores, a realism estimator with prescribed distances, a category search
- [ ] T017 Hand-computed probe fixtures in `tests/fixtures/roastme/` — the expected `v`, rates and
      standard errors worked out from the paper's equations, so assertions are arithmetic and not
      snapshots (SC-001, SC-002)
- [ ] T018 [P] [US1] `tests/generators/roastme/test_scoring.py` — `v` with its per-principle grades
      retained (FR-004), the weakness-map rate and its standard error (FR-012), `S(c)` (FR-029),
      the `κ` gate zeroing a query (FR-030), the `δ` budget discarding a category (FR-031), refinement to
      the minimal sub-conjunction (FR-032) (SC-001, SC-002, SC-007)
- [ ] T019 [P] [US1] `tests/generators/roastme/test_profiler.py` — control exclusion driven purely by a
      strategy with no plugin, including the pair of probes over the same documented entity that land on
      opposite sides of that line (FR-011, SC-003); a failed exchange recorded ungraded and moving neither
      numerator nor denominator (FR-016, SC-008); the profile crossing with prose and no strategy
      identifiers (FR-013)
- [ ] T020 [P] [US1] `tests/graders/test_logprob.py` — the verdict located at the last matching token, not
      the first (FR-007); discarded when the model's own final answer does not parse to one (FR-007); the
      sampling fallback marked as such (FR-008, SC-009)
- [ ] T021 [P] [US2] `tests/generators/roastme/test_dataset.py` — the conversion fills every required
      framework field as `data-model.md` specifies, `evidence_available` survives it, and `Toxicity`
      consumes the result with no change to it — chosen because it reads the assistant's answer alone,
      where `Humanity` and `Vision` score against an expected answer this conversion leaves empty
      (FR-034, SC-006)
- [ ] T022 [P] [US3] `tests/generators/roastme/test_catalogue.py` — every rejection path: dangling
      principle, dangling plugin, unknown transform, `doc` outside `{0,1}`, duplicate identifier, and an
      `entity_kind` no configured engine handles (FR-025, FR-026, SC-005)
- [ ] T023 [P] [US3] `tests/generators/roastme/test_library.py` — engine composition and duplicate merging
      with the originating engine recorded (FR-022); the graph engine's absence labels confirmed by exact
      enumeration and the retrieval engine's marked unreliable (FR-023, SC-004); the enumeration engine
      refusing to run with no enumerator injected (D14); no knowledge base yielding domain-agnostic probes
      with an empty hook (FR-024)
- [ ] T024 [P] [US4] `tests/generators/roastme/test_exploiter.py` — the failure report ranked by `S(c)`
      with the individual queries at or above `τ` surfaced (FR-035), an auditable category-evaluation
      record carrying the per-query rationale (FR-036), and the refusal to run when no on-profile filter
      is supplied
- [ ] T025 [P] Contract tests: every double in T016 satisfies its `core/` interface, so a third-party
      implementation has an executable definition of conformance (FR-019)
- [ ] T026 [P] Import isolation (FR-037, SC-011): every interface imports with no extra installed, and
      `import gaussia.generators.roastme` succeeds too — which fails if `probes/__init__.py` ever
      re-exports an engine
- [ ] T027 [P] The default suite is hermetic (SC-010): no test opens a socket, requires credentials or
      requires a GPU. Enforce rather than assert by convention — anything that needs one of those carries
      a marker and is deselected by default

**Checkpoint**: `uv run pytest tests/generators/roastme tests/graders` runs and **every test fails**.

---

## Phase 3: US1 — Profiler (Green)

- [ ] T028 [US1] `src/gaussia/generators/roastme/searches/scoring.py` — the violation score, the
      weakness-map rate with its sample size and standard error, `S(c)`, the `κ` and `δ` comparisons and
      refinement. Pure functions over values: no I/O, no model (FR-004, FR-012, FR-029…FR-032)
- [ ] T029 [US1] `src/gaussia/graders/logprob.py` and `__init__.py` — the shipped base grader: last
      verdict-shaped token in the sequence, discarded when the final answer does not independently parse
      to one, sampling fallback over `k` marked as fallback-derived (FR-007, FR-008)
- [ ] T030 [US1] `src/gaussia/generators/roastme/profiler.py` — drive the injected target over a probe
      set, grade each response, aggregate the weakness map and the retained hooks, strip strategy
      identifiers from what crosses, and emit `ProfilerResult` (FR-010…FR-016)
- [ ] T031 [US1] T018, T019 and T020 pass

**Checkpoint**: a profile can be built with no credentials, no network and no GPU, by pointing the target
interface at a recorded response set.

---

## Phase 4: US2 — Roast Dataset (Green)

- [ ] T032 [US2] `src/gaussia/generators/roastme/dataset.py` — the output-boundary conversion, filling
      every required framework field as specified and carrying the record's own fields as turn metadata
      (FR-034)
- [ ] T033 [US2] T021 passes

**Checkpoint**: a run's output is consumable by an existing metric unmodified.

---

## Phase 5: US3 — Probe Library (Green)

- [ ] T034 [US3] `src/gaussia/generators/roastme/probes/transforms.py` — the four `Transform`
      implementations and the registry that resolves the catalogue string **once**, so nothing branches on
      it afterwards (FR-025)
- [ ] T035 [US3] `src/gaussia/generators/roastme/probes/catalogue.py` — validate a catalogue against a
      contract **and against the configured engines' declared entity kinds** before generation runs
      (FR-025, FR-026, FR-027)
- [ ] T036 [P] [US3] `src/gaussia/generators/roastme/probes/retrieval.py` — the retrieval engine,
      recording on every absence probe that its label is unreliable (FR-021, FR-023)
- [ ] T037 [P] [US3] `src/gaussia/generators/roastme/probes/graph.py` — the graph engine, confirming
      absence from the complete graph (FR-021)
- [ ] T038 [P] [US3] `src/gaussia/generators/roastme/probes/grag.py` — the multi-hop engine, false
      premises spanning several entities (FR-021, FR-022)
- [ ] T039 [P] [US3] `src/gaussia/generators/roastme/probes/enumeration.py` — the enumeration engine,
      opt-in, refusing to run until an `EntityEnumerator` is injected. Reads `Document.structured` to know
      whether a document's boundary can be enumerated (FR-022, D14)
- [ ] T040 [US3] `src/gaussia/generators/roastme/probes/library.py` and `__init__.py` — compose the
      applicable engines, merge duplicates, record the originating engine, run the first three by default
      and the enumeration engine only when configured, and return domain-agnostic probes with an empty
      hook when there is no knowledge base. **`probes/__init__.py` must not import any engine** (FR-020,
      FR-022, FR-024, FR-037)
- [ ] T041 [US3] T022 and T023 pass

**Checkpoint**: probes can be generated from a knowledge base and a user-supplied catalogue.

---

## Phase 6: US4 — Exploiter (Green)

- [ ] T042 [P] [US4] `src/gaussia/generators/roastme/searches/realism.py` — the base realism estimator:
      expected cosine distance from a prior pool through an injected embedder, which is the instantiation
      the paper gives. Composes the framework's existing embedder rather than encoding vectors itself
      (FR-031)
- [ ] T043 [US4] `src/gaussia/generators/roastme/searches/attribute_iteration.py` — the training-free
      search: the common attributes of the highest-scoring pool, then attribute subsets. No GPU (FR-033)
- [ ] T044 [US4] `src/gaussia/generators/roastme/exploiter.py` and `src/gaussia/generators/roastme/searches/__init__.py` — compose
      search, query generator, on-profile filter, estimator and target; **refuse to run with no filter
      supplied**, since the `κ` gate would otherwise pass every query silently; emit the Roast Dataset and
      the failure report with the individual queries at or above `τ` surfaced alongside the category
      verdict (FR-030, FR-034…FR-036)
- [ ] T045 [US4] `src/gaussia/generators/roastme/__init__.py` — re-export `ProbeLibrary`, `Profiler`,
      `Exploiter`. **Do not touch `src/gaussia/generators/__init__.py`**: it imports eagerly, and
      registering there would make `import gaussia.generators` require an embedder (FR-037)
- [ ] T046 [US4] The full default suite passes, and T026 and T027 still hold

**Checkpoint**: the whole subsystem runs end to end on any machine, offline, against a recorded target.

---

## Phase 7: The reinforcement-learning search (opt-in)

Behind `gaussia[roastme-rl]`. Last, so every earlier checkpoint stays verifiable without a GPU.

- [ ] T047 Add the `roastme-rl` extra to `pyproject.toml`:
      `["gaussia[roastme]", "peft>=0.10.0", "accelerate>=0.25.0", "trl>=0.8.0"]`. Out of the `metrics` and
      `all` aggregates (FR-037)
- [ ] T048 [US4] `src/gaussia/generators/roastme/searches/policy_gradient.py` — the policy-gradient search
      behind the same interface, leaving the query generator unmodified (FR-033). Marked `requires_gpu`,
      so the default suite deselects it. Its optimisation loop has no automated coverage; the criterion it
      optimises is covered by T018

---

## Phase 8: Documentation and example

- [ ] T049 `docs/advanced/roastme.mdx` — alongside `generators` and `prompt-optimizer`, not under
      `docs/metrics/`, because this is not a metric. Must state that no grader has been calibrated against
      human labels and that the figures are a judge-only measurement (FR-038), that the on-profile filter
      has no shipped implementation and why, and that the training-free search has no published result
      behind it
- [ ] T050 Register the page in **both** navigation registries: `advanced/roastme` in `docs/docs.json` and
      `sdks/python/advanced/roastme` in `docs/docs-sync.json`. The second is the one that publishes to the
      docs site, so listing only the first leaves the page unpublished (FR-038)
- [ ] T051 [P] `examples/roastme/catalogue/` — schema examples for `PluginSpec` and `StrategySpec`: the
      shape with domain-neutral prose, **not** a domain catalogue (FR-027)
- [ ] T052 [P] `examples/roastme/jupyter/` — a runnable notebook: build a contract, validate a catalogue,
      profile a recorded response set through the target interface. Mirrors `examples/privacy/jupyter/` (FR-014)

---

## Phase 9: Polish

- [ ] T053 `uv run ruff check .` passes
- [ ] T054 `uv run ruff format .` leaves no diff
- [ ] T055 `uv run mypy src/gaussia` passes
- [ ] T056 `uv run pytest` passes with the configured coverage floor

---

## Dependencies

- T001–T010 are independent of each other. T011 depends on them for the types it references; T012 depends
  on all ten existing.
- Phase 2 depends on Phase 1: a test cannot import an interface that does not exist.
- T015 and T016 block every other test task; T017 blocks T018 and T019.
- Phases 3 through 6 depend on Phase 2 being red. Within them T028 comes first, because the Profiler, the
  Exploiter and the weakness map all consume it.
- US1 (Phase 3) and US3 (Phase 5) are independent once Phase 2 is done and can proceed in parallel. US2
  (Phase 4) needs US1 for something to emit. US4 (Phase 6) needs US1 for a profile to search from.
- Phase 7 depends on T044, since the policy-gradient search plugs into the same exploiter.
- Phase 8 depends on Phase 6: documenting an interface that is still moving wastes the writing.
- Phase 9 depends on everything.
