# Feature Specification: Roast Me — Profile-then-Exploit Adversarial Evaluation

**Feature Branch**: `004-roastme-metric`
**Created**: 2026-08-03
**Status**: Planned

Roast Me is not a scalar metric. It is a search problem: given an assistant treated as a black box, find the *categories* of realistic interactions that make it violate its behavioral contract reproducibly. Three components in sequence — a **Probe Library** that turns a knowledge base into tagged adversarial probes, a **Profiler** that grades the responses into a weakness profile, and an **Exploiter** that searches for reproducible failure categories.

## Paper Reference

- **Paper**: `gaussia-labs/papers/papers/2026-06-roastme/` — §Theoretical Formulation defines everything below.
- **Paper PR**: gaussia-labs/papers#20, merged to `main`.
- **Implementation Issue**: N/A — tracked through the SDD gate commits on this branch, as `001-privacy-metric` was.
- **Reference implementation**: `experiments/roastme/` in gaussia-labs/pygaussia#17. It is the executable record of the published runs and stays frozen; it is not the surface specified here.

### Notation used below

Only the symbols the requirements reference. The paper carries the full formulation.

| Symbol | Meaning | Eq. |
|---|---|---|
| `Π = {π_j}`, `w_j` | the behavioral contract: principles and their severity weights, `Σ w_j = 1`. **An input, not a library constant** | 2 |
| `π̂_j ∈ [0,1]` | a **grader**: estimates whether principle `j` was violated | §Graders |
| `v(x,r) = Σ_j w_j · π̂_j(x,r)` | the **violation score** of one response | 3 |
| `doc(g) ∈ {0,1}` | the **grounding label** on a probe's knowledge hook: 1 if the referenced content exists in the knowledge base, 0 if the probe deliberately invents it | 6 |
| `θ = (ω, H)` | the **profile**: a weakness map `ω : Π × Z → [0,1]` over probe descriptors, plus the retained knowledge hooks `H` | 8, 9 |
| `C_θ`, `η` | the categories whose every attribute comes from a weakness at or above `η`, or references a hook in `H` | 10 |
| `c = (a₁…a_ℓ)` | a **category**: a conjunction of natural-language attributes | 11 |
| `S(c) = Φ̂ₙ(c) − λ·seₙ(c)` | the category score. A lower-confidence bound, so consistent failure outranks a lucky spike | 13 |
| `κ`, `δ`, `τ` | the on-profile gate on a query, the realism budget on a category, the pass threshold on `S(c)` | 15, 16, 14 |

### Invariants the paper imposes

These are methodological, not stylistic. Breaking any one invalidates the measurement.

1. The assistant is sampled, never inspected — no parameters, system prompts or internal tools (Eq. 1).
2. The knowledge base is reached **only** through the Probe Library. The Profiler and the graders never touch it (Fig. 1).
3. `θ` is the **only** artifact crossing from Profiler to Exploiter, and it carries readable prose rather than internal identifiers (`roastme.tex:1300-1303`).
4. The `doc` label must be honest, and derived from the generating engine's own knowledge of the base's boundary — never from the enumeration used for scoring. It is what lets a grader separate "invented something absent" from "described something present".
5. Under the policy-gradient search, optimisation pressure applies to the category generator only; the query generator stays frozen, which is what preserves realism (Eq. 17).
6. A verdict is read from the logprobs of a single-word answer, located by scanning the sequence for the **last** verdict-shaped token — the first token belongs to a reasoning model's preamble. Whether logprobs are usable at all is a property of the serving provider, not the model: across twelve model/provider combinations screened, the same weights failed on two and worked on a third (`roastme.tex:1176-1180`, `1283-1287`).

### Reported baseline

A production Spanish-language assistant answering questions about Argentina's Monotributo regime, running on retrieval grounding alone (empty identity, rules and purpose fields).

| Reported | Value | Artifact |
|---|---|---|
| Probes generated, composed across 4 engines | 204 | `results/level1_probes/` |
| Scoreable probes, controls excluded | **151** | `results/level2_profiler/` |
| Fail rate — three separate runs, one judge model each, 5 iterations | 12.6% / 15.2% / 7.9% | same |
| Fail rate by strategy | 8 descriptors, `n` from 3 to 89 | same |
| Profile distilled from the 151 graded probes | 8 weakness entries, 24 hooks | same |
| Category search at `τ=0.35` | 384 queries, 34 categories, **0 passing**, max `S(c)=0.154` | `results/level3_exploiter/20260729T164442Z-*` |
| Individual queries at or above `τ` | 33 of 384, while no *category* evaluation reached it | same |

Two arithmetic facts any implementation must reproduce: under the reported 7-principle contract, the two heaviest principles both firing at full confidence reach only `v = 0.343`, still under `τ`; and evaluations per category were severely skewed (one category 38 times, 21 of 34 exactly once).

**Standing.** Levels 1 and 2 are reported as findings. Level 3 is a validated *integration*, explicitly not a validated finding — sample size is the stated blocker, the runs are stochastic RL against a live assistant, and the trained adapters no longer exist. No grader has been calibrated against human labels; cross-judge agreement shows the judges agree with each other, not with a person.

## Decisions

Resolved at the clarify gate. The requirements below implement these without restating them.

| # | Was open | Decided |
|---|---|---|
| D1 | How much code ships versus the user writes | The library ships the working machinery. The user supplies **models and configuration**: a judge model, a generator model, target credentials, a knowledge base, a plugin/strategy catalogue. Extension contracts exist for replacing a piece, not as the ordinary path |
| D2 | Multi-grader aggregation | Out. One grader per principle. The paper's three columns are three runs, not an ensemble |
| D3 | Rubrics and prompts | User-supplied. A rubric *is* what the metric measures, so the paper's Spanish ones ship as examples, never as defaults |
| D4 | Extending the shared logprob judge | No. Roast Me grades behind its own contract; `llm/judge.py` is untouched and `role_adherence` is unaffected |
| D5 | Packaging | One extra, `gaussia[roastme]`, covering all three levels — the repo's one-extra-per-metric convention |
| D6 | Reaching the assistant under test | A target-assistant contract, plus a shipped adapter for the Alquimia runtime so a run works without writing transport code |
| D7 | Which of the paper's two category searches | Both, behind one contract. Training-free attribute iteration is the default and needs no GPU; policy gradient is available and is the only one that produced the reported numbers |
| D8 | Where the Exploiter sits | It is a **generator, not a metric**: it emits the Roast Dataset, which the pipeline then consumes. Matches the paper's own framing of the dataset as the deliverable, and bends no article of the constitution |
| D9 | The paper's Implementation Considerations section is commented out (`roastme.tex:1461`) | This spec proposes that surface and the paper follows in a later PR. The inversion is deliberate; the traceability chain stays open until that PR lands |
| D10 | Fate of `experiments/roastme/` | Frozen as the paper's record. The duplication is accepted; its README states it is not the SDK API |
| D11 | Surfacing the missing human calibration | Documented limitation in the metric docs. No field on the result, no calibration gate |
| D12 | Test fixtures | A small synthetic fixture with hand-computed values runs by default, offline. Verification against the real 204-probe artifacts is opt-in, per `001-privacy-metric`'s precedent |
| D13 | How a control probe is recognised | The probe declares it: no principle under test. The reference implementation keyed off a hardcoded plugin name, which cannot survive a user-supplied catalogue |
| D14 | No verdict token in the top logprobs | Fall back to sampling, marked as such. Raising would make the fail-rate denominator depend on provider behaviour |

Two upstream documents that contradicted `roastme.tex` are already fixed on `gaussia-labs/papers` `main` (`11775da`, `1673440`). `roastme.tex` is the sole authority for what follows.

## User Scenarios & Testing

### US1 — Profile an assistant from graded probes (P1)

An evaluator has tagged adversarial probes and the assistant's responses to them. They need to know **where** it is weak: which principles it breaks, under which kind of probe, how often, with what confidence, and which specific entities broke it.

**Why first**: the irreducible unit. The Exploiter consumes nothing but this profile, the Probe Library exists to feed it, the Roast Dataset is a serialisation of its outcomes. It is also the only component that runs with no knowledge base, no GPU and no live target.

**Independent test**: a deterministic stub grader over a hand-built probe fixture; every component of `v`, `ω` and `se` checked against values computed by hand. Offline.

1. **Given** a contract of `m` principles, **When** a response is graded, **Then** the outcome carries the aggregate `v` *and* the `m` per-principle grades, so a failure traces to the principle it breaks.
2. **Given** probes that declare no principle under test, **When** aggregates are computed, **Then** those probes are excluded from every rate while staying in the graded record — 151 of 204 in the reported run.
3. **Given** a reasoning judge whose first token is its reasoning preamble, **When** the verdict is extracted, **Then** the last verdict-shaped token in the sequence is used, and the result is discarded if the model's own final answer does not independently parse to a verdict.

### US2 — Emit the run as a reusable Gaussia dataset (P2)

The evaluator wants the audit to outlive the audit: the graded outcomes as a dataset the rest of Gaussia loads, replayable against a modified assistant so a fix is demonstrated by the scores that move.

**Why second**: the paper's stated primary deliverable, and the smallest increment that makes US1 useful to the framework. It needs US1 alone — a Roast Dataset record is what the Profiler already produces per probe.

**Independent test**: emit from the US1 stub run, load through the standard dataset contract, have an existing metric consume it unchanged; then change one grade and confirm the diff is isolated to that record.

1. **Given** an emitted dataset, **When** it is loaded through the SDK's dataset contract, **Then** an existing metric consumes it with no change to that metric.
2. **Given** a run in black-box mode, **When** records are emitted, **Then** the absence of evidence is distinguishable from evidence sought and not found.

### US3 — Generate tagged probes from a knowledge base (P3)

The evaluator points the library at their documentation or regulation, supplies a plugin and strategy catalogue, and gets probes for their domain — each tagged with what is documented and what is plausible but absent. They also need to know how far to trust each tag, because a false "this does not exist" silently corrupts everything downstream.

**Why third**: it removes the need to hand-author 204 probes, and it is where the paper's measured engine trade-off lives. Still downstream of US1 in value: an evaluator with their own probes gets a full profile without it, and black-box mode is explicitly supported.

**Independent test**: over a fixture base with a known enumerable entity set, check absence labels against exact enumeration. Generation is stochastic, so the test asserts label correctness and composition, never probe text.

1. **Given** several engines over one base, **When** their outputs compose, **Then** duplicates merge and each surviving probe records its originating engine, so the trade-off stays measurable after composition.
2. **Given** an engine that cannot establish absence reliably, **When** it emits an absence probe, **Then** that unreliability is recorded on the probe rather than being indistinguishable from a verified absence.
3. **Given** a catalogue whose identifiers are entirely the user's own, **When** generation and profiling run, **Then** no library behaviour depends on any of those identifiers.

### US4 — Search for reproducible failure categories (P4)

Not "which prompt broke it" but "which kinds of realistic question break it, repeatably" — categories in readable attributes, scored so consistency beats luck, gated so blatant asks do not count, budgeted so the queries still read like real traffic, and reduced to the attributes actually responsible.

**Why last**: the paper's headline contribution and its least settled result. Also the heaviest, and the only one whose reported numbers cannot serve as a numerical baseline. US1-US3 already deliver a working product.

**Independent test**: stub target, stub grader with prescribed violations, stub encoder with prescribed distances. `S(c)` against hand computation; a high-variance category rejected where a consistent lower-mean one passes; `κ` zeroing a query; `δ` discarding a passing category; refinement returning the minimal sub-conjunction. Training-free search, so no GPU.

1. **Given** two categories with equal mean violation and different variance, **When** both are scored, **Then** the lower-variance one ranks higher, and a category evaluated once carries its `n` so the unreliability of its penalty is visible rather than implied.
2. **Given** a category that passes `τ`, **When** it is refined, **Then** the result is the smallest sub-conjunction still satisfying `S(c') ≥ τ` and `D ≤ δ`, and the dropped attributes are reported as incidental.
3. **Given** a run where no category passes `τ`, **When** results are reported, **Then** "no category broke it reproducibly" is distinguishable from "the assistant answered correctly", by surfacing the individual queries at or above `τ` — 33 of 384 against 0 categories, in the reported run.

### Edge cases

- An error response or an outright refusal is **not** a contract violation. An upstream billing failure once returned a fixed 245-character error string that a naive grader scored as a violation.
- A query citing no knowledge-base entity leaves the grader nothing to check against, so its score reflects the judge's own knowledge. 42.4% of the reported level-3 queries. Must be visible on the record, not silently averaged in.
- A descriptor resting on 3 probes cannot distinguish "never failed" from "undersampled". Five of the eight reported descriptors are in that range.
- A principle in `Π` with no grader bound must fail loudly at construction, never contribute a silent zero to `v`.

## Requirements

### Functional Requirements

**Contract and grading**

- **FR-001**: The contract MUST be user-supplied: principles with non-negative weights validated to sum to `1 ± 1e-9`, each bound to exactly one grader. No default contract, and no aggregation of several graders into one principle's score.
- **FR-002**: `v(x,r) = Σ_j w_j · π̂_j(x,r)` MUST be computed with the per-principle grades retained on the outcome.
- **FR-003**: Graders MUST be substitutable behind one contract, and every grade MUST record the grader, the model and the verdict method that produced it.
- **FR-004**: A working logprob grader MUST ship, taking the user's judge model as a parameter. It MUST locate the verdict by scanning the full per-token sequence for the last match against the configured surface forms, and MUST discard that verdict when the model's own final answer does not independently parse to one.
- **FR-005**: When logprobs are unusable or no verdict token appears among them, the grader MUST fall back to sampling over `k` samples and mark the grades as fallback-derived.
- **FR-006**: Rubric text, verdict surface forms and reasoning budget MUST be user-supplied configuration; the paper's values ship as examples.
- **FR-007**: A principle with no grader bound MUST fail at contract construction.
- **FR-008**: An error or refusal response MUST NOT be gradeable as a violation.

**Profile**

- **FR-009**: The Profiler MUST accept probes and tags only, with no access path to the knowledge base.
- **FR-010**: A probe MUST declare whether it puts a principle under test. Probes that do not MUST be excluded from every violation-rate aggregate while remaining in the graded record. No library behaviour may depend on a template, plugin or strategy identifier.
- **FR-011**: Weakness entries MUST be keyed by `(principle, descriptor)` and MUST carry the rate, its sample size and its standard error.
- **FR-012**: The profile MUST carry the retained hooks with their `doc` label, MUST express weaknesses as natural-language descriptors, and MUST be the only artifact passed to the Exploiter.
- **FR-013**: The Profiler MUST support grading a frozen set of responses without contacting the assistant — the mode every reported level-2 number came from, and the only one needing no target credentials.
- **FR-014**: A graded outcome MUST record whether the grader had knowledge-base evidence to check the response against.

**Target assistant**

- **FR-015**: A target-assistant contract MUST be the only path by which this feature contacts the assistant under evaluation: send a query, optionally within a persistent session, receive a response.
- **FR-016**: An adapter for the Alquimia runtime MUST ship, configured by base URL, token and assistant identifier.

**Probe generation**

- **FR-017**: Particularisation MUST be the only component with knowledge-base access, exposing results solely as tagged probes.
- **FR-018**: Each probe MUST carry a hook whose `doc` label is derived from the generating engine's own knowledge of the base's boundary. The enumeration used for scoring MUST NOT be visible to the engine.
- **FR-019**: The paper's four engines MUST ship behind one contract, composable over one base with duplicate merging, each surviving probe recording its originating engine.
- **FR-020**: An engine that cannot establish absence reliably MUST record that limitation on the probe.
- **FR-021**: With no knowledge base, particularisation MUST return domain-agnostic probes with an empty hook through the same interface.
- **FR-022**: The plugin and strategy catalogue MUST be user-supplied. The library ships schema examples, not a domain catalogue.

**Category search**

- **FR-023**: A category MUST be an ordered conjunction of natural-language attributes, each traceable to the weakness entry or hook that induced it.
- **FR-024**: `S(c)` MUST be computed per Eq. 13 and reported with its `n`.
- **FR-025**: A query below `κ` MUST contribute exactly 0 to its category's score.
- **FR-026**: The realism gap MUST be computed without querying the assistant, and a category over `δ` MUST be discarded regardless of `S(c)`. The estimator MUST be substitutable, since the search depends on it only through `δ`.
- **FR-027**: Refinement MUST return the minimal sub-conjunction satisfying both thresholds, reporting the dropped attributes.
- **FR-028**: Both search procedures MUST ship behind one contract, the training-free one as default. Documentation MUST state that only policy gradient produced the paper's numbers, and under it the query generator MUST remain unmodified.

**Outputs**

- **FR-029**: The Roast Dataset MUST carry one record per query — query, response, violation score, principles charged, grader rationale, and the supporting or contradicting evidence when knowledge-grounded — and MUST be loadable through the SDK's existing dataset contract and consumable by existing metrics unmodified.
- **FR-030**: A failure report MUST rank categories by `S(c)` with evaluation counts and representative samples, and MUST surface the individual queries at or above `τ` alongside the category verdict.
- **FR-031**: A category evaluation record MUST be auditable: the attributes proposed, the queries, the responses, the per-principle rationale, and the realism and on-profile checks.

**Packaging and documentation**

- **FR-032**: Every dependency this feature adds MUST sit behind one optional extra. Importing the rest of the framework MUST NOT require them.
- **FR-033**: The metric documentation MUST state that no grader has been calibrated against human labels and that the reported figures are a judge-only measurement.

### Key Entities

Beyond the notation above, the data the SDK carries:

- **Grader / Verdict**: a grade for one principle on one `(context, response)` pair, with its method, model identity and raw evidence — the per-token logprobs, or the sampled votes.
- **TargetAssistant**: the contract for reaching the assistant under evaluation.
- **Probe**: a concrete query, its knowledge hook, the principles it targets (empty for a control), the descriptor used for aggregation, the originating engine.
- **GradedOutcome**: a probe, its response, the per-principle grades, the aggregate `v`, and whether evidence was available.
- **CategoryEvaluation**: an attribute conjunction with each attribute's provenance, its sampled queries and responses, per-query violations, the filter and realism checks, and `S(c)` with its `n`.
- **RoastDatasetRecord**: the deliverable's unit.
- **FailureReport**: categories ranked by `S(c)`, plus the individual-query view.

### SDK Pipeline Fit

- **New base classes**: three contracts — grader, probe engine, target assistant. None has an equivalent in `core/`, whose detector, guardian and corpus-connector contracts are shaped for span prediction, bias classification and document loading.
- **New components**: the **Profiler is a metric** and emits its profile through the standard result channel. The **Probe Library and the Exploiter are generators** (D8): their product is a dataset the pipeline then consumes.
- **New schemas**: the entities above.
- **New strategies**: grading, probe engines (composed rather than selected), the realism estimator, the category search.
- **Existing patterns affected**: none. D4 leaves the shared logprob judge untouched. The precedent `role_adherence` set — a scoring Strategy with a logprob path and a non-logprob fallback — is followed, not modified.
- **Pipeline**: US1 and US2 fit `Retriever → Dataset → batch()` natively, since grading frozen responses is an evaluation pass over a static dataset. US3 and US4 are generation and sit with the generator machinery. No article is bent.

## Success Criteria

- **SC-001**: Every component of `v`, `ω`, the per-descriptor rates and `se` reproduces to within `1e-9` of hand computation over a synthetic fixture. Correctness rests on hand computation, not on re-running live judges — provider drift makes the published 12.6% / 15.2% / 7.9% non-reproducible on demand.
- **SC-002**: Under the reported 7-principle contract, the two heaviest principles at 1.0 and the rest at 0 yields `v = 0.343 ± 1e-9`, below `τ = 0.35`.
- **SC-003**: Controls are excluded purely through what each probe declares. No library code and no test references a template, plugin or strategy name.
- **SC-004**: An opt-in check against the published artifacts reproduces 151 scoreable of 204 and the reported profile shape — 8 weakness entries, 24 hooks — at the reported `η`.
- **SC-005**: Over a fixture base with a known entity set, the enumerating engine reaches 100% absence-label precision and the retrieval engine's absence labels are marked unreliable.
- **SC-006**: A Roast Dataset emitted from a run is consumed end-to-end by an existing metric with no change to that metric.
- **SC-007**: `S(c)`, the `κ` gate, the `δ` budget and refinement all verify against hand-computed fixtures with stub target, grader and encoder.
- **SC-008**: The default suite runs offline, with no target credentials and no GPU. Everything needing a network, a GPU or the published artifacts is opt-in.
- **SC-009**: Importing any other part of the framework succeeds with the `roastme` extra uninstalled.

## Assumptions

- The paper is the upstream source of truth for methodology. Where it was silent, the Decisions table records the call rather than leaving it implicit.
- The published artifacts stay available in `gaussia-labs/papers` for SC-004. The default suite does not depend on them.
- Judge models are reached through providers the framework already integrates. Whether a provider exposes usable logprobs is probed at runtime, not encoded in an allowlist.
- Nothing in the formulation is language-specific, and FR-006 keeps every model-facing string in the user's hands. The reported scenario happens to be Spanish.
