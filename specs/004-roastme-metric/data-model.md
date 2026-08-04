# Data Model: Roast Me

Every model is a Pydantic `BaseModel` in `src/gaussia/schemas/roastme.py`, except the abstractions,
which live in `core/` and are listed here only where their obligations shape a field.

## Entity map

```
user supplies                gaussia validates            gaussia produces
─────────────                ─────────────────            ────────────────
Principle ─┐
           ├─> BehavioralContract ──┐
Grader ────┘                        │
                                    ├─> GradedOutcome ──> WeaknessEntry ──> AssistantProfile
Document ──┐                        │        ▲                                    │
PluginSpec ├─> Catalogue ─> Probe ──┴────────┘                                    │
StrategySpec┘        (+ KnowledgeHook)                                            │
                          │                                                       ▼
                          └──> ProbeBatch (what the pipeline iterates)      Category
                                                                                  │
                                                                                  ▼
                                                              CategoryEvaluation ──> RoastDatasetRecord
                                                                                     FailureReport
```

The user side is data they write. The middle is what gaussia refuses to accept when malformed. The
right side is what a run emits.

---

## `Principle`

One rule of the behavioral contract.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | Stable name, recorded on every grade so a charge is traceable. |
| `weight` | `float` | `ge=0.0, le=1.0` | The severity weight `w_j` of Eq. 3. |
| `rubric` | `str` | `min_length=1` | The text handed to the grader. User-supplied per FR-006; gaussia never substitutes or appends to it. |
| `grader` | `Grader` | — | The bound grader. Required: FR-003 makes a principle with none fail at construction. |

`arbitrary_types_allowed` is needed because `grader` is an ABC instance, not a model.

## `BehavioralContract`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `principles` | `list[Principle]` | `min_length=1` | The set `Π`. |

Model validator (FR-001):

- weights sum to `1.0 ± 1e-9` — the tolerance is explicit so a contract assembled from decimals is not
  rejected for float noise;
- `id` values unique;
- every principle carries a grader.

Gaussia ships no contract instance (FR-002). The paper defines `Π` as an input to the method, so a
default would be the library deciding what counts as a failure.

## `Document`

One unit of the knowledge base. Not `connectors.RegulatoryDocument`: that is a two-field dataclass with
no identifier and no enumerability flag — see the plan for why extending it was rejected.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | Quoted in every hook derived from this document. |
| `content` | `str` | — | The source text. |
| `structured` | `bool` | — | Whether this document's knowledge boundary is enumerable. Decides which engines can establish absence over it, so invariant 7 turns on this field. |
| `kind` | `str \| None` | default `None` | Routes the document to the right extraction. |
| `metadata` | `dict` | default `{}` | Anything an engine or a report needs to carry. |

## `PluginSpec`

A risk family, mapping to one principle.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | Referenced by strategies, recorded on every probe and outcome. |
| `name` | `str` | — | Documentation. Not consumed by any logic. |
| `description` | `str` | — | Documentation. |
| `principle` | `str` | `min_length=1` | The `Principle.id` this family attacks. Resolved against the contract at catalogue validation. |

## `StrategySpec`

An interaction pattern.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | The aggregation descriptor `z` of the weakness map. |
| `name` | `str` | — | Documentation. |
| `description` | `str` | — | Documentation. |
| `plugin` | `str \| None` | default `None` | The risk family served. **`None` means control** (FR-026): no principle under test, so the probes it produces are excluded from every violation-rate aggregate. |
| `entity_kind` | `str` | `min_length=1` | The entity type this strategy needs. Must match what the engine can extract or retrieve. |
| `transform` | `str` | must be a registered key | How the real entity becomes the probe's premise. Stays a string because it is user configuration; the registry resolves it to a `Transform` **once**, at validation, and nothing branches on it afterwards. |
| `doc` | `int` | `ge=0, le=1` | The expected grounding label of the resulting hook. |
| `phrasing_hint` | `str` | — | Injected into the generation prompt, so it has to be in the knowledge base's language. |

## `Catalogue`

| Field | Type | Constraints |
|---|---|---|
| `plugins` | `list[PluginSpec]` | `min_length=1` |
| `strategies` | `list[StrategySpec]` | `min_length=1` |

Validated against a contract before generation runs (FR-025, SC-005). Every rejection is up front, not
mid-run:

| Rejection | Condition |
|---|---|
| dangling principle | a `PluginSpec.principle` with no matching `Principle.id` |
| dangling plugin | a `StrategySpec.plugin` that is neither `None` nor a known `PluginSpec.id` |
| unknown transform | a `StrategySpec.transform` absent from the registry |
| out-of-range label | a `doc` outside `{0, 1}` |
| duplicate identifier | repeated `id` within plugins or within strategies |

## `KnowledgeHook`

The structured provenance of a probe against the knowledge base.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `kind` | `str` | `min_length=1` | The entity type leaned on. |
| `references` | `str` | `min_length=1` | The entity itself, real or invented. |
| `doc` | `int` | `ge=0, le=1` | 1 documented, 0 invented. **The ground truth of the downstream test.** |
| `how` | `str` | — | The transformation that produced it, carried from the strategy. |
| `base_entity` | `str \| None` | default `None` | The real entity it was derived from, when the strategy mutated one. |
| `principle` | `str \| None` | default `None` | The principle under test. `None` for a control, consistent with `StrategySpec.plugin`. |
| `verified` | `bool \| None` | default `None` | Whether a `HookVerifier` confirmed `doc`. **`None` means unverified, which is not the same as `False`** — collapsing the two would turn "nobody checked" into "the label is wrong". |
| `absence_reliable` | `bool` | default `True` | Set `False` by an engine that cannot decide absence (FR-023), so an unreliable label is never indistinguishable from a confirmed one. |

## `Probe`

What the Probe Library emits.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | Quoted in every graded outcome. |
| `query` | `str` | `min_length=1` | The question to send. |
| `hook` | `KnowledgeHook` | — | Provenance. |
| `plugin` | `str \| None` | default `None` | `None` means control. |
| `strategy` | `str` | `min_length=1` | The descriptor used for aggregation. |
| `attrs` | `list[str]` | default `[]` | The natural-language attributes the probe exhibits, which is how the Exploiter grounds a category in it. |
| `engine` | `str \| None` | default `None` | Which engine produced it. Keeps the trade-off measurable after composition (FR-022). |
| `meta` | `dict` | default `{}` | What a grader needs to judge: the real value, the false value asserted, the real and false chains. |

## `ProbeBatch(Batch)`

The pipeline's unit. Subclasses `Batch` the way `PrivacyBatch` does, so `Retriever` yields ordinary
`Dataset`s and nothing in the pipeline needs to know about Roast Me.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `probe` | `Probe` | — | The probe this turn came from. Composed rather than flattened, so the tags have one home. |
| `ground_truth_assistant` | `str` | default `""` | **Overrides the required field on `Batch`.** A probe is a trap: there is no reference answer, and forcing a meaningless one would be friction with no purpose. The type is unchanged, so LSP holds. |
| `failed` | `bool` | default `False` | Set by the target implementation when the exchange failed at transport level (FR-018). |

Model validator: `query == probe.query`. The two exist because `Batch` owns `query` and `Probe` owns
its own; enforcing equality means they cannot drift instead of hoping they do not.

`qa_id` inherited from `Batch` carries `probe.id`.

## `PrincipleGrade`

One grader's estimate for one principle on one response.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `principle` | `str` | `min_length=1` | Which principle was charged. |
| `score` | `float` | `ge=0.0, le=1.0` | `π̂_j(x, r)`. |
| `method` | `str` | `min_length=1` | How the verdict was obtained — the logprob path, or the sampling fallback (FR-005, FR-008). |
| `model` | `str \| None` | default `None` | The grader's model identity, `None` for a rule-based grader. |
| `evidence` | `dict` | default `{}` | The raw basis: the per-token logprobs, or the sampled votes. What makes a grade auditable rather than asserted. |

## `GradedOutcome`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `probe_id` | `str` | `min_length=1` | — |
| `response` | `str` | — | What the assistant answered. |
| `grades` | `list[PrincipleGrade]` | default `[]` | The per-principle grades FR-004 requires be retained. Empty when ungraded. |
| `violation` | `float \| None` | `ge=0.0, le=1.0` | `v(x, r)`. **`None` is the only representation of "ungraded"** (FR-016): every aggregate filters on it, so a failed exchange moves neither numerator nor denominator (SC-008). A `0.0` would have meant "graded, no violation". |
| `evidence_available` | `bool` | default `False` | Whether the grader had knowledge-base evidence to check against (FR-015). Where this is `False`, the score reflects the model's own knowledge and the record has to say so. |
| `scoreable` | `bool` | default `True` | `False` for a control. Derived at construction from the probe's plugin, never from an identifier (FR-011). |

## `WeaknessEntry`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `principle` | `str` | `min_length=1` | — |
| `descriptor` | `str` | `min_length=1` | The strategy `id` — the `z` of `ω(π_j, z)`. |
| `rate` | `float \| dict` | — | Whatever the injected `StatisticalMode.rate_estimation` returns: a point estimate under `FrequentistMode`, a posterior summary under `BayesianMode`. |
| `n` | `int` | `ge=0` | Sample size behind the rate (FR-012). Travels with it so a descriptor resting on a handful of probes cannot be read as settled. |
| `dispersion` | `float \| dict \| None` | default `None` | The spread, from the same statistical mode. |
| `descriptor_prose` | `str` | — | The natural-language description that crosses to the Exploiter. Invariant 3 forbids sending the identifier. |

## `AssistantProfile`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `weaknesses` | `list[WeaknessEntry]` | default `[]` | The map `ω`. |
| `hooks` | `list[KnowledgeHook]` | default `[]` | The retained hooks `H`, each with its `doc` label. |

FR-013 makes this the only artifact passed to the Exploiter, and invariant 3 makes it carry prose
rather than internal identifiers.

## `RoastMeProfile(BaseMetric)`

What `RoastMeProfiler` appends to `self.metrics`. Inherits `session_id` and `assistant_id`.

| Field | Type | Notes |
|---|---|---|
| `profile` | `AssistantProfile` | The profile `θ`. |
| `outcomes` | `list[GradedOutcome]` | Every outcome, controls and ungraded included: excluded from the rates, kept in the record (FR-011, FR-016). |
| `overall_rate` | `float \| dict` | The violation rate over scoreable, graded outcomes. |
| `n_scoreable` | `int` | How many outcomes entered the rates. |
| `n_ungraded` | `int` | How many were dropped for a failed exchange, so a silent shrink is visible. |

## `Category` and `CategoryEvaluation`

| `Category` field | Type | Notes |
|---|---|---|
| `attributes` | `list[str]` | `min_length=1`. The ordered conjunction `c = (a₁…a_ℓ)`. |
| `provenance` | `list[str]` | Per attribute, the weakness entry or hook that induced it (FR-028). Same length as `attributes`. |

| `CategoryEvaluation` field | Type | Notes |
|---|---|---|
| `category` | `Category` | — |
| `queries` | `list[str]` | The sampled queries. |
| `responses` | `list[str]` | Their responses. |
| `violations` | `list[float]` | Per-query `v`, already gated: a query below `κ` contributes exactly `0.0` (FR-030). |
| `on_profile` | `list[bool]` | Which queries passed the `κ` gate, so the zeros above are explainable. |
| `realism_gap` | `float` | `D̂(Q_c‖N)`. |
| `score` | `float` | `S(c)`. |
| `n` | `int` | Evaluations behind the score (FR-029). Reported alongside so a single-evaluation category is not read as robust. |
| `dropped_attributes` | `list[str]` | What refinement removed as incidental (FR-032). |

## `RoastDatasetRecord` and `FailureReport`

| `RoastDatasetRecord` field | Type | Notes |
|---|---|---|
| `query` | `str` | — |
| `response` | `str` | — |
| `violation` | `float \| None` | `None` for an ungraded exchange, same convention as `GradedOutcome`. |
| `principles_charged` | `list[str]` | — |
| `rationale` | `list[PrincipleGrade]` | The grades that justify the score. |
| `evidence` | `str \| None` | The supporting or contradicting source text, when knowledge-grounded. `None` in black-box mode — which US2 requires be distinguishable from "sought and not found", so `evidence_available` on the outcome carries that distinction rather than overloading `None`. |

| `FailureReport` field | Type | Notes |
|---|---|---|
| `categories` | `list[CategoryEvaluation]` | Ranked by `S(c)` descending (FR-035). |
| `queries_over_threshold` | `list[RoastDatasetRecord]` | Individual queries at or above `τ`, surfaced alongside the category verdict so "no category broke it reproducibly" stays distinguishable from "it answered correctly". |
