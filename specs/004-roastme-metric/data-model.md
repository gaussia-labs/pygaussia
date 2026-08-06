# Data Model: Roast Me

Every model is a Pydantic `BaseModel` in `src/gaussia/schemas/roastme.py`, except the abstractions,
which live in `core/` and are listed here only where their obligations shape a field.

## Entity map

```
user supplies              gaussia validates          gaussia produces
─────────────              ─────────────────          ────────────────
Principle ─┐
           ├─> BehavioralContract ─┐
Grader ────┘                       │
                                   ├─> GradedOutcome ─> WeaknessEntry ─> AssistantProfile
Document ──┐                       │        ▲                                  │
PluginSpec ├─> Catalogue ─> Probe ─┴────────┤                                  │
StrategySpec┘       (+ KnowledgeHook)       │                                  ▼
                                            │                              Category
TargetAssistant ─────> TargetResponse ──────┘                                  │
 (the user's adapter)                                                          ▼
                                                        CategoryEvaluation ─> RoastDatasetRecord
                                                                              FailureReport
                                                                                   │
                                                                                   ▼
                                                                      the framework's dataset shapes
                                                                      (output boundary, FR-034)
```

The user side is data they write plus one adapter they implement. The middle is what gaussia refuses to
accept when malformed. The right side is what a run emits — and the last hop is the only place the
framework's own `Batch` and `Dataset` appear, because that is where Roast Me hands off to the pipeline.

---

## `Principle`

One rule of the behavioral contract.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `str` | `min_length=1` | Stable name, recorded on every grade so a charge is traceable. |
| `weight` | `float` | `ge=0.0, le=1.0` | The severity weight `w_j` of `eq:violation`. |
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

## Configuration

The thresholds the method takes as parameters. The paper defines them as inputs and fixes no values, so
they are required rather than defaulted: a default would be gaussia deciding how hard a category has to
fail before it counts. Grouped per component so nothing carries a knob it does not use.

**`GraderConfig`** — for the shipped logprob grader (FR-006, FR-007, FR-008)

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `positive_tokens` | `tuple[str, ...]` | `min_length=1` | The surface forms that mean "violated". Plural because a tokenizer splits the same word several ways. |
| `negative_tokens` | `tuple[str, ...]` | `min_length=1` | The forms that mean "did not violate". |
| `reasoning_budget` | `int` | `ge=1` | Tokens allowed before the verdict token is expected. A reasoning model needs room to finish; too little and the verdict is never reached. |
| `fallback_samples` | `int` | `ge=1` | The `k` of the sampling fallback. |
| `top_logprobs` | `int` | `ge=1` | How many alternatives to request per token. |

**`ExploiterConfig`**

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `tau` | `float` | `ge=0.0, le=1.0` | `τ`: the score a category must reach to pass. |
| `lambda_` | `float` | `ge=0.0` | `λ`: how hard inconsistency is penalised in `S(c)`. |
| `kappa` | `float` | — | `κ`: the on-profile score a query must reach to count at all. |
| `delta` | `float` | `ge=0.0` | `δ`: how far a category's queries may drift from the natural-query prior. |
| `eta` | `float` | `ge=0.0, le=1.0` | `η`: how strong a weakness must be to be worth attacking. |
| `queries_per_category` | `int` | `ge=1` | The `n` behind `S(c)`. Small `n` makes the standard-error penalty unreliable in both directions at once. |
| `pool_size` | `int` | `ge=1` | How many of the highest-scoring query/response pairs the training-free search keeps before intersecting their attributes. A method parameter the paper leaves open, so it is required rather than invented at implementation time. |

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
| `entity_kind` | `str` | `min_length=1` | The entity type this strategy needs, in the **user's own vocabulary** — gaussia never interprets it. Validated against what the configured engines declare they handle, so a typo fails loudly instead of yielding no probes. |
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
| unhandled entity kind | a `StrategySpec.entity_kind` no configured engine declares it can extract or retrieve |

The last one is the only rejection whose condition depends on the engines rather than on the catalogue
alone, and it exists because `entity_kind` is an opaque string to gaussia. Without it, a plural typo
would validate cleanly and produce an empty probe set with no error — the failure mode hardest to notice.

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

Model validator: `plugin is None` if and only if `hook.principle is None`. Control-ness would otherwise be
encoded in two places that can silently disagree, and FR-011 makes it load-bearing for every aggregate — a
probe with an empty plugin but a principle set would be excluded from the rates while still charging a
principle.

## `TargetResponse`

What the user's target adapter returns. The adapter reaches the assistant however it can — a hosted API,
a local model, a browser page, a recording — and hands back this shape. Gaussia consumes nothing else
about the assistant.

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `content` | `str` | — | What the assistant answered. Empty is legitimate only when `failed` is set. |
| `failed` | `bool` | default `False` | The exchange failed at transport level: an error status, an empty body, a payload shaped like an error. Only the adapter can recognise this (FR-018), and setting it makes the Profiler record the outcome ungraded (FR-016). |
| `failure_reason` | `str \| None` | default `None` | Free-form, for the record. Never parsed. |
| `session_id` | `str \| None` | default `None` | Set when the adapter maintains a persistent conversation, so a multi-turn probe can continue one. |
| `raw` | `dict` | default `{}` | Whatever the adapter wants preserved for auditing — the provider's payload, timings, token counts. |

Model validator: `failed` implies nothing about `content`, but `content` empty with `failed` unset is
rejected. An adapter that returns nothing without saying why would otherwise produce a graded outcome
over an empty string, which is exactly the silent failure FR-016 exists to prevent.

There is deliberately **no** `ProbeBatch`. An earlier draft had the Profiler consume the framework's
`Batch` carrying probe tags; with spec D15 there is no dataset on the way in, so there is nothing to
subclass. The framework's shapes appear only at the output boundary, below.

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
| `descriptor` | `str` | `min_length=1` | The natural-language description of the probe pattern — the `z` of `ω(π_j, z)` as it crosses to the Exploiter. **Not the strategy identifier**: invariant 3 forbids sending identifiers, so aggregation is keyed internally by the strategy `id` and only this prose lands on the profile. |
| `rate` | `float` | `ge=0.0, le=1.0` | The estimate of `ω(π_j, z)`: violations over trials for this pair. Computed in `searches/scoring.py`, not through the framework's statistical modes — see the plan for why. |
| `n` | `int` | `ge=0` | Sample size behind the rate (FR-012). Travels with it so a descriptor resting on a handful of probes cannot be read as settled. |
| `standard_error` | `float` | `ge=0.0` | The binomial standard error of `rate` — the same quantity `S(c)` subtracts, computed the same way, so one measurement has one statistical treatment. |
| `source_strategy` | `str \| None` | default `None` | The strategy `id` this entry was aggregated from, for auditing only. **Stripped before the profile crosses** (FR-013). Nothing downstream may branch on it. |

## `AssistantProfile`

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `weaknesses` | `list[WeaknessEntry]` | default `[]` | The map `ω`. |
| `hooks` | `list[KnowledgeHook]` | default `[]` | The retained hooks `H`, each with its `doc` label. |

FR-013 makes this the only artifact passed to the Exploiter, and invariant 3 makes it carry prose
rather than internal identifiers.

## `ProfilerResult`

What the Profiler returns. A plain result object, not a metric result: nothing here inherits the
framework's metric base, because Roast Me does not emit through the metric pipeline (spec D15).

| Field | Type | Notes |
|---|---|---|
| `profile` | `AssistantProfile` | The profile `θ`, and the only part the Exploiter receives (FR-013). |
| `outcomes` | `list[GradedOutcome]` | Every outcome, controls and ungraded included: excluded from the rates, kept in the record (FR-011, FR-016). |
| `overall_rate` | `float` | `ge=0.0, le=1.0`. The violation rate over scoreable, graded outcomes. |
| `n_scoreable` | `int` | How many outcomes entered the rates. |
| `n_ungraded` | `int` | How many were dropped for a failed exchange, so a silent shrink is visible rather than a denominator quietly shifting. |

## `Category` and `CategoryEvaluation`

| `Category` field | Type | Notes |
|---|---|---|
| `attributes` | `list[str]` | `min_length=1`. The ordered conjunction `c = (a₁…a_ℓ)`. |
| `provenance` | `list[str]` | Per attribute, the weakness entry or hook that induced it (FR-028). Model validator: same length as `attributes`, so an attribute can never lack provenance. |

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
| `rationale` | `list[list[PrincipleGrade]]` | Per query, the per-principle grades behind its violation. FR-036 requires a category evaluation be auditable, and a score with no grades behind it is not. Same length as `queries`. |
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

### The output boundary

`RoastDatasetRecord` is the last shape Roast Me owns. To satisfy FR-034 the records are converted into
the framework's dataset shapes so existing metrics consume them unmodified. Those models require fields
Roast Me has no natural value for, so the conversion states what each one gets rather than leaving it to
whoever writes the code:

**`RoastBatch(Batch)`** — the output-side subclass, following the `PrivacyBatch` precedent. The framework's
`Batch` has no free-form slot, so the record's own fields need somewhere to live that existing metrics can
ignore.

| Field | Filled with |
|---|---|
| `query` | the probe's question |
| `assistant` | the assistant's answer |
| `qa_id` | the probe identifier |
| `ground_truth_assistant` | `""` — a trap has no correct answer, and inventing one would let a metric score against it. This is why SC-006 names a metric that reads `assistant` alone |
| `weight` | left unset. It is the framework's own aggregation weight and has nothing to do with principle weights |
| `roast` (added) | the `RoastDatasetRecord`: violation score, principles charged, rationale, evidence and `evidence_available`. `evidence_available` has to survive, or absent evidence and evidence-sought-and-not-found collapse into the same thing — the distinction US2 requires be preserved |

**`Dataset`** — one per run, not one per probe.

| Field | Filled with |
|---|---|
| `session_id` | the run identifier |
| `assistant_id` | the target identifier the user supplied |
| `language` | the knowledge base's language, which the probes are written in. It defaults to `"english"` and is propagated into session metadata, so leaving it unset would label a Spanish corpus as English |
| `context` | a description of the run. **Not the knowledge hook**: `context` is one required string per session, and a hook is a model, one per probe. The hooks travel on each `RoastBatch.roast` instead |
| `chatbot_role` | left unset |
| `conversation` | the `RoastBatch` list |

Records whose `violation` is `None` are still emitted: an ungraded exchange is part of the audit trail.
What they must not do is enter any rate, which is why the profile carries `n_ungraded` explicitly.

This conversion is the **only** place the framework's `Batch` and `Dataset` appear, and it belongs to
`generators/roastme/dataset.py` rather than to any component that computes something.
