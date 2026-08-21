# Amendment — 004-roastme-metric, after a field run

The subsystem was merged with every task closed and `plan.md` reading *"Nothing in this document is
open."* It was then run end to end against a production RAG assistant, and that run found defects no
test could have: the tests do not make five hundred calls to a shared router, and they do not run
over a corpus of Spanish product names.

This amendment records what changed and what a reviewer has to decide. It is written against the
evidence in `amendment-draft.md`, which carries the measured case behind every entry.

**What did not change is as important as what did.** Four behaviours were reported as defects and
turned out to be decisions this specification had already made, with the consequence in view. They
are listed in §5 so the same reports do not come back.

---

## 1. Amended requirements

**FR-016** — *was:* an exchange the target marks as failed MUST be recorded as ungraded and MUST
count as neither a violation nor a pass.

> **FR-016**: An exchange that produces no complete set of grades MUST be recorded as ungraded and
> MUST count as neither a violation nor a pass — whether the **target** failed the exchange or the
> **grader** failed to rule on it. The outcome MUST carry which of the two occurred and why.
> Partial grades MUST be discarded rather than kept: `v` is a weighted sum over every principle of
> the contract, so a subset cannot produce one.

*Why.* The requirement covered the assistant and not the judge, and nothing in the subsystem caught
anything. One unparseable verdict ended a run and took with it every assistant call already paid
for — the half that cannot be reproduced, since a lost response is not recovered but replaced.

**FR-025** — the entity-kind clause gains a second obligation.

> …and every `entity_kind` is one the configured engines declare they handle. **An engine that
> declares its entity kinds MUST generate probes only for those kinds.** An engine that declares
> none is unrestricted, which is the default and what a user who named no kind asked for.

*Why.* The declaration was read by catalogue validation and by nothing at generation. Of the four
shipped engines only the enumeration one passes `kind` through: graph and multi-hop ignore the
argument and return every node and every chain, so on a catalogue with two kinds they hand **the
same boundary to both** and a strategy over values builds premises out of product names.

**FR-034** — extended to the search half.

> …and **both halves of a run MUST reach it**: the probes and outcomes the Profiler produces, and
> the categories the Exploiter evaluated. The conversion for the search half MUST require the
> language rather than defaulting it.

*Why.* `to_dataset` takes probes and outcomes, which a search has neither of. The half that produces
what the paper calls the Roast Dataset reached no metric at all, and its exchanges lived only inside
the report.

**FR-037** — the boundary is unchanged; its application is made complete.

> …and every symbol that pulls no dependency of the extra MUST be reachable from the subsystem
> facade.

*Why.* The rule was applied to three names. `to_dataset`, the single bridge to the metric pipeline,
was reachable only by naming a module while `Profiler` sat on the facade: a user could start a run
through the front door and not finish one without reading the source.

**FR-039** — the reference query generator gains two parameters.

> …and the shipped query generator MUST accept the **domain** and the **language** of the assistant
> under evaluation. Both are parameters of the run and neither reaches the profile, so invariant 3
> is untouched.

*Why.* A profile carries no identifiers by design, so the prose of the attributes was the only thing
telling the model what the assistant was for. Against a Dominican bank the shipped prompt wrote
*"What is the price of the new Airpods Xpro?"*. The assistant answered correctly, `v` came out zero,
and the report said nothing was found **having never laid a trap**.

**D13** — degrading to sampling stays the default, and gains two conditions.

> A grader MUST distinguish a provider that exposes no logprobs from a call that failed, and MUST
> degrade only for the first. The distinction MUST be established rather than inferred from an
> exception's type or text, which belong to whichever client the user supplied. A grade obtained by
> the fallback MUST carry why the primary path was abandoned. A user MUST be able to require the
> primary path and have the run fail rather than change estimator.

*Why.* A bare `except Exception` around the network call relabelled rate limits and timeouts as
"this model exposes no usable logprobs", then spent `k` more calls on the model that had just
failed. The two paths are two estimators — one reads a continuous probability from the verdict
token's distribution, the other votes across `k` samples and can land only on multiples of `1/k` —
and a rate averaged over both is a mean over two different measurements.

---

## 2. New requirements

> **FR-042**: A query generator MUST be able to declare the grading context a query it produced
> carries, and the search MUST pass that context to the grader as the Profiler passes a probe's.
> Declaring none MUST be the default.

*Why.* The Profiler graded against `Probe.meta` and the search graded against nothing, so every
context-dependent rubric reached its exit clause and returned compliance for every query the
Exploiter ever sent. The consequence is arithmetic and nothing raises: the highest violation the
search can record becomes the sum of the weights of the remaining principles — 0.30 of 1.0 for the
contract this was measured on — so a `τ` above that is unreachable by construction and the empty
report reads as a well-behaved assistant.

*Why the generator and not the grader.* A grader recognising premises on its own needs either entity
recognition over free prose, wrong in both directions and each direction either fabricating or
losing a violation, or a second copy of the generator's vocabulary — the same fact written twice in
two places that can drift apart with nothing to notice.

> **FR-043**: A dataset record MUST carry what produced it: for a probe, its identifier, strategy,
> engine and knowledge hook; for a query a search invented, the category that proposed it.

*Why.* A record carried a query and a score and nothing saying where either came from, and
`queries_over_threshold` flattens every category into one list. It is also what gives
`KnowledgeHook.absence_reliable` and `verified` a reader: both were written on every probe and
neither survived to where a rate is read, so an unreliable absence label was indistinguishable from
a confirmed one — the distinction FR-023 exists to preserve, held at the write and lost at the read.

> **FR-044**: A profiling result MUST be self-sufficient: the probes a run sent MUST travel with
> their outcomes, so a serialised result can be converted without a second artifact.

> **FR-045**: A run's result MUST report how many grades each judging method produced.

*Why.* The method was recorded per grade and aggregated nowhere, so a run that degraded throughout
looked identical to one that never did. Counted rather than flagged: the two readings it separates
are "the provider never exposed logprobs, so the whole run is the other estimator" and "eleven
grades of two hundred degraded", and a boolean collapses them.

> **FR-046**: A search MUST refuse to run when its own parameters admit nothing from the profile it
> was given, naming what the profile carries.

*Why.* An `η` above every rate produced a well-formed report with empty categories and **zero calls
to the assistant**. It is the cheapest wrong answer the subsystem can produce: no cost, no waiting,
and indistinguishable from a clean run. FR-035 rules out that reading; the rates are known before a
query is generated, so it is decidable rather than discovered.

*Scope.* The check belongs to whichever search owns the parameter. `η` is the attribute-iteration
search's; the policy-gradient search never reads it, and a user's own search is free to ground
itself another way. **No equivalent check exists for `τ`**, and none can: its ceiling depends on
which rubrics can be charged without context, and that lives in the prose the user writes.

---

## 3. Data model

| Model | Change |
|---|---|
| `RecordProvenance` | **New.** Optional throughout, because the two halves know different things — a probe record names its probe, a search record names its category, and that shape is the fact rather than an accident of which columns are null |
| `RoastDatasetRecord.provenance` | New, defaulting to empty |
| `GradedOutcome.ungraded_reason` | New. `violation is None` said a number was missing, not whether the assistant never answered or the judge never ruled — and those call for opposite responses. It also gives `TargetResponse.failure_reason` its first reader |
| `ProfilerResult.probes` | New |
| `ProfilerResult.grading_methods`, `FailureReport.grading_methods` | New |
| `GraderConfig.require_logprobs` | New, default `False` |
| `QueryGenerator.meta_for` | New method with a default returning `None` |
| `PromptedQueryGenerator(domain, language)` | New, both optional |
| `ProbeLibrary(verifier)` | New, optional. Records each hook's verdict and **never removes a probe**: shrinking a probe set silently is how a run reports a smaller denominator as though it had measured the whole thing |
| `StrategySpec.phrasing_hint` | May now carry a `{premise}` slot. The colon form remains the default, since a catalogue is data the user already wrote |

---

## 4. Corrections to this specification

* **`plan.md` states "Nothing in this document is open."** It was not true when written: PR #18's own
  description declares three open findings — the weakness-map key, the realism estimator, and `C*`
  unmarked — and none of them reached the docs, an issue, or this spec. Two are resolved in §5, one
  in §1; the sentence must go regardless.
* **`plan.md:189` and `:406` describe the shipped realism estimator as "the paper's own
  construction: expected cosine distance from a prior pool".** The implementation took each query's
  distance to its *nearest* pool member and argued for it in its own docstring. Since a maximum is
  never below a mean, its gate was never stricter than the paper's and usually looser. **The code
  now matches the plan.** Carried with it: the recommended `δ` is derived from the pool's own mean
  self-distance rather than fixed at `0.5`, which was calibrated for the smaller nearest-neighbour
  scale and, left unchanged, would have rejected almost every category.
* **`data-model.md:277` contradicts itself on `WeaknessEntry.source_strategy`**, which is to be
  populated *"for auditing only"* and *"stripped before the profile crosses"*. The field lives on
  `WeaknessEntry`, which lives on `AssistantProfile`, which **is** what crosses — so no artifact
  could hold the unstripped value. It is set nowhere. The spec has to say which it wants.
* **The docs' interface table** counted six reference implementations and labelled the realism
  estimator "the paper's construction" while the module admitted it was not. `HookVerifier` now
  ships one; both entries are corrected.

---

## 5. Reported as defects, and decided here already

Recorded so the same reports do not come back. Each was raised from the field run and each is
answered by this specification's own text.

| Report | Answer |
|---|---|
| The weakness map groups by strategy, where the paper's `Z` is (template, topic, hook type) | `spec.md:118` and `data-model.md:154` both define `StrategySpec.id` as *"the aggregation descriptor `z` of the weakness map"*. **Decided.** The cost is real and now stated in the docs: the Exploiter **misses** weaknesses rather than misreporting them |
| The map carries one entry per (strategy × principle), most of them zero | `eq:weakness` is `ω(π_j, z)`, indexed by principle **and** descriptor. Entries scoring zero are the definition |
| `_candidate` ranks by raw per-query violation rather than by `S(c)` | `data-model.md:105` defines `pool_size` as *"how many of the highest-scoring query/response pairs the training-free search keeps"*. **Decided.** Ranking by `S(c)` would be a better criterion and is not the approved one |
| `to_dataset` defaults `language` to `"english"` | `data-model.md:366` decided it *with the hazard written down*: *"leaving it unset would label a Spanish corpus as English"* |
| `StrategySpec.doc` is read by nothing | `data-model.md:160` calls it the *expected* label; invariant 4 has the real one derived by the engine. Where they disagree it is the transformation that did not do what the catalogue claims — now documented on the field. Cross-checking them is a proposal, not done |

---

## 6. Open

Replaces *"Nothing in this document is open."*

1. **`C*` is not marked on the report.** Reconstructable from each `score` and the `τ` now recorded
   in `components`, so this is reporting ergonomics rather than a wrong number.
2. **The default engine set is not the one the paper evaluated.** Retrieval, graph and multi-hop run
   by default; the canonical dataset came from retrieval, graph and enumeration. **The
   out-of-the-box configuration produced no published number.** Documented, not resolved.
3. **D9 remains open**: the paper's *Implementation Considerations* section is still commented out,
   so the traceability chain this spec deferred has not closed.
4. **`source_strategy`** — see §4.
5. **A first-class noise check for the mention extractor.** `MentionExtractor` post-dates this
   specification, which names neither it nor `CompoundTokenExtractor`. Over a Spanish product corpus
   the default returns 208 mentions that are not entities, and each becomes a probe with nothing
   raised. Documented on the class; a check is proposed, not built.
6. **No grader is calibrated against human labels.** Unchanged, and restated because everything
   above measures how a number is produced and none of it makes the number true.
