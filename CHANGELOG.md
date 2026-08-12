# CHANGELOG


## v1.1.0-b.6 (2026-08-12)

### Bug Fixes

- **judge**: Re-ask when the answer holds no yes/no token
  ([`b07f546`](https://github.com/gaussia-labs/pygaussia/commit/b07f546d335dae8ec4fe6124348365f3dae1516c))

A judge sampling at temperature 1.0 occasionally answers off-format. Observed live: asked a yes/no
  question with an explicit one-token instruction, a judge

began enumerating the criteria — ':', ' extra', ' explanation', ' apologies', ' politeness' — so no
  yes/no token was emitted at all and check_logprob_binary raised. One such draw ends a whole run,
  and the run in question was 54 judgements over 9 items.

Re-asking is a fresh draw, so it addresses the cause directly rather than hiding it: nothing is
  degraded, no fallback is taken, and a genuinely bad prompt still fails after the attempts are
  spent. Two retries by default, which on the observed rate of roughly one non-compliant answer in
  380 takes a 54-call run from about a 13% chance of dying to about one in a million. The happy path
  spends no extra call. extraction_retries=0 restores the previous behaviour.

The retry is conditional on the model having emitted tokens. An answer carrying no logprobs at all
  is a capability limit, and re-asking it would burn every attempt on every judgement of a whole run
  before failing with the same error. A test asserts the invocation count in both directions rather
  than only the exception.

The invoke and its metadata handling move into _request_logprobs so the retry loop stays short. The
  failure message now reports how many attempts were spent, so one bad draw is not mistaken for a
  broken prompt.


## v1.1.0-b.5 (2026-08-12)

### Bug Fixes

- **judge**: Score the token where the model commits, not position 0
  ([`6923af3`](https://github.com/gaussia-labs/pygaussia/commit/6923af396e124c35ef19017d4cb263409229a08f))

check_logprob_binary read the distribution at generated position 0 only, which assumes the judge's
  first token is its answer. Measured against an OpenAI-compatible router: with a short prompt,
  models reliably answered with a bare token; with a long grading rubric of around 900 tokens, the
  same model on the same prompt emitted the bare token on one call and a short preamble such as
  "Answer", ":", " " before it on the next, with nothing changed in between. Every call that
  produced a preamble raised LogprobsExtractionError, even though the answer token and a well-formed
  distribution over the candidates sat two or three positions later.

Compliance with "answer with one token" appears to degrade as a prompt grows, which makes position 0
  least reliable for exactly the long rubric prompts an LLM judge tends to need.

The scan now walks forward to the first position whose *sampled* token is a candidate. The quantity
  measured is unchanged — still a single-position distribution over the candidate tokens — so
  calibration is unaffected. Selection is on the sampled token rather than on a candidate appearing
  anywhere in the distribution, because a bare "Yes" sits in the top-N of almost any position of
  prose and the looser test would score off a word of preamble. The window is bounded, so a
  reasoning model, whose visible output is its trace, still fails loudly instead of being searched
  for a stray token. scan_tokens=1 restores the previous behaviour exactly.

Also fixes a crash in the same statement: a model that accepts the logprobs parameter and ignores it
  can answer with "logprobs": null, so the key is present and None. A two-argument get returns None
  rather than its default there, and chaining off it raised AttributeError instead of the
  LogprobsExtractionError the method intends.

Refs #23


## v1.1.0-b.4 (2026-08-11)

### Bug Fixes

- **roastme**: Let a user say how their corpus is read and how a premise is built
  ([`b87af65`](https://github.com/gaussia-labs/pygaussia/commit/b87af6551258a171ff426393e2022fce445e99f3))

Pointing Roast Me at a real knowledge base — twelve markdown pages scraped from a bank's site,
  Spanish prose — produced no usable probes, and the reason was design rather than a defect.

Three of the four engines found entities through one regex over compound identifiers. It reads
  POLICY-1 and Articulo_25, which is the shape of the paper's own corpus, and on prose it read the
  wrong things: 210 false positives across twelve documents — phone numbers, PDF filenames, footer
  anchors. No error, because a junk boundary is indistinguishable from a legitimate one at that
  layer. The four transforms assumed the same entity shape: a `-2` suffix reads as a near miss of
  POLICY-1 and as a typo of "Cuenta Digital Libre", so the probe stops discriminating — a refusal
  could mean the assistant caught the invention or that it saw broken text.

Both steps are now injectable, which is what `EnumerationProbeEngine` already did with its
  enumerator:

- the three corpus-reading engines take a `MentionExtractor`, defaulting to today's reading, so
  nothing that worked before changes. It lives beside the engines rather than in `core/` — a
  collaborator of three shipped engines, not part of the specification a user implements against,
  the same call already made for CategoryPolicy. The ten interfaces stay ten; - a catalogue may name
  any `Transform` the caller supplies. `Transform` was already listed among the ten interfaces and
  validation refused every implementation of it, so the interface was documented and unusable. The
  four shipped cannot be replaced and a colliding key is refused rather than preferred, since either
  resolution silently changes what an existing catalogue means. Validation and generation must be
  given the same sequence, and the docstrings say why.

`flip_value` and `flip_fact` are left alone: the first mangles figures written with thousands
  separators and returns an entity with no digits unchanged, which the engine then labels
  documented, quietly turning that strategy into a second control; the second prepends an English
  `not`. Neither is wrong for the corpus they were written against, and supplying one that fits is
  now the answer.

The tests that should have existed: a corpus of ordinary words asserting the default extractor finds
  nothing there and that the engines therefore produce no probes rather than probes over junk, that
  an injected extractor makes the same engines produce, that a supplied transform passes validation
  and builds its premise, and that a colliding key is refused. The 911 fixtures were all
  identifier-shaped, so nothing could have caught this.

920 tests, mypy clean on 145 files, ruff clean. Verified end to end against the bank corpus: probe
  premises went from "Cuenta Digital Libre-2" to "Cuenta Digital Impulsa", generated by the Probe
  Library with no custom engine.

### Chores

- **skills**: Drop the aws-lambda skill
  ([`b18928e`](https://github.com/gaussia-labs/pygaussia/commit/b18928ecc8381a0e6bef341557fe5680e67f8204))

No new Lambda examples are planned — the roastme one was removed on review — so a skill that
  generates them is a path nobody should take. Deprecating it in place would be worse than removing
  it: a skill left in the directory is still listed and still invoked.

The five existing examples under examples/*/aws-lambda/ are untouched. They work; they are just not
  a pattern being extended.

### Documentation

- **roastme**: Add the four Roast Me skills
  ([`b527569`](https://github.com/gaussia-labs/pygaussia/commit/b527569fce26e755a19522894d064c4c9e773e56))

One per stage a user has to author: the catalogue's plugins, its strategies, the Profiler's contract
  and target adapter, and the Exploiter's collaborators and thresholds. Each states the rules the
  library enforces, so a mistake surfaces at validation instead of halfway through a run that costs
  target calls.

Written for someone pointing Roast Me at their own assistant, so each carries what only shows up on
  a real run: that the adapter must report a transport failure by returning `failed` rather than
  raising, that the grader's sampling fallback binds its own temperature, what the call budget of a
  search actually is, and that an agent whose tools write needs a sandbox because the probes
  exercise them for real. Both run skills end with the failures to work through when a result comes
  back empty.

They sit here rather than in a marketplace because their content is tied to the schema they describe
  — which fields PluginSpec carries, which four transforms exist, what validate_catalogue checks.
  Next to that code they move with it.

Every python block parses, every symbol named resolves, and the verification snippets were run: the
  plugin-to-principle mapping, the attribute clauses, the control check, and the three
  validate_catalogue messages quoted are the ones the library produces.

- **roastme**: Tell readers the skills exist and how to get them
  ([`d583472`](https://github.com/gaussia-labs/pygaussia/commit/d5834727cf0825f0e31aa3a0de4523bba57eaf60))

The four skills were reachable only from skills/README.md, which you find by browsing the folder you
  needed the instructions to find. Nothing in the published docs mentioned them at all.

The Roast Me guide now carries the four names, the two commands that put them in .claude/skills/,
  and a line saying to skip the section if you are not using Claude Code — everything they say is
  already on that page and in the notebooks.

The instructions clone the repo because the skills are not in the pip package. That is the cost of
  keeping them next to the schema they describe; publishing them to a marketplace would replace both
  commands with one.

- **roastme**: Warn the strategies skill about the corpus the engines expect
  ([`2d40bfd`](https://github.com/gaussia-labs/pygaussia/commit/2d40bfd72d4e90a09486bcafbf8df0f19d712d18))

Three of the four shipped engines find entities with one regex over hyphen/underscore compound
  tokens. On a corpus of web-scraped prose that returns junk rather than nothing — 210 phone
  numbers, PDF filenames and footer anchors across twelve documents of a real bank site — and the
  engine then generates `len(junk) × len(strategies)` probes with no error. The skill now opens with
  the snippet that checks your own corpus before you trust it, and points at EnumerationProbeEngine
  as the escape, including that its contract is completeness and that `structured=True` is what
  makes the engine see a document at all.

The transforms table now carries what each one actually does to a premise, because two of the four
  have narrow ranges and neither says so when it misses: flip_value increments every digit run, so
  RD$500,000 becomes RD$501,1 and an entity with no digits comes back unchanged and silently turns
  the strategy into a second control; flip_fact prepends a hardcoded English "not". The registry is
  read-only, so the guidance is to leave the principle to the transforms that do fit — the Profiler
  grades every principle on every response regardless.

Found by pointing the SDK at a Spanish bank corpus. Every claim checked against the source.


## v1.1.0-b.3 (2026-08-11)

### Bug Fixes

- **roastme**: Corrections found reviewing the arithmetic against the paper
  ([`a2556c1`](https://github.com/gaussia-labs/pygaussia/commit/a2556c11b3fcebfa517113510d93648437e6587d))

A pass that recomputed every fixture from the paper's equations rather than from the code. All 49
  agree. What it did find was three claims that did not.

The data model said the weakness map carries "the binomial standard error". That is the estimator
  for a proportion of Bernoulli trials, and the quantity is a mean of violation scores — continuous
  three ways over: the shipped grader returns a logistic probability, the sampling fallback returns
  a vote fraction, and a weighted sum over several principles is fractional even from binary grades.
  The two forms coincide only on binary data, which is why the weakness map's own fixtures read
  either way and why the wrong name spread. The paper settles it in its own words: "the empirical
  mean of v ... its standard error ... the penalty rewards categories that fail consistently". A
  penalty that cannot see dispersion cannot do what the paper says the penalty is for. The code was
  already right and two tests already held the line, so only the sentence changed; the test that
  pins the coincidence on binary values now says that is what it is.

The same line described the rate as "violations over trials". It is the mean of the per-principle
  grades. On [0.4, 0.6] the mean is 0.5 where a count is 1/2, 2/2 or 0/2 depending on where the line
  goes, and a proportion's standard error at that rate and n is five times the correct value. Three
  fixtures now pin the general form, all at one rate and one n so that sqrt(rate(1-rate)/n) is
  constant across them and only the mean's form tracks the dispersion.

The realism estimator claimed to be the paper's construction. The quantity is the paper's; the
  estimator is not. The paper takes the expectation over the whole pool and offers a cheaper
  centroid variant, where this takes each query's distance to its nearest pool member. The docstring
  argues that reading and it is defensible, but a maximum is never below a mean, so this gate is
  never stricter than the paper's and over a diverse pool the gap is enough to flip it. FR-039
  exists to attribute constructions correctly, so the module now says which half is the paper's and
  what the consequence of the other half is.

Two search knobs hardened while there. The query generator's re-ask carried no memory of what it had
  already collected, so the likeliest reply to "write two more" was the two just written; the
  duplicates were then discarded and the attempt spent making no progress, which could burn the
  whole budget and fail a run that would have succeeded. And max_attributes now refuses a value
  below one rather than degrading to seeding without conjoining, which is a different search rather
  than a narrower one. Its docstring also claimed the cap is what keeps a run finite; it bounds
  refinement only, and the cost model is now stated in full.

- **roastme**: Defects found reviewing the code
  ([`84947c3`](https://github.com/gaussia-labs/pygaussia/commit/84947c35a0d22702405741b7ca64bfce9d26b236))

Reviewers that had not written the subsystem went over it against the requirements and the
  constitution. What they found, and the coverage that pins each one.

A contract whose weights sum to 1 + 5e-10 — accepted on purpose, so a contract assembled from
  decimals is not rejected for float noise — produced a violation score above 1.0 and crashed the
  run at the one moment the assistant violated every principle. The tolerance exists so contracts
  are accepted, not so scores may leave [0,1], so the score is clamped where it is produced and the
  two means that consume it inherit the bound.

At tau = 0.0 the failure report surfaced queries the kappa gate had zeroed, which were never sent:
  no response, no grades, no rationale. That contradicted the file's own docstring and FR-036, which
  asks for auditable records. Having been asked is now a second condition rather than something the
  score implies.

FR-005 asks every grade to record the grader, and no field held it. Neither method nor model
  identifies one — a grader has two methods, and a rule-based grader has no model — so grader is
  required rather than defaulted: nothing legitimately produces a grade anonymously.

The stripped-identifier test asserted that a field nothing ever writes is unwritten, so it passed by
  construction and would have survived the stripping breaking entirely. It now scans the serialised
  entry. The retained hooks keep their kind and transform key, since a hook stripped of them is no
  longer provenance an evaluator can act on; what invariant 3 forbids is the Exploiter steering on
  that vocabulary, so it is enforced at the consumer instead.

Delete gated_violation rather than wire it into the pipeline. The pipeline gates before the target
  call, precisely so it does not spend one on an answer that cannot count, so a gated query never
  has a violation to pass in and sharing the function would have meant inventing one. The shared
  constant is shared instead.

Fix a sigmoid that raised OverflowError on a separation no exponent can carry, reachable when a
  provider reports a sentinel logprob. Raise on a principle graded twice rather than silently
  keeping the last, which made v depend on the order two disagreeing grades arrived in. Correct
  three fixture literals that disagreed with their own stated derivations, and the docstring that
  explained the disagreement with a reason that was not true.

Branch coverage of scoring.py to 100% and of the default exploiter path from nothing: the guards are
  the point of these modules, so a guard with no test is a guard that has never run.

- **roastme**: Type the policy update's config structurally, not against trl
  ([`500542f`](https://github.com/gaussia-labs/pygaussia/commit/500542f603d26f323b9dcd73e57f886cc1ace50d))

trl 1.9.2 moved PPO to `trl.experimental.ppo`, so `from trl import PPOConfig` stopped resolving and
  mypy failed on develop. No single import path is correct across the declared `trl>=0.8.0`: on 0.x
  it is `trl.PPOConfig`, on 1.x the experimental one — which trl itself warns "may change or be
  removed without notice".

The module never called trl. The clipped surrogate is computed here and the import existed only to
  name an annotation, so the annotation is now a Protocol over the two fields actually read,
  `cliprange` and `learning_rate`. trl's PPOConfig still satisfies it — checked with mypy against
  the installed 1.9.2 — and stays what the docstring says to pass. The `trl.*` mypy override goes
  with it, unused and reported as such.

Verified under `uv sync --all-extras`, which is what CI does and a default install does not: mypy
  clean on 144 files, ruff clean, 911 tests pass. The failure was invisible locally because trl is
  absent without the extra and `ignore_missing_imports` covers it.

### Build System

- **roastme**: Add the roastme extra
  ([`005086d`](https://github.com/gaussia-labs/pygaussia/commit/005086d536fad1ac9cd363ce25ae0851f04c5f27))

sentence-transformers, torch and networkx, excluded from both the metrics and the all extras so the
  base install stays light and importing gaussia.core keeps working with none of them present.

Relocking also carries a version correction inherited from develop: uv.lock still recorded gaussia
  1.0.0b3 while pyproject.toml declares 1.1.0-b.2. This commit is kept on its own so that correction
  is visible rather than buried in a feature diff.

- **roastme**: Add the roastme-rl extra
  ([`5f956eb`](https://github.com/gaussia-labs/pygaussia/commit/5f956eb30a7338d28184d4851b0d40a4c53d07c9))

peft, accelerate and trl on top of gaussia[roastme], kept out of every aggregate: the
  training-backed update step is the only module that imports them, and the policy-gradient loop
  must stay runnable on CPU with none of them installed. Their mypy overrides go alongside, since
  none ships py.typed.

### Chores

- **skills**: Add /docs and /aws-lambda ([#19](https://github.com/gaussia-labs/pygaussia/pull/19),
  [`e45e42f`](https://github.com/gaussia-labs/pygaussia/commit/e45e42fa74370a91802cbbdbb0ae297f960d6e2a))

Written against the patterns already in the tree rather than against one metric, so they apply to
  any metric or module.

/docs picks the directory from whether the subject emits a BaseMetric, picks the tier from the
  section markers the existing pages share, and requires registering the page in docs.json and
  docs-sync.json — which have different JSON shapes and different path forms. That is the step that
  gets missed: metrics/role-adherence is registered in neither and does not publish, and
  metrics/privacy in only one.

/aws-lambda separates the six mechanical files from run.py and README.md, and gates on the extra
  before anything is written: every existing example targets an extra that is empty or absent, so a
  module pulling torch or sentence-transformers needs a decision about the 10 GB image ceiling
  first.

### Documentation

- **roastme**: Guide, catalogue examples and notebook
  ([`fee6148`](https://github.com/gaussia-labs/pygaussia/commit/fee614894444082a672784e5cefc0db32832dd1e))

The page sits under advanced/ rather than metrics/: Roast Me generates a dataset for the metrics to
  read, it is not one of them. Registered in both nav registries, since docs-sync.json is the one
  that publishes.

It states the four things the requirements make non-optional: no grader here is calibrated against
  human labels; the query generator and on-profile filter are gaussia's construction, so
  substituting them changes what the search measures; where kappa and delta come from when the user
  supplies neither; and that the training-free search has no published result behind it. The worked
  ExploiterConfig carries tau and eta, which is what makes the two required thresholds copyable
  rather than guessable.

Every code block on the page and every notebook cell was executed, not just written, so the numbers
  quoted are real output.

- **roastme**: Plan gate — implementation plan and data model
  ([`d5a2115`](https://github.com/gaussia-labs/pygaussia/commit/d5a21156c9f2991e4089be43d61c8efdfb7268e9))

Maps the approved spec onto the SDK: seven abstractions in core/, the Pydantic shapes in
  schemas/roastme.py, RoastMeProfiler as the metric, and the Probe Library and Exploiter as
  generators whose product the pipeline consumes. All 38 functional requirements have a file, all 11
  success criteria have a test, and the four constitution gates are filled with reasons rather than
  ticks.

NEEDS DECISION — ALEX: which category searches ship. The interface is settled by the approved spec;
  what is open is which implementations land now. Three options with their dependency, CI-coverage
  and default-path consequences are laid out in the section right after the summary. The
  recommendation puts the training stack behind its own extra, which contradicts spec D5 and would
  need that decision amended. Everything below that section is written for the recommendation and
  degrades cleanly under the others.

Notable design points: the Profiler takes no target assistant at all, so grading without contacting
  the assistant is structural rather than a mode; the weakness map's rate goes through the injected
  StatisticalMode, so a descriptor resting on few probes returns a credible interval instead of a
  zero that reads as settled; and the catalogue's transform string is resolved by a registry once,
  at validation, so nothing branches on it afterwards.

llm/judge.py is deliberately untouched.

- **roastme**: Quickstart notebook, and the guide trimmed to reference
  ([`0fd6797`](https://github.com/gaussia-labs/pygaussia/commit/0fd6797109dd4c8c452edc7b2a8b4bec390353ad))

A quickstart beside the full walkthrough: the whole arc — contract, catalogue, probes, profile,
  exploit, Roast Dataset — offline in a few seconds behind crude stand-ins, at half the size of the
  long one. Its probes are derived from a two-strategy catalogue through the library's own
  `principle_by_plugin` and `strategy_attributes`, so `plugin` holds a plugin id rather than a
  principle and every field is shaped the way the engines shape it. Both helpers import with no
  extra installed.

The guide drops from 905 lines to 721. It had grown a full runnable walkthrough because there was no
  quickstart when it was written; the 73-line block of stand-ins that existed only to make the page
  executable is gone, and the snippets name the shipped components instead — LogprobGrader with its
  GraderConfig, PromptedQueryGenerator, JudgeOnProfileFilter, SentenceTransformerEmbedder — so what
  a reader copies is what they would deploy.

Three corrections came out of re-running the guide's blocks: the policy-gradient snippet had lost
  its AssistantProfile and Category imports along with the block that used to provide them; the
  realism figures were a hashing stand-in's, and measured again with the default all-MiniLM-L6-v2
  they are 0.135 against 0.502, so the second clears the recommended delta of 0.5 by two
  thousandths; and the multi-hop chain in the prose omitted the mutation on its last hop.

Both notebooks store the output of a real run, which is how the generators notebooks read. Ten of
  the guide's eleven python blocks were executed to verify it; the eleventh constructs the Exploiter
  with the real judge-backed collaborators and needs a provider.

- **roastme**: Rework the plan and data model after review
  ([`3dfab32`](https://github.com/gaussia-labs/pygaussia/commit/3dfab32a086f32c85b56f23d4ab95b5387d97d90))

Drops the framework's statistical modes, reversing an earlier position: the category score needs a
  standard error and the framework's dispersion utility returns a mean absolute deviation. Using one
  utility for the weakness map and hand arithmetic for the score would put two statistical
  treatments inside one measurement.

Makes the output boundary buildable. The previous table named framework fields in prose, which hid
  that two of its targets do not exist: there is no metadata slot on a turn, and the session context
  is one required string per session rather than a per-probe hook. The conversion now goes through a
  Batch subclass on the output side and states what every required field is filled with, including
  the language that would otherwise label a Spanish corpus as English.

Adds the configuration the design assumed and never defined — every threshold the method takes as a
  parameter, required rather than defaulted, since a default would be the library deciding how hard
  a category has to fail before it counts.

Probe engines now declare which entity kinds they handle, so catalogue validation can reject a kind
  nothing can produce instead of generating an empty probe set.

Which of three pluggable pieces ship a working implementation is left open for the reviewer. An
  earlier draft resolved it with a rule that contradicted itself: the query generator was to ship
  because the Exploiter cannot run without one, while the filter was not, on the ground that the
  Exploiter should refuse to run without one. The paper treats both the same way, naming them
  without constructing them.

The policy-gradient search shipping without automated coverage is now an unchecked box in a Testing
  Gate rather than a footnote under a claim of no violations.

- **roastme**: Tasks gate — TDD task breakdown
  ([`bded411`](https://github.com/gaussia-labs/pygaussia/commit/bded411e042575243e0846f2d60d6d1c5bee72dd))

Fifty-six tasks in nine phases, every path taken from the plan's file tables and every task tagged
  with the requirement or success criterion it exists for. Tests are written and verified failing
  before the code that satisfies them.

The reinforcement-learning search lands last, in its own phase behind its own extra, so every
  earlier checkpoint stays verifiable without a GPU. Documentation and the runnable example follow
  the implementation rather than preceding it.

This is here for continuity; the gate under review is the plan.

### Features

- **roastme**: Exploiter and threshold resolution
  ([`c2cd608`](https://github.com/gaussia-labs/pygaussia/commit/c2cd608627a82631389066c0542139949b345823))

Thresholds resolve once, at construction: a supplied value wins, else the configured component's
  recommendation, else the Exploiter refuses to be built, naming both the component and the
  parameter. Nothing downstream re-resolves or branches on which path produced the number, so the
  search cannot silently run on a threshold nobody chose.

The realism estimator follows the paper. The query generator and the on-profile filter are gaussia's
  own construction, and each module docstring says so and says that substituting them changes what
  the search measures — the paper offers no method for either, so silence would read as fidelity it
  does not have.

Give StrategySpec.description a minimum length. Its clauses become the probe's attributes and from
  there the prose descriptor of the weakness map, which is the only thing about a strategy allowed
  to cross to the Exploiter. An empty description passed validation and then broke the Profiler
  mid-run — exactly the failure mode catalogue validation exists to prevent. The data model called
  the field documentation; it is load-bearing, and now says so.

- **roastme**: Interfaces and schemas
  ([`a097afe`](https://github.com/gaussia-labs/pygaussia/commit/a097afec2653c32a4b999191843f749e354f0d2e))

Ten abstractions in core/ — Grader, ProbeEngine, EntityEnumerator, HookVerifier, Transform,
  TargetAssistant, QueryGenerator, OnProfileFilter, RealismEstimator and CategorySearch — plus every
  model and validator the data model specifies.

All ten are plain ABCs, matching Embedder, Guardian and Reranker rather than the Pydantic-hybrid
  PIIDetector: the two recommended thresholds are resolved in one place downstream, so validating
  them at construction would make two of the ten structurally different from the rest for no gain.

The import direction is inverted exactly once. Every core/ module imports its models under
  TYPE_CHECKING, but schemas/roastme.py imports Grader at runtime, because Pydantic must resolve
  Principle.grader when it builds the model. No runtime cycle results, and the module docstring
  records the reason.

- **roastme**: Probe library
  ([`72ca663`](https://github.com/gaussia-labs/pygaussia/commit/72ca6638484921a1d568d0025d67fb8f88fe0685))

The four engines share one flow and differ in a single step, so the flow is a Template Method —
  ParticularisingEngine — and each engine supplies only the entities it can see for a kind and
  whether absence from that view decides the label. Writing the flow four times would make FR-021
  four promises that drift. This adds particularisation.py, which the plan's file table does not
  list; FR-020 names the stage, hence the filename.

That same abstraction is what keeps FR-023 true after composition. An engine declares what it can
  see; membership is the doc label. The graph engine sees every mention, so absence from it is
  absence. The retrieval engine sees the top k, a sample, so absence from it is only absence from a
  sample and every doc = 0 hook it emits is marked unreliable. Presence is reliable either way.

Merging is keyed on probe id, and the shipped engines scope ids by engine name. Engine-independent
  ids would collide across graph and retrieval over one corpus and collapse a confirmed absence
  label into an unreliable one, so FR-023 would stop holding at exactly the point composition
  happens. What merges is one probe surfaced twice; the survivor records every contributor.

The transform registry is a MappingProxyType: FR-025 closes the set, so it cannot be reopened at
  runtime. Order is load-bearing and documented — the three that invent a premise come first,
  identity last, since identity is what a control strategy asks for.

networkx joins the existing mypy ignore_missing_imports override. It ships no py.typed, and
  types-networkx would need a relock.

- **roastme**: Profiler and roast dataset
  ([`96eb904`](https://github.com/gaussia-labs/pygaussia/commit/96eb904f9d0f5dd1db1e0ef6f1718086dac8d9f0))

Scoring is pure functions over values, on stdlib arithmetic rather than numpy: the suite turns
  warnings into errors, and numpy's degenerate-variance warning fires at n = 1, which the standard
  error must reach and return zero for. One uncorrected formula serves both the weakness map and the
  S(c) penalty, so it collapses to the binomial form on binary values.

LogprobGrader is standalone rather than an extension of the shared judge, which reads the first
  generated token and raises instead of falling back — wrong for a reasoning model, and
  role_adherence depends on that behaviour. The fallback is driven by the two exceptions core
  already defines, not by a flag.

The profiler builds its weakness descriptors from the probes' own attributes and never sets a
  strategy identifier, rather than setting one and stripping it: no field on the result would hold
  the un-stripped value, so clearing it would be ceremony.

Correct one assertion in the dataset test. It required Toxicity to echo the dataset's session id,
  but Toxicity collapses every session into one aggregate labelled global_stream, so satisfying it
  would demand the very change the test's own docstring forbids. It now asserts the assistant id,
  which Toxicity does carry over from the dataset's metadata. The session id stays verified where it
  is ours to set: on the conversion itself.

- **roastme**: Reinforcement-learning search
  ([`d87d4c4`](https://github.com/gaussia-labs/pygaussia/commit/d87d4c424bc2dda883855a4e94d119146e375144))

The loop and the two abstractions it samples from and applies through live in policy_gradient.py and
  import nothing heavy, so the loop is testable on CPU in the default suite. policy_update.py is the
  only module that imports the training stack and the only one marked requires_gpu. The abstractions
  stay there rather than in core/: they are collaborators of one shipped search, not part of the
  specification a user implements against.

Split the discarded-candidate test in two. It asserted through rewards.get(attrs, 0.0), which passes
  whether the candidate is present with a zero or absent altogether — and those are not
  interchangeable. A candidate gated on realism was judged, so it belongs in the batch with a zero;
  one the target never answered was not judged, so rewarding it zero would teach the policy a
  verdict the run never reached, and would let a failed exchange count as a pass, which FR-016
  forbids. An update step centring rewards on a batch baseline gets a different gradient from each.

### Testing

- **roastme**: Red phase
  ([`d932ced`](https://github.com/gaussia-labs/pygaussia/commit/d932ced25bcb20265369868725914d48f4cee2d4))

Deterministic doubles, one per interface, plus the two-variant on-profile filter and realism
  estimator the threshold-resolution paths need. Hand-computed fixtures carry the derivation beside
  every literal, so assertions are arithmetic against the paper's equations rather than snapshots of
  whatever the code emits.

Every test that depends on an implementation module fails on that module being absent. Conformance,
  hermeticity and the fixture self-checks pass already: they exercise the interfaces and the
  doubles, both of which exist, and their passing is what confirms the interfaces landed.


## v1.1.0-b.2 (2026-07-15)

### Bug Fixes

- **evalhub**: Finalize MLflow models and retry dataset races
  ([`4810e0a`](https://github.com/gaussia-labs/pygaussia/commit/4810e0a474a4aec58999e7d64119be83cf8326ad))


## v1.1.0-b.1 (2026-06-25)

### Bug Fixes

- **privacy**: Address review — immutable domain weights and fail-fast corpus validation
  ([`9b03e3a`](https://github.com/gaussia-labs/pygaussia/commit/9b03e3a24724a1559417169c4fc6a30f1f208bdc))

- PrivacyDomainConfig: wrap criticality_weights/fn_severity_weights in MappingProxyType
  (FrozenWeights) so weights cannot drift after construction; the frozen model only blocked field
  reassignment, not in-place dict mutation. - metrics: add _validate_corpus, called once at the
  start of each batch() before the detector loop, rejecting non-PrivacyBatch turns and ground-truth
  span labels outside the domain classes. Both previously produced misleading metrics silently
  (empty GT -> all-FP; out-of-domain GT FN dropped by class_metrics). Validating outside
  PrivacyRanker's per-detector try/except makes a corpus data error fail hard rather than masquerade
  as N failed detectors. - tests: weight-map immutability + dict serialisation; corpus validation
  for Privacy and PrivacyRanker.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

- **privacy**: Set up each ranked detector once across sessions
  ([`88e86fe`](https://github.com/gaussia-labs/pygaussia/commit/88e86fe58e013887b437028d7654803947383687))

PrivacyRanker.batch runs once per dataset/session, and _evaluate_one called detector.setup() on
  every call — re-loading a heavy backend (Presidio, a HF pipeline) once per session. Memoise the
  load per detector (keyed by identity, cached load_time) inside the per-detector try/except so the
  fail-soft contract is preserved: a setup failure still yields a failed PrivacyMetric rather than
  aborting the whole ranking. Privacy already loads once in __init__; this aligns the ranker with
  that behaviour.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

### Code Style

- **privacy**: Format new test files and example with ruff
  ([`5986689`](https://github.com/gaussia-labs/pygaussia/commit/5986689a66ba1dae5ef9476d68f5446f98b998fa))

### Documentation

- **privacy**: Add a realistic multi-detector walkthrough to the notebook
  ([`3411ea9`](https://github.com/gaussia-labs/pygaussia/commit/3411ea987747a0a03c5fa1ec0a5bb1566835df31))

Part 2: a 4-class domain, a 3-turn labelled corpus (incl. a PII-free turn), and three detectors with
  distinct failure modes (clean, critical blind spot, noisy with FP + out-of-domain + overlap) so
  the per-class breakdown, the risk index and the ranking are all observable. Heavily commented.

- **privacy**: Add implementation plan and data model; fix spec writing errors
  ([`ec4f574`](https://github.com/gaussia-labs/pygaussia/commit/ec4f574a0da762826a2ca0070a68712c6f458fcf))

Plan gate for the 001-privacy-metric feature (spec gate merged in #11).

- Add specs/001-privacy-metric/plan.md (implementation plan) - Add
  specs/001-privacy-metric/data-model.md (Pydantic schema contracts) - Correct writing errors in the
  already-approved spec.md (recorded in a Revision Note): the regression baseline
  chatbot_v3_results.json was generated over the 500-turn corpus, not the eval_*_100.txt files;
  verification is now StubDetector-based with the sandbox reproduction as an optional opt-in
  integration test. Housekeeping: Status Draft->Planned, resolved NEEDS CLARIFICATION, FR-003
  set[str]->frozenset[str].

tasks.md is intentionally excluded; it is the next gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

- **privacy**: Align privacy.mdx with the minimal-docs tier
  ([`092b870`](https://github.com/gaussia-labs/pygaussia/commit/092b870c07c753fea0caeba945ab16339c7019f6))

Privacy has no LLM-judge and no statistical mode, so it belongs in the minimal tier
  (regulatory/vision) rather than the full template. Drop the H1 (frontmatter title renders it),
  fold the formula into Overview, move install extras into the trailing Note, and remove the Next
  Steps card group to match the house style. Document the new fail-fast corpus validation and
  once-per-run detector setup.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

- **privacy**: Document the custom-detector contract and fit guidance
  ([`80dad8f`](https://github.com/gaussia-labs/pygaussia/commit/80dad8fe8b799ce93dcf86d84f9c4ecd99aed3b2))

Add a "Bring your own detector" example subclassing PIIDetector (supported_classes / predict /
  optional setup), a contract table for the members a subclass must implement, guidance on choosing
  domain_fit / regulatory_fit, and clarify that span offsets are character-level half-open.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

- **privacy**: Metric docs page and runnable example (phase 7)
  ([`f3da504`](https://github.com/gaussia-labs/pygaussia/commit/f3da50407fcc66f3d729bce619c45f854cb12bde))

- docs/metrics/privacy.mdx following the full docs template (overview, install, usage,
  parameter/output tables, edge cases); registered in docs.json and the metrics overview. -
  examples/privacy/run.py: a dependency-free, self-explanatory script showing how to build a domain
  config, call Privacy.run / PrivacyRanker.run, and read the score, interpretation and risk fields.
  - Drop the now-unused spacy.* mypy override.

- **privacy**: Replace example script with a Jupyter notebook
  ([`de4cf98`](https://github.com/gaussia-labs/pygaussia/commit/de4cf98c9eb0905292e42f644dfa2e4f4093f6a6))

Match the examples/<metric>/jupyter/ convention used across the repo: convert
  examples/privacy/run.py into a runnable, dependency-free notebook (evaluate one detector, rank
  several, visualize the ranking) and point the docs page at it.

- **privacy**: Tasks gate — TDD task breakdown for 001-privacy-metric
  ([`c5083fc`](https://github.com/gaussia-labs/pygaussia/commit/c5083fcd4937b05323d4bf0cac7e6ae09d8052c8))

Generated per the speckit tasks-template: Schema & Contracts -> Tests (Red) -> Implementation per
  user story (Green) -> Polish. Tasks carry [P]/[US] tags, exact file paths from plan.md, and FR/SC
  traceability to spec.md.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

### Features

- **privacy**: Privacy/privacyranker metrics and detector adapters (phases 2-4)
  ([`969451d`](https://github.com/gaussia-labs/pygaussia/commit/969451d6a40dda97a3e8ead3b41ef65dd189cb25))

Implement the Domain-Adjusted Privacy Detection metric end to end (001-privacy-metric, T006-T020),
  test-first.

- Privacy(Gaussia): IoU span matching, greedy NMS, out-of-domain filtering and the full score/risk
  computation, one PrivacyMetric per dataset. - PrivacyRanker(Gaussia): per-detector fail-soft,
  score-descending ranking. - PresidioDetector and HuggingFacePIIDetector adapters behind their
  extras, plus a pure label canonicaliser; both import-fail cleanly without the extra. -
  statistical_mode is explicitly rejected (FR-020). Deterministic StubDetector tests assert every
  formula component against hand-computed values (SC-001).

Note: the paper's worked example cites Score_100 = 28.36, which still included the InfraScore
  removed in v3; the five-factor formula yields 32.97 (documented in test_privacy.py).

- **privacy**: Schemas, PIIDetector contract and extras (phase 1)
  ([`17629d6`](https://github.com/gaussia-labs/pygaussia/commit/17629d67496d03122c1fb9ef585299c391705094))

Add the Pydantic schemas (Span, PrivacyBatch, PrivacyDomainConfig, ClassMetrics, contributions,
  PrivacyMetric, PrivacyRanking) and the abstract PIIDetector strategy for the Domain-Adjusted
  Privacy Detection metric (001-privacy-metric, T001-T005).

- PIIDetector is a BaseModel+ABC so domain_fit/regulatory_fit are validated at construction (FR-005)
  while predict/supported_classes stay abstract. - *_100 fields and interpretation are recomputed in
  the model validator so they cannot drift; field names mirror the sandbox JSON for panel
  compatibility. - Declare privacy-presidio / privacy-huggingface extras (FR-017) and enable the
  pydantic mypy plugin to type-check the models without suppressions.


## v1.0.0-b.3 (2026-06-02)

### Documentation

- **role-adherence**: Add mdx docs and aws-lambda example
  ([`0bed671`](https://github.com/gaussia-labs/pygaussia/commit/0bed671ffdac884072fbbc8ec606aacdb3d8f32a))

- Add docs/metrics/role-adherence.mdx following the full docs template - Add
  examples/role_adherence/aws-lambda/ with handler, run, Dockerfile, README and deploy scripts -
  Register role-adherence optional dependency in pyproject.toml

### Features

- **metrics**: Add role adherence metric
  ([`590f383`](https://github.com/gaussia-labs/pygaussia/commit/590f383914751449d6ff0ab24c8be311679ffff5))

Implements RoleAdherence(R, T) = (1/n) Σᵢ adhere(tᵢ, T<i, R) from the Gaussia role adherence paper.
  Evaluates per-turn role compliance using an LLM judge with full conversation history as context,
  without ground truth.

- Add RoleAdherence metric with LLMJudgeStrategy (binary + continuous modes) - Add RoleAdherenceTurn
  / RoleAdherenceMetric output schemas - Add RoleAdherenceJudgeOutput to llm/schemas.py - Add binary
  and continuous system prompts for the role adherence judge - Fix judge._check_regex: escape JSON
  schema braces for ChatPromptTemplate - Fix judge._extract_json: fallback bare-JSON extraction +
  comma repair - Add chatbot_role optional field to Dataset schema - Add 25 unit tests covering
  batch, binary/continuous, strict/threshold modes - Add Jupyter notebook example with FinTrack
  dataset (2 sessions)

- **role-adherence**: Structured-output fallback when provider lacks logprobs
  ([`b79c586`](https://github.com/gaussia-labs/pygaussia/commit/b79c586397337d77e06ff6c679ddbdc1995f1f65))

Adds StructuredOutputJudgeStrategy and an optional fallback on LLMJudgeStrategy: when the provider
  does not expose logprobs, emit a warning and degrade to structured-output scoring instead of
  raising. Addresses the #8 review request for a logprobs/structured-output choice.

### Refactoring

- **judge**: Add logprob-based scoring path
  ([`2bcf2fa`](https://github.com/gaussia-labs/pygaussia/commit/2bcf2fa66599d0a5a279f2b1901cb4e7314aaa37))

Add Judge.check_logprob_binary() implementing P(YES)/(P(YES)+P(NO)) scoring via first-token logprobs
  with log-sum-exp aggregation across surface-form variants.

- Provider capability registry: raise LogprobsNotSupportedError for providers known not to expose
  logprobs (Anthropic, Gemini, Bedrock). No silent fallback to text-based scoring. - Raise
  LogprobsExtractionError when neither positive nor negative tokens appear in top_logprobs. -
  temperature exposed as parameter (default 1.0 per paper, None to inherit the model's own config).
  - No changes to existing Judge.check() path; no metric migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

- **llm**: Replace logprob provider allowlist with try/except on invocation
  ([`92b0774`](https://github.com/gaussia-labs/pygaussia/commit/92b0774e6276b076ebbadfd1df846cdda125ea74))

- **role-adherence**: Use logprob-based judge scoring
  ([`565be17`](https://github.com/gaussia-labs/pygaussia/commit/565be17768c0a826bfdabae1c93f5cdaaa28520d))

Migrates RoleAdherence to consume Judge.check_logprob_binary() from PR A. Replaces the previous
  text-based judge call (model returns a JSON score plus textual reason) with a calibrated [0, 1]
  score derived from the first-token YES/NO logprobs.

Changes: - LLMJudgeStrategy: drop `binary` / `use_structured_output` / `strict` / JSON-clause
  parameters. Expose `temperature` (default 1.0 per paper) and `top_logprobs` (default 10) —
  forwarded to Judge.check_logprob_binary(). - ScoringStrategy.score() returns float (no reason).
  The ABC is preserved to allow future deterministic strategies (paper evaluated several; out of
  scope for this PR). - Drop `include_reason` from RoleAdherence and `reason` from RoleAdherenceTurn
  / RoleAdherenceJudgeOutput. The logprob path does not produce reasoning; a second model call would
  be required. - Replace `role_adherence_binary_system_prompt` and
  `role_adherence_continuous_system_prompt` with a single `role_adherence_judge_system_prompt`
  (YES/NO). - Update tests, mdx docs, aws-lambda example and Jupyter notebook to reflect the new API
  and the provider-compatibility constraint (Anthropic/Gemini/Bedrock unsupported).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>


## v1.0.0-b.2 (2026-05-13)

### Bug Fixes

- **evalhub**: Satisfy release type checks
  ([`e7582af`](https://github.com/gaussia-labs/pygaussia/commit/e7582afbbdd54052c64b59570e519fae725a4cbd))

- **guardians**: Address PR review — runtime bugs and schema consistency
  ([`c42fe90`](https://github.com/gaussia-labs/pygaussia/commit/c42fe90b668a55bbec605bfac4bacd34552bf015))

- HuggingFaceGuardianProvider: instantiate tokenizer via AutoTokenizer.from_pretrained(model) in
  __init__ instead of passing None; tokenizer was used in _parse_output and _get_probabilities,
  causing AttributeError at runtime - HuggingFaceGuardianProvider.infer: replace self.model.device
  with next(model.parameters()).device; self.model is a str so .device would raise AttributeError at
  runtime - GuardianLLMConfig.overrides: change mutable dict default {} to
  Field(default_factory=dict) for consistency with project Pydantic patterns -
  OptimizationResult.history, MIPROv2Result.demos/trials: change mutable list defaults [] to
  Field(default_factory=list) for same reason - test_toxicity: add missing context= argument to two
  batch() call sites to match updated Gaussia.batch() signature - Add test_huggingface_provider.py
  covering tokenizer initialization and device resolution in infer()

- **tests**: Assign lazy-loaded property to _ to satisfy ruff B018
  ([`1a99bcf`](https://github.com/gaussia-labs/pygaussia/commit/1a99bcf94dc24069e9ec1a80f853741e78201885))

- **types**: Eliminate all type: ignore comments and fix root causes
  ([`fc4a1b0`](https://github.com/gaussia-labs/pygaussia/commit/fc4a1b07a8c767da229ae07c303cfba35c77e33b))

Replace all 25 occurrences of # type: ignore across the codebase with proper type fixes:

- pyproject.toml: add follow_imports = skip for torch, transformers, langchain and langchain_core to
  silence incomplete third-party stubs without per-line suppressions - schemas/bias.py: correct
  tokenizer parameter from AutoTokenizer to PreTrainedTokenizerBase in LLMGuardianProvider -
  guardians/__init__.py: annotate self.tokenizer as PreTrainedTokenizerBase in IBMGranite and
  LLamaGuard; remove all arg-type suppressions - embedders/qwen.py, rerankers/qwen.py: lazy-init
  fields typed as PreTrainedTokenizerBase | None and PreTrainedModel | None with assert is not None
  guards in properties - llm/judge.py, guardians/llms/providers.py: annotate json.loads and
  response.json() return values as dict[str, Any] to resolve no-any-return -
  statistical/frequentist.py: use isinstance(v, (int, float)) inline to narrow float | dict[str,
  Any] union without suppression - metrics/agentic.py, metrics/toxicity.py: assert isinstance before
  indexing into float | dict[str, Any] in Bayesian code paths - metrics/toxicity.py: fix
  score_cluster key type to dict[int, float], use GroupProfiling.model_validate() instead of passing
  raw dict - prompt_optimizer/base.py: use isinstance-filtered list comprehension to narrow Dataset
  | StreamedBatch union - mipro/mipro.py: fix _collect_examples return type from object to Batch -
  schemas/generators.py, gepa/gepa.py, mipro/proposer.py, llm/judge.py: remove stale suppressions
  made redundant by follow_imports = skip

Add 38 targeted tests verifying runtime correctness of all changes: -
  tests/guardians/test_ibm_granite.py: tokenizer loading, safe/unsafe token configuration,
  is_biased() return values and prompt structure - tests/guardians/test_llama_guard.py:
  chat_completions flag, content list format, categories construction, is_biased() return values -
  tests/embedders/test_qwen_embedder.py: lazy init, caching, eval() call -
  tests/rerankers/test_qwen_reranker.py: same pattern as embedder

mypy: Success — no issues found in 88 source files

pytest: 441 passed, 83.93% coverage

- **types**: Resolve 107 mypy errors and remove ignore_errors suppression
  ([`44b3ba0`](https://github.com/gaussia-labs/pygaussia/commit/44b3ba0b7b4f26852e98c288123a47a52b1a8851))

Resolves gaussia-labs/pygaussia#1.

Remove the blanket `ignore_errors = true` overrides for 16 modules from `pyproject.toml` and fix
  each underlying type error:

Group A — missing stubs for optional dependencies: - `rerankers/qwen.py`, `embedders/qwen.py`: add
  `# type: ignore[import-not-found]` for torch imports; annotate lazy-init properties with proper `T
  | None` types - `metrics/humanity.py`: mark scipy import as untyped

Group B — `batch()` signature mismatch with base class: - `metrics/toxicity.py`: reorder params to
  match `(session_id, context, assistant_id, batch, language)` - `metrics/bias.py`: fix `language`
  default from `str = "en"` to `str | None = "english"`

Group C — FrequentistMode and BayesianMode return types too narrow: - `statistical/frequentist.py`,
  `statistical/bayesian.py`: widen all method return type declarations to `float | dict[str, Any]`
  matching the base class; switch dict parameters to `Mapping` for covariance -
  `statistical/base.py`: update abstract method signatures to use `Mapping` instead of `dict` so
  subclasses can accept narrower callers

Group D — Agentic.run() override with incompatible signature: - `metrics/agentic.py`: remove `k` as
  a positional param, thread via `**kwargs` instead; add proper type annotations for local variables

Group E — LangChain / attr-defined errors: - `llm/judge.py`: add `# type: ignore[no-any-return]` on
  `json.loads` - `guardians/llms/providers.py`: fix `super().__init__()` positional arg order;
  annotate return types; add targeted ignores for torch attrs - `prompt_optimizer/mipro/mipro.py`:
  remove stale union-attr ignores

Group F — BaseContextLoader.load() signature mismatch: - `schemas/generators.py`: widen
  `load(source: str)` to accept `str | list[str]` matching the concrete implementation

Remaining targeted fixes: - `metrics/best_of.py`: fix `result_reasoning` type from `str | None` to
  `dict | None` matching `BestOfContest.reasoning` - `metrics/humanity.py`: add `defaultdict` type
  annotation; guard `None` language with fallback - `metrics/toxicity.py`: use `np.ndarray` for
  accumulated embeddings; add ignores for numpy integer subclass check - `metrics/regulatory.py`,
  `metrics/vision.py`: targeted fixes - `prompt_optimizer/schemas.py`: make `MIPROv2Result` extend
  `OptimizationResult` (LSP compliance); rename `trials_run` to `iterations_run`; update test
  accordingly - `guardians/__init__.py`: remove stale type: ignore comments

Pin Python to 3.13 via `.python-version` to avoid pydantic v1 incompatibility with Python 3.14.

`uv run mypy src/gaussia` now reports: Success: no issues found in 88 source files.

### Chores

- Remove spec plan from tracked files
  ([`713a8a7`](https://github.com/gaussia-labs/pygaussia/commit/713a8a76c726a597859932321ae87b2a94aca9bb))

### Documentation

- Add PyPI badges to README
  ([`83426ea`](https://github.com/gaussia-labs/pygaussia/commit/83426eac5a227261eed418062d3ba54eaa4c037c))

- Add usage examples for all metrics and deployment targets
  ([`b0d08da`](https://github.com/gaussia-labs/pygaussia/commit/b0d08daad035cbda61ee671f0f1df233633e2bf4))

- Update trials_run references to iterations_run
  ([`a83e1a7`](https://github.com/gaussia-labs/pygaussia/commit/a83e1a76976c717cec4746aa55ffd109b5d00a6c))

Follow-up to the MIPROv2Result schema change in the previous commit. The `trials_run` field was
  renamed to `iterations_run` to align with the `OptimizationResult` base class. Update all
  references:

- examples/prompt_optimizer/mipro/jupyter/mipro.ipynb: result.trials_run → result.iterations_run -
  tests/prompt_optimizer/test_mipro.py: rename test method accordingly

### Features

- **evalhub**: Add built-in provider adapter
  ([`469cf9a`](https://github.com/gaussia-labs/pygaussia/commit/469cf9a053c4a590a7c905a6f14d727299a5a815))

- **guardians**: Add provider overrides support and fix null content crash
  ([`869ec57`](https://github.com/gaussia-labs/pygaussia/commit/869ec575d40504ec61a7caa23c101a55edfc6aa4))

Resolves gaussia-labs/pygaussia#2.

- Add `overrides: dict[str, Any] = {}` field to `GuardianLLMConfig` so callers can pass extra HTTP
  body fields (e.g. OpenRouter provider routing, transforms) without subclassing - Thread
  `overrides` through `IBMGranite` and `LLamaGuard` constructors into the underlying
  `LLMGuardianProvider` instance - Spread `**self._overrides` into both `_with_chat_completions` and
  `_with_completions` request bodies in `OpenAIGuardianProvider` - Guard against null message
  content in `_parse_guardian_response`: when `choice["message"]["content"]` is `None` return
  `(False, 1.0)` instead of crashing with a `TypeError`


## v1.0.0-b.1 (2026-04-09)


## v1.0.0 (2026-04-09)

### Bug Fixes

- Resolve mypy errors and suppress pre-existing type issues
  ([`1da8b78`](https://github.com/gaussia-labs/pygaussia/commit/1da8b7881edcb6bcf0ed5f544205abc013d98036))

- **ci**: Install all extras for test dependencies
  ([`5ab32cf`](https://github.com/gaussia-labs/pygaussia/commit/5ab32cf9b647653c6e21dd1a7189d2d1e6238061))

- **ci**: Pin Python 3.13 in release workflow
  ([`08ca47d`](https://github.com/gaussia-labs/pygaussia/commit/08ca47df647f8aeb786f3e4a30f09c0a6c147e10))

- **ci**: Use master branch in docs sync workflow trigger
  ([`88b2bc4`](https://github.com/gaussia-labs/pygaussia/commit/88b2bc41b955c519735a3b4a2fab2df1ef73ef1d))

- **ci**: Use python -m build for semantic-release container
  ([`8a1894e`](https://github.com/gaussia-labs/pygaussia/commit/8a1894ed3cf99d0b97ed1cf56d8c73373423f814))

- **core**: Export __version__ in __all__
  ([`a8fcb94`](https://github.com/gaussia-labs/pygaussia/commit/a8fcb9422b7922d7838f2435fb1be0b0a1ae8fa3))

### Chores

- Remove metric-creator skill
  ([`9d5fc72`](https://github.com/gaussia-labs/pygaussia/commit/9d5fc72e79fcd73b6315ae3a32192d968be6aefa))

### Code Style

- **core**: Simplify module docstring
  ([`77778f7`](https://github.com/gaussia-labs/pygaussia/commit/77778f7caf236d195fd05e298342925fb7553ec1))

- **docs**: Remove trailing newline from docs.json
  ([`3e0fd8a`](https://github.com/gaussia-labs/pygaussia/commit/3e0fd8ae350c7afc6a786c694572a8f6408086ae))

### Continuous Integration

- Add pyproject.toml to release trigger paths
  ([`7e7871d`](https://github.com/gaussia-labs/pygaussia/commit/7e7871df9bc97d61d7382f4cdd0653c31af35f70))

- Add release workflow and fix semantic-release config
  ([`ba088ae`](https://github.com/gaussia-labs/pygaussia/commit/ba088aeafdac03fcc398d7e3e699c539634c0cf1))

- Restrict release trigger to source code changes only
  ([`3f9c7e5`](https://github.com/gaussia-labs/pygaussia/commit/3f9c7e543f8673b29a84c0ffed6defe37e02c8e3))

- Trigger release workflow on workflow file changes
  ([`5c35881`](https://github.com/gaussia-labs/pygaussia/commit/5c358818c177de1bcd1b49da5f3cf9d46a4bc3f2))

- Use commit short SHA in docs sync branch name
  ([`e016552`](https://github.com/gaussia-labs/pygaussia/commit/e01655232c9f28da18c90f4464256b81f2f4150f))

### Documentation

- Add mintlify documentation and sync workflow
  ([`e000077`](https://github.com/gaussia-labs/pygaussia/commit/e0000772d342452518800c44980c6ed3ee156521))

- Add MIT license text
  ([`e8b0307`](https://github.com/gaussia-labs/pygaussia/commit/e8b0307fbb363a5b093b37699229ba07b3b1f0b1))

- Add README with metrics overview and usage examples
  ([`622b2b6`](https://github.com/gaussia-labs/pygaussia/commit/622b2b67bae131050b89d1471a71c9489aa0101b))

- Expand metric guides and add metrics overview page
  ([`6702c2b`](https://github.com/gaussia-labs/pygaussia/commit/6702c2b43e620c042b900f74d0719740786bb7da))

- Update SDK display name
  ([`12dac8c`](https://github.com/gaussia-labs/pygaussia/commit/12dac8c028334e7b76f0fb179c0ea99bef25fd40))

### Features

- Adopt paper-driven SDD workflow
  ([`19fe757`](https://github.com/gaussia-labs/pygaussia/commit/19fe75786ce4535794dcb366f9b92e0ae67f37c7))

Add SDK-specific constitution extension, CONTRIBUTING guide documenting the paper-to-code lifecycle,
  and update CLAUDE.md to reference the shared speckit skill from gaussia-labs/skills.

- Initialize pygaussia from fair-forge migration
  ([`371d6ca`](https://github.com/gaussia-labs/pygaussia/commit/371d6cad9d5b574f50f4eecdd35f871fca00b086))

### Refactoring

- Rename package from pygaussia to gaussia
  ([`1d7e271`](https://github.com/gaussia-labs/pygaussia/commit/1d7e271b65fa0a0de27d11cddc7830f3b25d73d3))
