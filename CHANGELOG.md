# CHANGELOG


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
