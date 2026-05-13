# CHANGELOG


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
