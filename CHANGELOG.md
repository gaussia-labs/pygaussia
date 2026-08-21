# CHANGELOG


## v1.1.0 (2026-08-21)

### Build System

- **roastme**: Declare the extras, and let the gates read the subsystem
  ([`e27796d`](https://github.com/gaussia-labs/pygaussia/commit/e27796d0fb8bc4c3135f5a36ab79698dadab274f))

Three changes, all of them about the new code being seen correctly rather than about what it does.

The `roastme` and `roastme-rl` extras. Deliberately outside `metrics` and `all`: the three
  corpus-reading engines want an embedder and a graph library, and only the policy-gradient update
  step wants the training stack.

`ruff` was linting against py313 while the package declares 3.11. At py313 it asked for PEP 695
  generics, which are a *parse* error before 3.12 — code that `mypy --python 3.11` in CI cannot
  read, and that no local gate would catch.

`mypy` gets the pydantic plugin, so a list of a Batch subclass stops reading as an incompatible
  argument, plus the third-party modules the subsystem imports: networkx, peft and accelerate as
  missing stubs, and torch and transformers as skipped, since their annotations resolve to Any and
  `warn_return_any` then fires at every boundary that returns a Tensor.

### Features

- **roastme**: The Roast Me adversarial evaluation subsystem
  ([`e4989f7`](https://github.com/gaussia-labs/pygaussia/commit/e4989f7bf99ccdc2d9f8f84b03b20038c5ef38bb))

Profile an assistant's weaknesses from tagged adversarial probes, then search for the categories of
  realistic question that break it reproducibly. A generator subsystem, not a metric: nothing
  subclasses Gaussia and nothing is registered in gaussia.generators. What enters the metric
  pipeline is the Roast Dataset it emits.

What lands here:

- the eleven interfaces in gaussia.core, of which nine ship a reference implementation, and the
  schemas in gaussia.schemas.roastme; - the Probe Library with its five engines, three of which read
  a corpus and sit behind the roastme extra, while the grounded one and the two model-driven
  collaborators need only the user's model; - the Profiler, the Exploiter with threshold resolution,
  both category searches and the shipped logprob grader; - the docs page, the four Claude Code
  skills, the catalogue schema examples, the two notebooks and the specification.

Two pieces here are not roastme's own and are carried because roastme cannot import without them:
  gaussia.llm.structured, which its three model-driven components use to bind a schema, and
  chatbot_role on Dataset, an optional field that arrived with role adherence and that a roastme
  test asserts is left unset. Both are additive and change no existing behaviour.

Neither extra is part of gaussia[metrics] or gaussia[all], so whoever only profiles pays for neither
  training nor retrieval.


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
