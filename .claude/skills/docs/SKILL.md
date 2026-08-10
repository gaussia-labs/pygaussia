---
name: docs
description: Write or update a Gaussia .mdx documentation page for a metric or module, in the tier the repo uses for its kind, and register it in both navigation files.
argument-hint: <metric-or-module-name>
---

# Gaussia documentation page

Produce `docs/<dir>/<page>.mdx` for the subject named in the argument and register it in **both**
navigation files.

A page that exists and is registered nowhere does not publish. That is this task's default failure
rather than a hypothetical: `docs/metrics/role-adherence.mdx` is on `develop` and appears in neither
registry, and `docs/metrics/privacy.mdx` appears in only one. Step 4 is not optional.

## 1. Which directory

| What you are documenting | Directory |
|---|---|
| something whose `run()` returns a `BaseMetric` subclass | `docs/metrics/` |
| anything else — generator, optimiser, subsystem, tooling | `docs/advanced/` |

Verify instead of inferring from the name. The metric modules are flat files under
`src/gaussia/metrics/<name>.py`, and the classes they emit live in `src/gaussia/schemas/<name>.py`:

```bash
grep -rln "(BaseMetric)" src/gaussia/schemas/          # the eleven metric schemas
grep -rln "(BaseMetric)" src/gaussia/<path-to-subject>/ # your subject: no hit means advanced/
```

Nothing subclassing `BaseMetric` means `docs/advanced/`, however metric-like the subject sounds. Some
subsystems produce input *for* metrics rather than scores; those are `advanced/`.

The file name is kebab-case and often differs from the module name: `role_adherence` →
`role-adherence.mdx`, `bestof` → `best-of.mdx`.

## 2. Which tier

`docs/metrics/` has two, and they are recognisable from the outside: an H1, an `## Installation`
section and a `## Statistical Modes` section co-occur in the five full pages and are absent from all
six minimal ones.

| Tier | Pages | Lines |
|---|---|---|
| full | `context`, `conversational`, `agentic`, `toxicity`, `role-adherence` | 300–400 |
| minimal | `best-of`, `bias`, `humanity`, `regulatory`, `vision`, `privacy` | 78–215 |

`docs/advanced/` has a single tier, shaped like the minimal one.

**The tier is not derivable from the code — do not try.** It is a choice about how much the page
carries, and the obvious candidate rule is false: `bias`, `regulatory` and `privacy` all accept a
`statistical_mode` and all have minimal pages. Their pages name the parameter in passing — 3, 3 and 1
mentions — where the full pages average 16.

So decide deliberately, and know what you are deciding:

```bash
grep -rn "statistical_mode" src/gaussia/metrics/<name>.py
```

A hit means the subject supports Frequentist/Bayesian modes. Writing a minimal page for it is choosing
not to document a feature it has — legitimate for a first pass, but say so rather than leaving it
looking unsupported. If in doubt, ask which tier is wanted; it is a five-line question and a
three-hundred-line difference.

Whichever you pick, keep the three markers consistent. A page with an H1 and no `## Statistical Modes`,
or an `## Installation` section in sentence case, is neither tier and reads as unfinished.

**Read the closest sibling in the tier you picked before writing anything.** That file is the
specification for MDX component syntax and section order. This skill tells you which one to read and
what has to appear; it does not restate their contents, which drift.

## 3. Section order

### full — `docs/metrics/`, documenting the statistical modes

1. frontmatter: `title`, `description`
2. `# <Name> Metric` — the only tier with an H1 — then one paragraph: what it measures, what it
   aggregates over, and what the per-item list preserves
3. `## Overview` — bullets: key score, aggregate, per-item detail, statistical modes
4. `## Installation` — `uv add gaussia` with the extra, plus the LLM provider
5. `## Basic Usage` — `<CodeGroup>`, one tab per statistical mode
6. `### Required Parameters` then `### Optional Parameters` — tables. Required: Parameter, Type,
   Description. Optional adds Default.
7. `## Statistical Modes` — `<Tabs>`, stating which CI fields are `None` in each mode
8. one or more sections for whatever is specific to this subject
9. `## Output Schema` — one `###` per emitted class with field types, then a score-interpretation table
10. `## Complete Example` — self-contained and runnable, including a real `Retriever` subclass
11. `## LLM Provider Options` — `<CodeGroup>` per provider. Optional: 2 of the 5 full pages have it.
12. `## Best Practices` — `<AccordionGroup>`, 3–4 entries specific to this subject
13. `## Next Steps` — `<CardGroup cols={3}>`

### minimal — the other six `docs/metrics/` pages, and all of `docs/advanced/`

No H1. Sentence case, not Title Case: `## Output schema`, `## How it works`.

1. frontmatter
2. `## Overview`, with the subject's name bold in the first sentence
3. one section naming the central concept — existing pages use `## How it works`,
   `## Protected attributes`, `## Verdicts`, `## Dimensions`, `## Available algorithms`
4. `## Usage`
5. `## Parameters`
6. `## Output schema`, one `###` per class
7. optional, when the subject has one: the extension point — `## Guardian interface`,
   `## Custom scorer`, `## Context loaders` with a `### Custom loader` under it

Add `## Installation` when the subject has an extra in `[project.optional-dependencies]` — a reader who
skips it gets an `ImportError`. Follow this rather than the precedent: of the four advanced pages only
`roastme` has the section, and `explainability` omits it while declaring a `torch` extra.

## 4. Register the page — required

Two files, two different JSON shapes, two different path forms.

**`docs/docs.json`** — `navigation.tabs[]`, each tab holding `groups[].pages[]`. Paths carry no prefix.

| Directory | Tab | Group |
|---|---|---|
| `docs/metrics/` | `Metrics` | `Evaluation metrics` |
| `docs/advanced/` | `Advanced` | `Advanced topics` |

**`docs/docs-sync.json`** — `navigation` is a **single object**, not a list of tabs, so the path is
`navigation.groups[]`. Its groups are `Getting started`, `Core concepts`, `Metrics`, `Advanced`. Every
page is prefixed with this file's own `target_dir`:

```json
"pages": ["sdks/python/metrics/context", "…", "sdks/python/metrics/<your-page>"]
```

Read `target_dir` from the file rather than hard-coding `sdks/python`. This is the file that publishes:
it syncs `source_dir` into `target_dir` of `target_repo` on `target_branch`.

## 5. Verify

Registration, mechanically:

```bash
python3 - <<'PY'
page = "metrics/<your-page>"          # or "advanced/<your-page>"

import json
docs = json.load(open("docs/docs.json"))
sync = json.load(open("docs/docs-sync.json"))

in_docs = any(page in g["pages"] for t in docs["navigation"]["tabs"] for g in t.get("groups", []))
prefixed = sync["target_dir"].rstrip("/") + "/" + page
in_sync = any(prefixed in g["pages"] for g in sync["navigation"]["groups"])

print("docs.json:", in_docs, "| docs-sync.json:", in_sync)
assert in_docs and in_sync, "unregistered page: it will not publish"
PY
```

Then run every python block on the page. A snippet nobody has executed is the other way these pages go
wrong. If a block cannot run offline — it needs a provider key, a GPU, a live endpoint — say so in the
prose beside it rather than leaving a reader to find out.

Finally, state in your report which blocks you ran and which you could not, and whether the page is
registered in both files. "The file was created" is not the deliverable.
