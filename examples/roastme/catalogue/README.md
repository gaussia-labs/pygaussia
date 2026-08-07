# Roast Me — catalogue schema examples

`catalogue.json` is a **schema example**, not a catalogue you can run. Every identifier and every
sentence in it is a placeholder for something you write.

Gaussia ships **no domain catalogue** and never will. A risk taxonomy *is* what the evaluation
measures, so a shipped one would quietly become a cross-user standard nobody chose. What gaussia
owns is the shape, and the validation that rejects a catalogue before a single probe is generated.

Gaussia also owns no file format for it. `Catalogue`, `PluginSpec` and `StrategySpec` are Pydantic
models: JSON is used here because it reads well in a repository, and YAML, TOML or Python literals
are equally fine.

## Loading it

```python
import json
from pathlib import Path

from gaussia.schemas.roastme import Catalogue

catalogue = Catalogue.model_validate(json.loads(Path("catalogue.json").read_text()))
```

That call checks the *shape*. The six semantic rejections need the contract and the engines that will
run, so they live in one function you call before generation:

```python
from gaussia.generators.roastme.probes.catalogue import validate_catalogue

validate_catalogue(catalogue, contract, engines)   # raises ValueError, naming what is wrong
```

## `PluginSpec` — a risk family

One plugin maps to exactly one principle of your behavioral contract.

| Field | Meaning |
|---|---|
| `id` | Referenced by strategies, and recorded on every probe and graded outcome. Yours; no library behaviour depends on it. |
| `name` | Documentation for the report. Not consumed by any logic. |
| `description` | Documentation for the report. Not consumed by any logic. |
| `principle` | The `Π` member this family attacks. Must exist in the contract. |

## `StrategySpec` — an interaction pattern

Which kind of entity it operates on, how it transforms it, and whether the resulting hook is
documented or invented.

| Field | Meaning |
|---|---|
| `id` | The aggregation descriptor of the weakness map. Recorded on the probe; never crosses to the Exploiter. |
| `name` | Documentation. |
| `description` | **Load-bearing.** Its comma-separated clauses become the probe's attributes, which is what lets the Exploiter ground a category in part of a pattern rather than all of it. Write prose a reader of a failure report would understand. |
| `plugin` | The risk family this strategy serves. **Absent means control** — see below. |
| `entity_kind` | The kind of entity this strategy needs. Your own vocabulary; it has to match what a configured engine declares it can produce. |
| `transform` | How the real entity becomes the probe's premise. One of the four below. |
| `doc` | The expected grounding label of the resulting hook: `1` documented, `0` invented. |
| `phrasing_hint` | Injected into the generation prompt, so it must be in the knowledge base's **language**. |

## The four transforms

The set is closed at four, because `transform` is the one field whose value changes what a probe
*means*. These keys are interface, not naming preference: a catalogue is data you have already
written, so they do not change.

| Key | Turns the real entity into |
|---|---|
| `mutate_to_fake` | A plausible sibling the base does not contain |
| `flip_value` | The same entity with a figure the base does not carry |
| `flip_fact` | The documented fact asserted the other way round |
| `keep_real` | Itself, unchanged |

A transform decides the **text** of a premise and never its label. Whether the premise turns out to
be documented is the generating engine's call, derived from its own view of the knowledge base's
boundary — so `flip_value` can legitimately produce a documented hook when the shifted figure
happens to exist in the base.

## A control is a strategy with no plugin

`strategy-control` in the example carries `"plugin": null`. That is the **only** mechanism by which a
control is recognised — structurally, from an absent plugin, never from an identifier or a naming
convention. A control puts no principle under test, so its probes carry none, and they are excluded
from every violation-rate aggregate while staying in the graded record.

Compare it with `strategy-documented-under-test` directly above it. The two are identical except for
`plugin`: same `entity_kind`, same `keep_real` transform, same `doc: 1`, same phrasing. One is
scored and one is not.

That is the point worth internalising: **`doc: 1` does not mean control.** The two fields answer
different questions — `doc` says whether the entity exists, `plugin` says whether a principle is on
the line. To bring "did it answer real content correctly" inside the violation rate, add a principle
for it and point a `keep_real` strategy at a plugin that serves it, exactly as
`strategy-documented-under-test` does. A control stays out because nothing is on the line, not
because its entity is real.

## What validation rejects

All six before generation runs, so a run either generates or fails at the start:

1. a `principle` no contract principle resolves;
2. a `plugin` the catalogue does not carry (a *missing* plugin is a control, not a dangling one);
3. a `transform` outside the four above;
4. a `doc` outside `{0, 1}`;
5. a duplicate plugin or strategy identifier;
6. an `entity_kind` **no configured engine declares it handles**.

The last one matters more than it looks. `entity_kind` is your own vocabulary and gaussia never
learns what it means, so without that check a plural typo would validate cleanly and yield an empty
probe set with no error at all — the failure mode hardest to notice.

## Further reading

The `docs/advanced/roastme.mdx` page covers the surrounding subsystem, and
`examples/roastme/jupyter/roastme.ipynb` runs a complete offline evaluation, this catalogue shape
included.
