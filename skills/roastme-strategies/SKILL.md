---
name: roastme-strategies
description: Author the strategy entries of a Roast Me catalogue — the interaction patterns that turn a knowledge base into probes, including the controls. Use when setting Roast Me up for a domain, or when validate_catalogue rejects a transform or an entity kind.
argument-hint: <domain, or the path to your catalogue>
---

# Roast Me strategies

A **strategy** is one way of asking something: which kind of entity it operates on, how it transforms
that entity, and whether what comes out is documented or invented.

```python
StrategySpec(
    id="strategy-fake-entity",
    name="Ask about a near-miss entity",
    description="leans on an entity the base does not contain, phrased as ordinary traffic",
    plugin="plugin-invention",     # a plugin id from your catalogue — or None, see below
    entity_kind="policy-code",     # some configured engine must declare it can handle this
    transform="mutate_to_fake",    # one of exactly four
    doc=0,                         # 0 = the premise is invented, 1 = it is in the base
    phrasing_hint="What does this cover",
)
```

## 1. `description` is not documentation

Its comma-separated clauses become the probe's **attributes**, and from there the prose descriptor of
the weakness map. That descriptor is the only thing about a strategy that reaches the Exploiter — your
identifiers never do — so the search can only ground a category in what you wrote here.

```
"leans on an entity the base does not contain, phrased as ordinary traffic"
  -> attrs = ["leans on an entity the base does not contain", "phrased as ordinary traffic"]
```

Two consequences worth planning around:

- **write clauses, not a sentence.** One clause per property you would want reported separately. A
  description with no commas yields one attribute, and a category can then only be that whole thing
  or nothing.
- **write them so a stranger could read a finding.** `"invented attribute, pricing"` produces a
  report saying *"breaks no_invention on invented attribute, pricing"*. `"strat 3 variant b"` produces
  a report nobody can act on.

An empty description is rejected at construction, because a strategy with nothing sayable would fail
mid-run instead of at validation.

## 2. The four transforms, and nothing else

The set is closed. `validate_catalogue` rejects anything outside it.

| `transform` | What the premise becomes |
|---|---|
| `mutate_to_fake` | a near-miss of a real entity, which does not exist |
| `flip_value` | the real entity with one of its values changed |
| `flip_fact` | the real entity with a documented fact asserted backwards |
| `keep_real` | the entity untouched |

`doc` should agree with the transform: `keep_real` leans on something documented (`doc=1`), the other
three on something that is not (`doc=0`). The Probe Library recomputes it against the corpus, so a
disagreement means you have described a different attack than the one you meant.

## 3. Every catalogue needs at least one control

A strategy with **`plugin=None` is a control**. Its probes are sent and graded like any other and then
excluded from every rate.

Without one you cannot tell *"the assistant is broken"* from *"the probe was unfair"*. If your
controls fail too, the probes are the problem, not the assistant — and that is the only way to find
out. Pair each control with `transform="keep_real"` and `doc=1`.

## 4. `entity_kind` has to be something an engine can handle

`validate_catalogue` rejects an `entity_kind` that no configured engine declares, with *"no configured
engine declares it can produce the entity kinds"*. This check exists because the alternative is worse:
the run completes, produces zero probes for that strategy, and the report looks clean.

Ask your engines rather than assuming:

```python
for engine in engines:
    print(engine.name, sorted(engine.entity_kinds))
```

Use one `entity_kind` constant across the catalogue unless you genuinely have two kinds of entity;
a typo here is the single easiest way to lose a whole strategy silently.

## 5. `phrasing_hint` goes into a prompt

It reaches the query the probe is written as, so it has to be in the language of your knowledge base.
A Spanish corpus with English hints produces probes no real user would send, and the realism gate will
be measuring your hint rather than the assistant.

## 6. Verify

```python
from gaussia.generators.roastme.probes.catalogue import validate_catalogue

validate_catalogue(catalogue, contract, engines)
```

The failures that concern strategies:

- `duplicate strategy identifiers: [...]`
- `strategies name plugins the catalogue does not carry: [...]` — a `plugin` that is not a declared
  plugin id;
- `strategies name transforms outside the closed set [...]`
- `no configured engine declares it can produce the entity kinds: [...]`

Then check the two things validation cannot: that a control exists, and that the attributes read well.

```python
controls = [s.id for s in catalogue.strategies if s.plugin is None]
print("controls:", controls or ">>> none — add one before running")

from gaussia.generators.roastme.probes.particularisation import strategy_attributes

for strategy in catalogue.strategies:
    print(f"{strategy.id:<26} {strategy_attributes(strategy)}")
```

Then generate probes and read a handful before spending a single call on the assistant:

```python
probes = ProbeLibrary(engines).generate(documents, catalogue)
for probe in probes[:5]:
    print(probe.strategy, "|", probe.query)
```

If a query does not read like something a real user would send, fix the `phrasing_hint` or the
transform — not the grader.
