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

## 0. First, check that an engine can see your entities at all

Do this before writing a single strategy. Three of the four shipped engines — retrieval, graph,
multi-hop — find entities with **one regex**, in `probes/particularisation.py`:

```python
_MENTION = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")
```

An entity is a run of alphanumerics joined by a hyphen or underscore. `POLICY-1`, `FORM-7`,
`entity_alpha` match. `Cuenta Digital Libre`, `Mastercard Infinia`, `insulin` do **not**.

**The failure is silent.** An empty boundary makes the engine return `[]` with no error, and a corpus of
web-scraped prose returns something worse than nothing: junk that looks like entities. Run this against
your own corpus before you trust anything:

```python
import re
mention = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")
found = sorted({m for d in documents for m in mention.findall(d.content)})
print(len(found), found[:30])
```

If what comes back is phone numbers, PDF filenames, URL slugs or footer anchors, those three engines
will generate `len(found) × len(strategies)` nonsense probes and the run will look successful. On one
real bank corpus this was 210 junk entities across 12 documents.

**The escape is `EnumerationProbeEngine` plus an `EntityEnumerator` you write.** It is the only engine
that does not use the regex, the only one that can defend an absence label, and it needs no extra — no
torch, no networkx. Its contract is **completeness**: return the whole set, because a sample turns every
absence label into a guess. Slice the generated probe list afterwards if you need fewer.

```python
class MyEnumerator(EntityEnumerator):
    def enumerate_entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        if kind != "my-kind":
            return frozenset()          # unknown kind: produce nothing rather than invent
        return frozenset(...)           # the complete set, read from your corpus
```

Two things that decide whether this works:

- `EnumerationProbeEngine.can_handle` returns `document.structured`, so **mark the enumerable documents
  `structured=True`** or the engine receives `[]` and degenerates to hookless probes.
- find a *structural* source for the set — consistent headings, a filename convention, a closed table.
  Reading `## ` headings out of product pages gives a defensible complete list; scraping prose does not.

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

| `transform` | Implementation | Premise for `Cuenta Digital Libre` |
|---|---|---|
| `mutate_to_fake` | appends `-2` | `Cuenta Digital Libre-2` |
| `flip_value` | increments **every** digit run | unchanged — no digits |
| `flip_fact` | prepends the literal `"not "` | `not Cuenta Digital Libre` |
| `keep_real` | identity | `Cuenta Digital Libre` |

Read that table against your own entities before choosing, because two of the four have narrow ranges of
applicability and neither says so when it misses:

- **`flip_value` only behaves on an entity with exactly one digit run.** `POLICY-1` → `POLICY-2`, as
  intended. But `RD$500,000` → `RD$501,1`, because it increments `500` *and* `000` separately. And on an
  entity with no digits it returns the entity **unchanged**, which the engine then labels documented —
  so the strategy silently becomes a second control and charges nothing. If your values carry thousands
  separators or currency symbols, this transform is not for them.
- **`flip_fact` writes English.** `"not "` is hardcoded, so on a Spanish or Portuguese corpus the premise
  is `"not Cuenta Digital Libre"` — a string no user would type, which makes the probe test whether the
  assistant tolerates broken language rather than whether it accepts a false premise.

The registry is read-only (`MappingProxyType`), so you cannot supply a replacement. If neither fits your
entities, say so in the catalogue and leave the principle to be charged by the transforms that do fit —
the Profiler grades **every** principle on **every** response, so a principle with no strategy of its own
is still measured. Asking about an entity that does not exist is already a false premise.

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
