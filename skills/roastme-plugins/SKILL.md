---
name: roastme-plugins
description: Author the plugin entries of a Roast Me catalogue — the risk families that map onto the principles of a behavioral contract. Use when setting Roast Me up for a domain, or when validate_catalogue reports that plugins name principles the contract does not carry.
argument-hint: <domain, or the path to your contract>
---

# Roast Me plugins

A **plugin** is a risk family: one kind of thing that can go wrong. It carries its own identifier and
it names the single principle of the contract it attacks.

Get this right before anything else, because everything downstream inherits it:

```python
PluginSpec(
    id="plugin-invention",        # the plugin's own identifier
    name="Invented entity",
    description="Questions leaning on an entity the knowledge base does not contain.",
    principle="no_invention",     # an identifier that must already exist in your contract
)
```

**A plugin is not a principle.** They are two identifiers and putting the principle's id in `id`
is the most common way to get this wrong. Nothing in the library will stop you: `Probe.plugin` is
only ever read for whether it is empty, so the run completes and the mistake surfaces later, when
your catalogue no longer lines up with anyone else's.

## 1. Start from the contract, never from the plugins

The contract is the input. Read it and list its principle identifiers:

```python
print([principle.id for principle in contract.principles])
```

Every `PluginSpec.principle` must be one of those strings. A plugin naming a principle the contract
does not carry is rejected by `validate_catalogue` with *"plugins name principles the contract does
not carry"*.

If you do not have a contract yet, write it first: it is a set of `Principle` objects whose weights
sum to `1.0`, each with a rubric and a grader. `roastme-profiler` covers it.

## 2. One plugin per way of going wrong, not one per principle

A principle can be attacked several ways, and each way is its own family. A contract with two
principles can have four plugins. What it cannot have is one plugin covering two principles — the
field takes one identifier, because a probe charges one principle at a time.

Name families after **what the attack is**, not after the domain object:

| Good | Why |
|---|---|
| `plugin-invention` | names the failure: the assistant invents something |
| `plugin-overreach` | names the failure: it promises what it cannot |
| `plugin-stale-fact` | names the failure: it answers from superseded knowledge |

Avoid `plugin-pricing` or `plugin-bookings`: those are topics, and topics belong in the strategies'
descriptions, where they become attributes the search can conjoin.

## 3. `description` is for a reader, unlike the strategies'

A plugin's `description` is documentation. A **strategy's** `description` is not — its clauses become
the probe's attributes. Do not carry that habit over: write the plugin's description as a sentence.

## 4. Verify

Plugins are validated together with the strategies, against the contract and your engines:

```python
from gaussia.generators.roastme.probes.catalogue import validate_catalogue

validate_catalogue(catalogue, contract, engines)   # raises, or returns silently
```

The two failures that concern plugins:

- `duplicate plugin identifiers: [...]` — two plugins share an `id`;
- `plugins name principles the contract does not carry: [...]` — a `principle` that is not in the
  contract, usually a typo or a principle that was renamed.

And one check the library cannot do for you: confirm each plugin's `principle` is the one you meant.
Print the mapping and read it:

```python
from gaussia.generators.roastme.probes.particularisation import principle_by_plugin

for plugin, principle in principle_by_plugin(catalogue).items():
    print(f"{plugin} attacks {principle}")
```

If a line reads `grounding attacks grounding`, you have put a principle id in `id`.

Then continue with `roastme-strategies`, which is where the attacks themselves are written.
