# Roast Me skills

Four skills for the pieces a user has to author to run Roast Me against their own assistant. Each one
covers a stage and states the rules the library will enforce, so a mistake surfaces at validation
rather than in the middle of a run that costs target calls.

| Skill | Covers |
|---|---|
| `roastme-plugins` | the catalogue's risk families, and the principles of the contract they attack |
| `roastme-strategies` | the catalogue's interaction patterns, the four transforms, and the controls |
| `roastme-profiler` | the contract, the target adapter, the grader, and reading the weakness profile |
| `roastme-exploiter` | the four collaborators, the thresholds, the call budget, and the failure report |

Run them in that order for a first setup. `roastme-profiler` and `roastme-exploiter` are also the two
to reach for when a run comes back empty: each ends with the failures to work through, in order.

## Using them

Claude Code discovers skills in a `.claude/skills/` directory. This folder is the source, versioned
alongside the code the skills describe — copy the ones you want into your project:

```bash
mkdir -p .claude/skills
cp -r path/to/pygaussia/skills/roastme-* .claude/skills/
```

Then `/roastme-plugins`, `/roastme-strategies`, and so on.

## Why they live here rather than in a marketplace

Their content is tied to the version of the SDK: which fields `PluginSpec` carries, which four
transforms exist, what `validate_catalogue` checks. Kept next to that code, they move with it. Kept in
a separate repository they would drift silently and teach a schema that no longer exists — which is a
mistake this project has already made once, inside its own notebook.
