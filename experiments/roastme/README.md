# Roast Me — Profiler (levels 1 and 2)

The Probe Library and the Profiler: generates **probes** (red-teaming trap questions) from a
Knowledge Base, sends them to a real assistant, and measures whether the assistant **falls**
for the trap or **resists** — aggregating the verdicts into a weakness profile θ=(ω,H).

Level 3 (the Exploiter, which consumes that profile) is a separate project in
[`exploiter/`](exploiter/); see [`HANDOFF-profiler-exploiter.md`](HANDOFF-profiler-exploiter.md)
for what crosses between the two.

## Where the rest of this lives

This experiment is split across three repositories on purpose:

| What | Where |
|---|---|
| Code, notebooks and frozen results (this directory) | `gaussia-labs/pygaussia` |
| The paper (`roastme.tex`/`.pdf`) and a copy of the results | `gaussia-labs/papers`, `papers/2026-06-roastme/` |
| The curated plugin/strategy catalogue for the reported scenario | `Alquimia-ai/roast-me` |

`config/plugins.yaml` and `config/strategies.yaml` are therefore **not** in this repository —
they are scenario-specific. Copy the shipped examples to get running:

```bash
cp config/plugins.example.yaml config/plugins.yaml
cp config/strategies.example.yaml config/strategies.yaml
```

The examples keep the same schema and the same `id`s as the catalogue that produced
`results/` — the ids are recorded in every graded probe. Note that `entity_kind` values like
`articulo` are tied to the sample extractor in `src/kb.py`, so the examples are the same
structure with the scenario wording removed, not a domain-free config.

## To see the results: open the notebooks, they already have everything inside

No API keys needed, and nothing gets called over the network. The notebooks use the plain
`python3` kernel:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Install the full `requirements.txt` even just to read them: all six notebooks import `pandas`
and `PyYAML`. `gaussia` is imported *lazily*, inside the retrieval engines' functions, so
every module still loads without it — you only hit the missing dependency if you actually
execute a cell that runs the `rag`/`graphrag`/`grag` engines live. Reading the frozen results
never does.

Open in this order:

1. **`jupyter/roastme.ipynb`** — Level 1: how probes are generated, the engines
   compared, the canonical dataset of 204 probes.
2. **`jupyter/compare_models.ipynb`** — Level 1b: how each LLM family generates
   probes (gemma / z.ai / kimi).
3. **`jupyter/profiler.ipynb`** — Level 2: the real agent evaluated, weakness
   profile per judge, ranking stability across iterations.
4. **`jupyter/roastme_promptfoo.ipynb`** — generic risk plugins + evasion
   (inspired by promptfoo — scaffold/exploratory, see the warning below).

Two more, outside the tour:

- **`jupyter/demo_roastme_with_outputs.ipynb`** — walkthrough of the whole metric with every
  output already saved, so it reads end to end without executing a cell. Includes a
  "what we still don't know" section.
- **`jupyter/demo_roastme.ipynb`** — the runnable variant of that walkthrough (outputs
  stripped). Neither is a subset of the other; they overlap on most cells.

All of them read data already frozen in `results/` and show real results on open —
regenerating live is optional (last section).

## Setup (only needed to run things, not to read them)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env.example` documents every variable and, more useful, **which ones each level actually
needs**. The short version:

| What you want to run | What you need |
|---|---|
| Read the notebooks | nothing |
| Level 2, re-judging the frozen transcripts (cheapest live path) | `HF_TOKEN` |
| Level 1 probe generation, `rag`/`graphrag`/`grag` engines | `GROQ_API_KEY` + the `gaussia` library |
| Level 2 against a live assistant (re-freezing transcripts) | the `TARGET_*` block |

**About the `gaussia` dependency.** The only symbol imported from the library is
`gaussia.embedders.SentenceTransformerEmbedder` (in `engines_rag.py`, `engines_grag.py`,
`compare_models.py`), and only the retrieval-based engines use it. `pygaussia` is a private
repo, so the pinned git URL in `requirements.txt` needs credentials — if you have it checked
out locally, `pip install -e /path/to/pygaussia` is simpler. Everything else here (the
`deterministic` engine, the whole Profiler, all the notebooks) runs without it.

The pinned versions are the ones the reported runs used; `requirements.txt` explains the one
judgement call in the pin (the library was installed editable, 13 commits past its tag).

## The idea: how the knowledge boundary is known

Each probe needs a label (exists / doesn't exist). How reliable that label is depends
on how the KB's knowledge boundary is known. Three engines behind the
same contract, which **compose** (you don't pick just one):

| Engine | How it knows the boundary | Absence (reliable label) | Own contribution |
|---|---|---|---|
| `deterministic` | extractor enumerates the KB | yes (baseline, perfect label) | absence control |
| `rag` | retrieves chunks via embeddings | no | maximum false-premise coverage |
| `graphrag` | FULL graph as enumeration | yes (retrieves it) | absence without a hand-built extractor |
| `grag` | retrieves a textual SUBGRAPH (GRAG paper, Hu et al. NAACL 2025) | no (doc=1) | MULTI-HOP false premise |

## Tangible result — Level 1 (compose over Ley 24.977)

| engine | probes | absence | acc. absence | false premise |
|---|---|---|---|---|
| deterministic | 65 | 7 | 1.00 | 58 |
| graphrag | 40 | 8 | 1.00 | 32 |
| rag | 99 | 10 | 0.00 | 89 |

Only the deterministic engine and GraphRAG get absence right (GraphRAG **retrieves** it
with the graph); RAG invents articles that do exist (acc 0.00) but delivers the widest
false-premise coverage. That's the trade off, measured on the same KB.

## Tangible result — multi-model generation comparison

| model | sec/probe | reading |
|---|---|---|
| `google/gemma-4-31B-it` | ~1.1s | workhorse: fast, cheap, good quality |
| `zai-org/GLM-5.2` (reasoning) | ~59.6s | more sophisticated multi-hop traps; ~54x slower |
| `moonshotai/Kimi-K2.6` (reasoning) | ~103.6s | the most elaborate (chains 2+ real facts); ~94x slower |

## Result — Level 2 (Profiler: real agent evaluated)

An **LLM judge** decides for each agent response whether it **fell** for the trap or
**resisted**, and aggregates the verdicts into an *assistant profile* (where it's most
likely to fall + which KB entities broke it).

These are the numbers the paper reports. The run covers the **full dataset** — 204 probes,
of which **151 are scoreable** (the other 53 are controls, excluded from every
violation-rate aggregate; see `grade()` in `src/profiler.py`) — with the three judges of
`src/judge.py::DEFAULT_JUDGES` under the same configuration:

| Judge | Method | Fail rate (151 probes) |
|---|---|---|
| `hf_router:google/gemma-4-31B-it` | logprobs | 0.1258 |
| `hf_router:zai-org/GLM-5.2` | logprobs | 0.1523 |
| `hf_router:Qwen/Qwen3.6-35B-A3B:scaleway` | logprobs | 0.0795 |

Judges evaluated and then dropped (Kimi-K2.6, groq-llama-3.3-70b) are kept in
`results/level2_profiler/archive/`, and the evidence for the roster choice is in
`results/level2_profiler/candidate_judges_test*.json` and the `*_logprobs_verification.json`
files.

> **Standing caveat.** The judges were never audited against human labels. They agree with
> each other, but agreement between LLM judges is not the same as being right. The verdicts
> are stochastic, which is why the profile is built over several iterations and
> `weakness_evolution_ley_compose.json` records the ranking stability per iteration.

## Result — generic risk plugins + evasion (inspired by promptfoo)

We investigated promptfoo (`plugins` = risk category, `strategies` = delivery
disguise) to see what it adds to the metric in general.

> **Warning: scaffold/exploratory.** `GenericRiskEngine.generate_proposed()` generates
> probes via LLM (none hand-written) for 7 KB-entity-free plugins (prompt-extraction,
> excessive-agency, hallucination, etc. — 4 remain documented as `scaffolded`, not
> implemented), optionally anchored to a description of the agent under test. The
> evasion layer (`base64`/`rot13`)
> found something real: the response provider (`claude-opus-4-8`) blocks with
> `content_filter` the "decode this and answer" pattern — 8/8 probes blocked in
> both strategies, before the assistant's logic even comes into play. The
> multi-turn scaffold (`ConversationSession`) proves the session persists across turns, but
> does NOT implement a real escalation strategy (it's not Crescendo). See
> `jupyter/roastme_promptfoo.ipynb` for the detail with real data.

## Methodological integrity

The oracle (exact enumeration of the KB) is used **only for scoring**. It's never passed
to the LLM engine or to the verifier, which works via retrieval (RAG) or graph
membership (GraphRAG).

## Structure

```
experiments/
├── README.md                       # this file
├── HANDOFF-profiler-exploiter.md   # what level 3 consumes + the cheap re-run path
├── requirements.txt                # pinned to the versions the reported runs used
├── .env.example                    # every variable, and who needs which
├── exploiter/                      # level 3 — separate project, own pyproject.toml
├── jupyter/                        # 6 notebooks — the tour starts here
├── results/                        # frozen artifacts (83 files; see results/README.md)
├── src/                            # all the code (plain scripts, run from HERE)
├── data/                           # sample KBs (Ley 24.977 = 58 files, FAQ Aurora)
│                                   #   ley_24977/ is the SINGLE copy; exploiter/ reads it too
├── config/                         # plugins/strategies (KB) + generic_plugins/evasion
└── promptfoo_research/             # promptfoo research notes (not code)
```

Everything in `src/` is a plain script meant to be invoked **from this directory**
(`python src/run_profiler.py ...`), not installed as a package. Paths inside are relative to
here, which is why running from `src/` or from `results/` breaks them.

## Optional: regenerate from scratch

Only needed if you want to run something live. Copy `.env.example` to `.env` and fill it
in — it documents every variable and which ones each level actually needs. The cheapest
live path by far is **re-judging the frozen transcripts** (`HF_TOKEN` only, never touches
the target assistant); see `HANDOFF-profiler-exploiter.md`.

Everything is invoked **from this directory** (`experiments/`):

```bash
PY=.venv/bin/python     # or just `python`, with the venv activated

# Level 1: generate the canonical dataset (the 3 composed engines)
$PY src/universal_probe_library.py --kb ley --engine compose

# Multi-model generation comparison
$PY src/compare_models.py --engine grag \
   --models "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"

# Direct chat with the target agent (bypassing the harness)
$PY src/target_client.py --chat

# Level 2: profile the agent with the reported 3-judge roster. That roster is already
# the default (src/judge.py::DEFAULT_JUDGES), so --judges is only needed to override it.
# With transcripts_ley_compose.json present this runs in fixed_transcripts mode: it
# re-judges the frozen responses and never calls the target, so HF_TOKEN is enough.
# The 53 control probes (plugin=None) are excluded from judging -> 151 scoreable of 204.
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
   --iterations 5

# Judges that were evaluated and dropped (Kimi-K2.6, groq-llama-3.3-70b) are kept in
# results/level2_profiler/archive/ — pass them via --judges to reproduce that comparison.

# Generic risk plugins (promptfoo) — no KB, templates or paraphrasing split
# across models (gemma does most of it, GLM/Kimi only a couple, they're slow reasoners)
$PY src/generic_probe_library.py --model "google/gemma-4-31B-it" --context "..."

# Evasion: wrap an already-generated dataset before profiling (writes to
# results/level2_profiler_evasion/, not results/level2_profiler/)
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
   --evasion base64 --judges "groq:llama-3.3-70b-versatile" --iterations 1 --limit 8

# Multi-turn scaffold (plumbing test, not a real strategy)
$PY src/run_multiturn_demo.py --limit 5 --turns 2
```

Each script has more parameters (`--help` lists them); `src/universal_probe_library.py`
supports a single engine, a custom KB, quantity/seed control, etc. — no need
to memorize them, they're in each file's docstring.

## Files (`src/`)

- `contract.py` — durable contract (Document, KnowledgeHook, Probe, ProbeEngine).
- `probe_library.py` — DeterministicEngine + composition (merge/dedup) + config/doc loading.
- `engines_rag.py`, `engines_graphrag.py`, `engines_grag.py` — the 3 generation engines.
- `kb.py`, `oracle.py` — example extractor for Ley 24.977 + honest scoring.
- `config.py` — LLM provider registry (Groq, HuggingFace) + logprobs + target credentials.
- `judge.py`, `profiler.py`, `run_profiler.py` — the Profiler (Level 2).
- `target_client.py` — client for the target assistant (Alquimia runtime, SSE; `--chat` to talk to it directly; `ConversationSession` for the multi-turn scaffold).
- `universal_probe_library.py`, `compare_models.py` — generation entrypoints (Level 1 and 1b).
- `engines_generic.py`, `generic_probe_library.py` — generic risk plugins (promptfoo, no KB).
- `evasion.py` — evasion layer (Base64/ROT13 implemented; see `config/evasion_strategies.yaml`).
- `run_multiturn_demo.py` — multi-turn scaffold (plumbing, not a real attack).
- `export_profile_for_exploiter.py` — **the bridge to level 3**: translates a
  `profile_ley_compose_*.json` into the `AssistantProfile` schema the Exploiter expects. Not
  just field renaming — read its docstring before touching it, in particular why the
  descriptors it writes must stay in Spanish (the Exploiter grounds Cθ by token overlap
  against Spanish text, so translating them breaks it silently).
- `merge_profiler_run.py` — merges a profiler run into an existing artifact.
- `verify_glm_logprobs.py`, `test_new_candidates.py` — one-off probes of whether a given
  model:provider combo exposes usable logprobs. These produced the
  `*_logprobs_verification.json` and `candidate_judges_test*.json` files that justify the
  judge roster.
