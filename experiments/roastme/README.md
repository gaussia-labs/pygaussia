# Roast Me — experiment code

The code behind the Roast Me paper: a **Probe Library** that generates red-teaming trap
questions from a knowledge base (level 1), a **Profiler** that judges the target assistant's
answers and builds a weakness profile θ=(ω,H) (level 2), and an **Exploiter** that searches for
reproducible failure categories (level 3, in [`exploiter/`](exploiter/)).

## Where the rest lives

| What | Where |
|---|---|
| Code (this directory) | `gaussia-labs/pygaussia` |
| The paper and the frozen results behind every reported number | `gaussia-labs/papers`, `papers/2026-06-roastme/` |
| The curated plugin/strategy catalogue for the reported scenario | `Alquimia-ai/roast-me`, `catalog/` |

This directory holds **no results**. `results/` is gitignored and is only where your own runs
land. To work against the reported artifacts, copy them in from the papers repo:

```bash
git clone https://github.com/gaussia-labs/papers /tmp/papers
cp -R /tmp/papers/papers/2026-06-roastme/results .
```

That matters for one thing in particular: with
`results/level2_profiler/transcripts_ley_compose.json` in place (the 204 frozen assistant
answers), the Profiler runs in `fixed_transcripts` mode — it re-grades without ever calling the
target assistant, so it needs only `HF_TOKEN` and no target credentials.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config/plugins.example.yaml config/plugins.yaml
cp config/strategies.example.yaml config/strategies.yaml
```

`.env.example` lists every variable and which level needs it. The short version:

| What you want to run | What you need |
|---|---|
| Level 2, re-grading the frozen transcripts (cheapest path) | `HF_TOKEN` |
| Level 1 probe generation, `rag`/`graphrag`/`grag` engines | `GROQ_API_KEY` + the `gaussia` library |
| Level 2 against a live assistant (re-freezing transcripts) | the `TARGET_*` block |

`config/plugins.yaml` and `config/strategies.yaml` are gitignored: the curated catalogue for the
reported scenario lives in `Alquimia-ai/roast-me`. The `.example.yaml` files keep the same schema
and the same `id`s (the ids are recorded in every graded probe, so renaming them decouples the
published results) with domain-neutral prose. Note that `entity_kind` values like `articulo` are
tied to the sample extractor in `src/kb.py`, so the examples are the same structure with the
scenario wording removed, not a domain-free config.

**About the `gaussia` dependency.** The only symbol imported from the library is
`gaussia.embedders.SentenceTransformerEmbedder` (in `engines_rag.py`, `engines_grag.py`,
`compare_models.py`), and it is imported lazily, inside those engines' functions. Every module
loads without it — you only hit the missing dependency if you execute a retrieval engine live.
`requirements.txt` pins `v1.0.0-b.3`, the tag the reported runs used.

## The four engines and their trade-off

Each probe needs a label (the entity exists / does not exist). How trustworthy that label is
depends on how the engine knows the knowledge base's boundary. Four engines behind one contract,
which **compose** — you do not pick just one:

| Engine | How it knows the boundary | Reliable absence label | Own contribution |
|---|---|---|---|
| `deterministic` | an extractor enumerates the KB | yes (perfect label) | absence control |
| `rag` | retrieves chunks via embeddings | no | widest false-premise coverage |
| `graphrag` | the FULL graph as enumeration | yes (it retrieves it) | absence with no hand-built extractor |
| `grag` | retrieves a textual SUBGRAPH (Hu et al., NAACL 2025) | no (doc=1) | MULTI-HOP false premise |

The measured trade-off, and every other reported figure, is in the paper — see
`papers/2026-06-roastme/roastme.tex` and `METRICAS-ROASTME.md`.

## Running it

Everything in `src/` is a plain script invoked **from this directory**, not an installed package.
Paths inside are relative to here, so running from `src/` breaks them.

```bash
PY=.venv/bin/python     # or just `python`, with the venv activated

# Level 2 — the reported 3-judge roster is already the default (src/judge.py::DEFAULT_JUDGES).
# With the frozen transcripts present this runs in fixed_transcripts mode and never calls the
# target. The 53 control probes (plugin=None) are excluded -> 151 scoreable of 204.
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json --iterations 5

# Level 1 — regenerating calls an LLM, so the probes will NOT come out identical to the
# published dataset (generation is stochastic).
$PY src/universal_probe_library.py --kb ley --engine compose

# Level 1b — how each model family generates probes
$PY src/compare_models.py --engine grag \
   --models "google/gemma-4-31B-it=12,zai-org/GLM-5.2=4,moonshotai/Kimi-K2.6=3"

# Generic risk plugins (promptfoo-inspired), no KB
$PY src/generic_probe_library.py --model "google/gemma-4-31B-it" --context "..."

# Evasion: wrap an already-generated dataset before profiling
$PY src/run_profiler.py --dataset results/level1_probes/dataset_ley_compose.json \
   --evasion base64 --judges "groq:llama-3.3-70b-versatile" --iterations 1 --limit 8

# Direct chat with the target assistant, bypassing the harness
$PY src/target_client.py --chat
```

Every script takes more flags than shown; `--help` lists them, and each file's docstring explains
the non-obvious ones.

## Structure

```
roastme/
├── requirements.txt                # pinned to the versions the reported runs used
├── .env.example                    # every variable, and which level needs it
├── src/                            # levels 1-2 (plain scripts, run from here)
├── exploiter/                      # level 3 — separate project, own pyproject.toml
├── data/                           # sample KBs (Ley 24.977 = 58 files, FAQ Aurora)
└── config/                         # generic plugins + evasion, and the two .example.yaml
```

## Files in `src/`

- `contract.py` — the durable contract (`Document`, `KnowledgeHook`, `Probe`, `ProbeEngine`).
- `probe_library.py` — the deterministic engine, composition (merge/dedup), config/doc loading.
- `engines_rag.py`, `engines_graphrag.py`, `engines_grag.py` — the three retrieval engines.
- `kb.py`, `oracle.py` — the sample extractor for Ley 24.977, and honest scoring.
- `config.py` — LLM provider registry (Groq, HuggingFace) + logprobs + target credentials.
- `judge.py`, `profiler.py`, `run_profiler.py` — the Profiler (level 2).
- `target_client.py` — client for the target assistant (SSE; `--chat` to talk to it directly,
  `ConversationSession` for the multi-turn scaffold).
- `universal_probe_library.py`, `compare_models.py` — generation entry points (levels 1 and 1b).
- `engines_generic.py`, `generic_probe_library.py` — generic risk plugins, no KB.
- `evasion.py` — the evasion layer (Base64/ROT13 implemented; see `config/evasion_strategies.yaml`).
- `run_multiturn_demo.py` — multi-turn scaffold (plumbing, not a real attack strategy).
- `export_profile_for_exploiter.py` — **the bridge to level 3**: translates a `profile_*.json`
  into the `AssistantProfile` schema `exploiter/` expects. Not just field renaming — read its
  docstring, in particular why the descriptors it writes must stay in Spanish (the Exploiter
  grounds Cθ by token overlap against Spanish text).
- `merge_profiler_run.py` — merges a profiler run into an existing artifact.
- `verify_glm_logprobs.py`, `test_new_candidates.py` — one-off probes of whether a given
  `model:provider` combo exposes usable logprobs. These produced the verification artifacts that
  justify the judge roster.

## A note on language

Judge rubrics (`judge.py`), generation prompts (`engines_*.py`) and `phrasing_hint` values stay in
**Spanish** on purpose: they are injected into model prompts and the knowledge base is in Spanish.
Translating them would change what the metric measures. Everything else is in English.

## Methodological integrity

The oracle (exact enumeration of the KB) is used **only for scoring**. It is never passed to the
LLM engine or to the verifier, which work via retrieval (RAG) or graph membership (GraphRAG).
