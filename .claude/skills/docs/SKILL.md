---
name: docs
description: Create or update a metric documentation page in docs/metrics/ following the Gaussia .mdx pattern.
argument-hint: <metric-name>
---

# Gaussia Metric Docs Generator

Create or update `docs/metrics/<metric-name>.mdx` following the established pattern.

## Which template to follow

There are two doc tiers in this repo:

- **Full docs** — context.mdx, agentic.mdx, conversational.mdx, toxicity.mdx
- **Minimal docs** — best-of.mdx, bias.mdx, humanity.mdx, regulatory.mdx, vision.mdx

For **RoleAdherence** use the **full docs template**. It is an LLM-judge metric with statistical modes — same tier as Context/Agentic/Conversational.

## Full docs structure (in order)

Read `docs/metrics/context.mdx` as the canonical reference for section order and MDX component syntax.

1. **Frontmatter** — `title` and `description`
2. **H1 + intro paragraph** — what the metric measures, how it aggregates (session-level), what `turns` preserves
3. **Overview** — bullet list: key score, session aggregate, per-turn detail, Bayesian mode
4. **Installation** — `uv add gaussia` + LLM provider
5. **Basic Usage** — `<CodeGroup>` with Frequentist (default) and Bayesian tabs
6. **Required Parameters table** — columns: Parameter, Type, Description
7. **Optional Parameters table** — columns: Parameter, Type, Default, Description
8. **Statistical Modes** — `<Tabs>` with Frequentist and Bayesian showing which CI fields are `None`
9. **[Metric-specific section]** — e.g. Interaction Weights (copy from context.mdx verbatim for weight logic)
10. **Output Schema** — show `class RoleAdherenceMetric(BaseMetric)` and `class RoleAdherenceTurn(BaseModel)` with field types; add score interpretation table
11. **Complete Example** — self-contained runnable snippet with a realistic `Retriever` subclass
12. **LLM Provider Options** — `<CodeGroup>` with OpenAI, Groq, Anthropic, Ollama tabs (copy from context.mdx)
13. **Best Practices** — `<AccordionGroup>` with 3–4 metric-specific tips
14. **Next Steps** — `<CardGroup cols={3}>` linking to related metrics/concepts

## RoleAdherence specifics

### Instantiation pattern

RoleAdherence uses `LLMJudgeStrategy` explicitly — it does **not** accept `model=` directly:

```python
from gaussia.metrics.role_adherence import RoleAdherence, LLMJudgeStrategy
from langchain_openai import ChatOpenAI

judge_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
strategy = LLMJudgeStrategy(model=judge_model, binary=True)

metrics = RoleAdherence.run(
    MyRetriever,
    scoring_strategy=strategy,
    binary=True,
    strict_mode=False,
    threshold=0.5,
    include_reason=False,
    verbose=True,
)
```

### Dataset requirement

`Dataset` objects **must** include `chatbot_role: str` — the role definition string `R`. This field is unique to RoleAdherence; no other metric requires it.

### Key parameters to document

**`LLMJudgeStrategy` constructor:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseChatModel` | required | LangChain-compatible judge model |
| `binary` | `bool` | `True` | Binary (0/1) vs continuous [0,1] scoring |
| `use_structured_output` | `bool` | `False` | Use LangChain structured output |
| `strict` | `bool` | `True` | Strict mode for structured output parsing |
| `bos_json_clause` | `str` | \`\`\`json | JSON block start marker |
| `eos_json_clause` | `str` | \`\`\` | JSON block end marker |
| `verbose` | `bool` | `False` | Enable verbose logging |

**`RoleAdherence.run()` / constructor:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `retriever` | `Type[Retriever]` | required | Data source class |
| `scoring_strategy` | `ScoringStrategy` | required | Strategy for per-turn scoring (e.g. LLMJudgeStrategy) |
| `statistical_mode` | `StatisticalMode` | `FrequentistMode()` | Statistical computation mode |
| `binary` | `bool` | `True` | If True, per-turn scores binarized at threshold |
| `strict_mode` | `bool` | `False` | If True, session adherent only if ALL turns pass |
| `threshold` | `float` | `0.5` | Score cutoff for binary classification |
| `include_reason` | `bool` | `False` | Include judge reasoning in per-turn output |

### Output schema

```python
class RoleAdherenceMetric(BaseMetric):
    session_id: str
    assistant_id: str
    n_turns: int
    role_adherence: float               # Weighted mean (0.0–1.0)
    role_adherence_ci_low: float | None  # Bayesian only
    role_adherence_ci_high: float | None # Bayesian only
    adherent: bool                       # Session-level pass/fail
    turns: list[RoleAdherenceTurn]

class RoleAdherenceTurn(BaseModel):
    qa_id: str
    adherence_score: float   # Per-turn score (0.0–1.0)
    adherent: bool           # Turn-level pass/fail
    reason: str | None       # Judge reasoning — populated when include_reason=True
```

Session `adherent` logic:
- `strict_mode=False` (default): `role_adherence >= threshold`
- `strict_mode=True`: all turns must be individually adherent

### Rules for content

- `STREAM_BATCHES` iteration level is **not** supported — add a `<Note>` warning in the Best Practices section
- Use a FinTrack-style banking chatbot for the complete example (consistent with the existing dataset in `examples/role_adherence/jupyter/`)
- Include both an adherent session and a violations session to show the `adherent: bool` contrast
- Score interpretation table must cover five 0.0–1.0 bands (same as context.mdx)
- The `scoring_strategy` param is in the Required Parameters table; `binary`/`strict_mode`/`threshold`/`include_reason` go in Optional
- `LLMJudgeStrategy` parameters should appear in their own sub-table under Basic Usage, before the main Optional Parameters table

## After writing

Run `ls docs/metrics/` to confirm the file was created.
