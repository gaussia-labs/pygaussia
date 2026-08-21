# Gaussia

[![PyPI version](https://img.shields.io/pypi/v/gaussia)](https://pypi.org/project/gaussia/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gaussia)](https://pypi.org/project/gaussia/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/gaussia)](https://pypi.org/project/gaussia/)
[![PyPI - License](https://img.shields.io/pypi/l/gaussia)](https://pypi.org/project/gaussia/)

AI evaluation framework for measuring fairness, quality, and safety of AI models and assistants.

## Installation

```bash
pip install gaussia
```

With specific metric dependencies:

```bash
pip install gaussia[toxicity]              # Toxicity analysis
pip install gaussia[bias]                  # Bias detection
pip install gaussia[privacy-presidio]      # Privacy with the Presidio detector
pip install gaussia[privacy-huggingface]   # Privacy with a HuggingFace NER detector
pip install gaussia[evalhub]               # EvalHub provider adapter
pip install gaussia[metrics]               # All metrics
pip install gaussia[all]                   # Everything
```

Roast Me ships its own extras, deliberately outside `metrics` and `all`, so whoever only profiles an
assistant pays for neither retrieval nor training:

```bash
pip install gaussia[roastme]      # The three corpus-reading probe engines
pip install gaussia[roastme-rl]   # The above, plus the reinforcement-learning search
```

Most metrics judge with a LangChain-compatible chat model, which you install separately:
`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-groq`,
`langchain-ollama`.

## Quick Start

```python
from gaussia import Retriever, Dataset, Batch
from gaussia.metrics.context import Context
from langchain_openai import ChatOpenAI

# 1. Define your data source
class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        return [
            Dataset(
                session_id="session-1",
                assistant_id="assistant-1",
                language="en",
                context="France is a country in Western Europe.",
                conversation=[
                    Batch(
                        qa_id="q1",
                        query="Where is France?",
                        assistant="France is located in Western Europe.",
                        ground_truth_assistant="France is a country in Western Europe.",
                    )
                ],
            )
        ]

# 2. Run a metric. `run` takes the retriever *class* and instantiates it for you.
metrics = Context.run(MyRetriever, model=ChatOpenAI(model="gpt-4o-mini", temperature=0.0))
```

Metrics are imported from their own modules — `gaussia.metrics` declares the surface but imports
nothing, so no metric drags in another's optional dependencies.

## Metrics

| Metric | Description | Install extra |
|--------|-------------|---------------|
| **Context** | Evaluates response alignment with provided context | — |
| **Conversational** | Dialogue quality via Grice's maxims (memory, language, quality, quantity, relation, manner) | — |
| **BestOf** | King-of-the-hill tournament comparison of multiple assistants | — |
| **Agentic** | Agent evaluation with pass@K and tool correctness | — |
| **RoleAdherence** | Whether the assistant stays inside its defined role, scored from judge first-token logprobs | — |
| **Toxicity** | Cluster-based toxicity profiling with demographic and sentiment analysis | `[toxicity]` |
| **Bias** | Bias detection across protected attributes using guardians | `[bias]` |
| **PIIDetectorBenchmark** | Domain-adjusted detection score rating one PII/PHI detector's fitness for a regulated domain | `[privacy-presidio]` / `[privacy-huggingface]` |
| **PIIDetectorRanker** | The same score across several detectors, ranked, to decide which one to ship | `[privacy-presidio]` / `[privacy-huggingface]` |
| **Humanity** | Emotion, empathy, and human-like quality analysis | `[humanity]` |
| **Regulatory** | Compliance evaluation against regulatory documents | `[regulatory]` |
| **VisionSimilarity** | VLM description comparison via semantic similarity | `[vision]` |
| **VisionHallucination** | Hallucination detection in VLM outputs | `[vision]` |

## Features

### Guardians

Pluggable bias detection backends. The guardian is passed as a class, like the retriever:

```python
from gaussia.guardians import IBMGranite, LLamaGuard
from gaussia.metrics.bias import Bias

metrics = Bias.run(MyRetriever, guardian=IBMGranite)
```

### PII Detectors

Pluggable detection backends behind the `PIIDetector` contract, the same way guardians sit behind
`Guardian`. A detector carries the expert `[0, 1]` scalars that place it in your domain, so it is
passed as an instance:

```python
from gaussia.detectors.presidio import PresidioDetector
from gaussia.metrics.privacy import PIIDetectorBenchmark
from gaussia.schemas.privacy import PrivacyDomainConfig

domain = PrivacyDomainConfig(
    classes=frozenset({"email_address", "phone_number"}),
    criticality_weights={"email_address": 0.6, "phone_number": 0.4},
    fn_severity_weights={"email_address": 0.7, "phone_number": 0.3},
    regulatory_framework="GDPR",
)
detector = PresidioDetector(name="presidio", domain_fit=0.9, regulatory_fit=0.8)

metrics = PIIDetectorBenchmark.run(MyRetriever, detector=detector, domain_config=domain)
```

`PIIDetectorRanker` takes `detectors=[...]` instead and ranks them under the same domain config.

### Role Adherence

Whether the assistant stayed in the role it was given, scored per turn and aggregated per session.
The judge derives a calibrated `[0, 1]` score from first-token logprobs, so it needs a provider that
exposes them — `StructuredOutputJudgeStrategy` is the fallback for providers that do not:

```python
from gaussia.metrics.role_adherence import RoleAdherence, LLMJudgeStrategy
from langchain_openai import ChatOpenAI

strategy = LLMJudgeStrategy(model=ChatOpenAI(model="gpt-4o-mini"))
metrics = RoleAdherence.run(MyRetriever, scoring_strategy=strategy)
```

### Statistical Modes

Choose between frequentist and Bayesian aggregation:

```python
from gaussia import FrequentistMode, BayesianMode
from gaussia.metrics.context import Context

metrics = Context.run(MyRetriever, model=judge, statistical_mode=FrequentistMode())
metrics = Context.run(MyRetriever, model=judge, statistical_mode=BayesianMode())
```

### Synthetic Data Generation

Generate evaluation datasets from documents:

```python
from gaussia.generators import BaseGenerator, create_markdown_loader
from langchain_openai import ChatOpenAI

generator = BaseGenerator(model=ChatOpenAI(model="gpt-4o-mini"))
loader = create_markdown_loader()

datasets = await generator.generate_dataset(
    context_loader=loader,
    source="./docs/knowledge_base.md",
    assistant_id="my-assistant",
)
```

### Roast Me

Adversarial evaluation as a search problem: profile an assistant's weaknesses from tagged probes,
then look for the *categories* of realistic question that break it reproducibly. A generator
subsystem, not a metric — nothing subclasses `Gaussia`. What enters the metric pipeline is the Roast
Dataset it emits, which existing metrics consume unchanged.

```python
from gaussia.generators.roastme import ProbeLibrary, Profiler, to_dataset

probes = ProbeLibrary(engines).generate(documents, catalogue)
result = Profiler(contract=contract, target=your_adapter).profile(probes)

dataset = to_dataset(
    probes,
    result.outcomes,
    session_id="roast-run-1",
    assistant_id="support-assistant",
    context="Roast Me run over the policy knowledge base",
)
```

> Every number Roast Me produces is a judge-only measurement: one language model's estimate of
> whether another one misbehaved. No grader here has been calibrated against human labels, so read a
> violation rate as evidence to look at, never as a measured error rate.

### Explainability

Token-level attribution analysis. The method is a class, not a string:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gaussia.explainability import AttributionExplainer, Lime

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

explainer = AttributionExplainer(model, tokenizer)
result = explainer.explain(
    prompt=tokenizer.apply_chat_template([{"role": "user", "content": "What is gravity?"}], tokenize=False),
    target="Gravity is the force of attraction between objects.",
    method=Lime,
)
print(result.get_top_k(5))
```

### Prompt Optimization

Optimize prompts using evolutionary and multi-objective strategies:

```python
from gaussia.prompt_optimizer import GEPAOptimizer, MIPROv2Optimizer
```

### EvalHub Provider

Run Gaussia as an EvalHub BYOF provider:

```bash
python -m gaussia.integrations.evalhub.adapter
```

## Documentation

Full documentation available at [docs.gaussia.ai](https://docs.gaussia.ai).

## Requirements

- Python >= 3.11

## License

MIT
