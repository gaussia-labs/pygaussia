# Gaussia

AI evaluation framework for measuring fairness, quality, and safety of AI models and assistants.

## Installation

```bash
pip install gaussia
```

With specific metric dependencies:

```bash
pip install gaussia[toxicity]    # Toxicity analysis
pip install gaussia[bias]        # Bias detection
pip install gaussia[metrics]     # All metrics
pip install gaussia[all]         # Everything
```

## Quick Start

```python
from pygaussia import Retriever, Dataset, Batch
from pygaussia.metrics import Context

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

# 2. Run a metric
metrics = Context.run(retriever=MyRetriever())
```

## Metrics

| Metric | Description | Install extra |
|--------|-------------|---------------|
| **Context** | Evaluates response alignment with provided context | — |
| **Conversational** | Dialogue quality via Grice's maxims (memory, language, quality, quantity, relation, manner) | — |
| **BestOf** | King-of-the-hill tournament comparison of multiple assistants | — |
| **Agentic** | Agent evaluation with pass@K and tool correctness | — |
| **Toxicity** | Cluster-based toxicity profiling with demographic and sentiment analysis | `[toxicity]` |
| **Bias** | Bias detection across protected attributes using guardians | `[bias]` |
| **Humanity** | Emotion, empathy, and human-like quality analysis | `[humanity]` |
| **Regulatory** | Compliance evaluation against regulatory documents | `[regulatory]` |
| **VisionSimilarity** | VLM description comparison via semantic similarity | `[vision]` |
| **VisionHallucination** | Hallucination detection in VLM outputs | `[vision]` |

## Features

### Guardians

Pluggable bias detection backends:

```python
from pygaussia.guardians import IBMGraniteGuardian, LLamaGuardGuardian

metrics = Bias.run(retriever=MyRetriever(), guardian=IBMGraniteGuardian())
```

### Statistical Modes

Choose between frequentist and Bayesian aggregation:

```python
from pygaussia import FrequentistMode, BayesianMode

metrics = Context.run(retriever=MyRetriever(), statistical_mode=FrequentistMode())
metrics = Context.run(retriever=MyRetriever(), statistical_mode=BayesianMode())
```

### Synthetic Data Generation

Generate evaluation datasets from documents:

```python
from pygaussia.generators import BaseGenerator, create_markdown_loader

loader = create_markdown_loader(path="./docs")
generator = BaseGenerator(context_loader=loader)
datasets = generator.generate()
```

### Explainability

Token-level attribution analysis:

```python
from pygaussia.explainability import AttributionExplainer

explainer = AttributionExplainer(method="lime")
attributions = explainer.explain(text="Your input text")
```

### Prompt Optimization

Optimize prompts using evolutionary and multi-objective strategies:

```python
from pygaussia.prompt_optimizer import GEPAOptimizer, MIPROv2Optimizer
```

## Documentation

Full documentation available at [docs.gaussia.ai](https://docs.gaussia.ai).

## Requirements

- Python >= 3.11

## License

MIT
