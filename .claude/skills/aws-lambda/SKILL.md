---
name: aws-lambda
description: Create the aws-lambda example for a Gaussia metric under examples/<metric-name>/aws-lambda/ following the established pattern.
argument-hint: <metric-name>
---

# Gaussia AWS Lambda Example Generator

Create `examples/<metric-name>/aws-lambda/` with all required files.

## Reference implementation

`examples/bestof/aws-lambda/` is the canonical reference. Read every file there before writing. For the Dockerfile, prefer `examples/agentic/aws-lambda/Dockerfile` — it has better inline comments.

## Files to create

```
examples/<metric-name>/aws-lambda/
├── handler.py
├── run.py
├── requirements.txt
├── Dockerfile
├── README.md
└── scripts/
    ├── deploy.sh
    ├── update.sh
    └── cleanup.sh
```

## File-by-file instructions

### handler.py

Copy `examples/bestof/aws-lambda/handler.py` verbatim, changing only the module docstring first line:
```python
"""AWS Lambda handler for Gaussia <MetricName> metric."""
```
Everything else is identical across all Lambda examples.

### run.py

This is the metric-specific file. Follow the structure of `examples/bestof/aws-lambda/run.py`:

1. Module docstring explaining what the metric does
2. `create_llm_connector(connector_config)` — copy verbatim from bestof (identical across all examples)
3. `PayloadRetriever(Retriever)` — adapt for the metric:
   - For RoleAdherence: `Dataset.model_validate(data)` includes `chatbot_role` — it's part of the Dataset schema, no extra handling needed
4. `run(payload) -> dict` — adapt for the metric's instantiation pattern

**For RoleAdherence specifically:**

```python
"""Gaussia RoleAdherence Lambda business logic.

Evaluates whether an AI assistant adheres to its defined role across conversation turns.
"""

import importlib
import os
from typing import Any

from gaussia.core import Retriever
from gaussia.metrics.role_adherence import LLMJudgeStrategy, RoleAdherence
from gaussia.schemas import Dataset


# create_llm_connector — copy verbatim from bestof/run.py


class PayloadRetriever(Retriever):
    """Load datasets from Lambda payload."""

    def __init__(self, payload: dict):
        self.payload = payload

    def load_dataset(self) -> list[Dataset]:
        datasets = []
        for data in self.payload.get("datasets", []):
            datasets.append(Dataset.model_validate(data))
        return datasets


def run(payload: dict) -> dict[str, Any]:
    """Run RoleAdherence evaluation on payload datasets."""
    connector_config = payload.get("connector", {})
    if not connector_config:
        return {"success": False, "error": "No connector configuration provided"}

    try:
        model = create_llm_connector(connector_config)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to create LLM connector: {e}"}

    datasets = payload.get("datasets", [])
    if not datasets:
        return {"success": False, "error": "No datasets provided"}

    config = payload.get("config", {})

    strategy = LLMJudgeStrategy(
        model=model,
        binary=config.get("binary", True),
        use_structured_output=config.get("use_structured_output", True),
        verbose=config.get("verbose", False),
    )

    try:
        metrics = RoleAdherence.run(
            lambda: PayloadRetriever(payload),
            scoring_strategy=strategy,
            binary=config.get("binary", True),
            strict_mode=config.get("strict_mode", False),
            threshold=config.get("threshold", 0.5),
            include_reason=config.get("include_reason", False),
            verbose=config.get("verbose", False),
        )
    except Exception as e:
        return {"success": False, "error": f"RoleAdherence evaluation failed: {e}"}

    if not metrics:
        return {"success": False, "error": "No metrics produced"}

    return {
        "success": True,
        "results": [
            {
                "session_id": m.session_id,
                "assistant_id": m.assistant_id,
                "n_turns": m.n_turns,
                "role_adherence": m.role_adherence,
                "role_adherence_ci_low": m.role_adherence_ci_low,
                "role_adherence_ci_high": m.role_adherence_ci_high,
                "adherent": m.adherent,
                "turns": [
                    {
                        "qa_id": t.qa_id,
                        "adherence_score": t.adherence_score,
                        "adherent": t.adherent,
                        "reason": t.reason,
                    }
                    for t in m.turns
                ],
            }
            for m in metrics
        ],
    }
```

### requirements.txt

Copy `examples/bestof/aws-lambda/requirements.txt` verbatim. RoleAdherence uses the same LLM providers.

### Dockerfile

Base on `examples/agentic/aws-lambda/Dockerfile`. Change only:
- The build arg comment at the top: `# Build command: docker build -t gaussia-role-adherence -f examples/role_adherence/aws-lambda/Dockerfile --build-arg MODULE_EXTRA=role_adherence .`
- `ARG MODULE_EXTRA=role_adherence`
- The COPY paths: `examples/role_adherence/aws-lambda/requirements.txt` and `examples/role_adherence/aws-lambda/handler.py examples/role_adherence/aws-lambda/run.py`

Check `pyproject.toml` for the correct extra name for role_adherence (it may be `role-adherence` with a hyphen — use whatever is defined there).

### README.md

Base on `examples/bestof/aws-lambda/README.md`. Adapt:
- Title and description for RoleAdherence
- Test example curl payload using `chatbot_role` field, with a FinTrack-style banking assistant dataset (3–4 turns)
- Request format section reflecting RoleAdherence fields: `chatbot_role` in each dataset, `config.binary`, `config.strict_mode`, `config.threshold`, `config.include_reason`
- Response format showing `results[]` with `role_adherence`, `adherent`, `turns[]`
- Module-specific fields table replacing BestOf fields
- Error table with RoleAdherence-specific errors
- Deploy command: `./scripts/deploy.sh role_adherence us-east-2`
- Log tail command: `aws logs tail "/aws/lambda/gaussia-role-adherence" --follow --region us-east-2`

### scripts/

Copy `examples/bestof/aws-lambda/scripts/deploy.sh`, `update.sh`, and `cleanup.sh` verbatim — the scripts are generic and identical across all examples.

## Verification

After creating all files:
1. Run `find examples/role_adherence/aws-lambda -type f | sort` to confirm all files exist
2. Check `pyproject.toml` for the `role_adherence` or `role-adherence` extra to confirm the correct `MODULE_EXTRA` value for the Dockerfile
