"""Gaussia BestOf Lambda business logic.

Tournament-style comparison of multiple AI assistants to find the best one.
"""

import importlib
import os
from typing import Any

from gaussia.core import Retriever
from gaussia.metrics.best_of import BestOf
from gaussia.schemas import Dataset


def create_llm_connector(connector_config: dict) -> Any:
    """Factory method to create LLM connector from dynamic class path.

    Args:
        connector_config: Configuration dict with:
            - class_path: Full class path (e.g., "langchain_groq.chat_models.ChatGroq")
            - params: Dict of parameters to pass to the class constructor

    Returns:
        Instantiated LLM connector

    Supported connectors:
        - langchain_groq.chat_models.ChatGroq
        - langchain_openai.chat_models.ChatOpenAI
        - langchain_google_genai.chat_models.ChatGoogleGenerativeAI
        - langchain_ollama.chat_models.ChatOllama
    """
    class_path = connector_config.get("class_path")
    params = connector_config.get("params", {})

    if not class_path:
        raise ValueError("connector.class_path is required")

    # Dynamic import
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Support environment variable fallback for api_key
    if "api_key" not in params or not params["api_key"]:
        env_key = os.environ.get("LLM_API_KEY")
        if env_key:
            params["api_key"] = env_key

    return cls(**params)


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
    """Run BestOf tournament on payload datasets.

    Args:
        payload: Request JSON body with connector, datasets and config

    Returns:
        dict: Tournament results with winner and contest details

    Example payload:
        {
            "connector": {
                "class_path": "langchain_groq.chat_models.ChatGroq",
                "params": {
                    "model": "qwen/qwen3-32b",
                    "api_key": "your-api-key",
                    "temperature": 0.0
                }
            },
            "datasets": [
                {
                    "session_id": "comparison_session",
                    "assistant_id": "assistant_a",
                    "language": "english",
                    "context": "System context...",
                    "conversation": [
                        {
                            "qa_id": "q1",
                            "query": "User question",
                            "assistant": "Assistant A response",
                            "ground_truth_assistant": "Expected response"
                        }
                    ]
                },
                {
                    "session_id": "comparison_session",
                    "assistant_id": "assistant_b",
                    "language": "english",
                    "context": "System context...",
                    "conversation": [
                        {
                            "qa_id": "q1",
                            "query": "User question",
                            "assistant": "Assistant B response",
                            "ground_truth_assistant": "Expected response"
                        }
                    ]
                }
            ],
            "config": {
                "criteria": "Overall response quality",
                "use_structured_output": true,
                "verbose": false
            }
        }
    """
    # Get connector config
    connector_config = payload.get("connector", {})
    if not connector_config:
        return {"success": False, "error": "No connector configuration provided"}

    try:
        model = create_llm_connector(connector_config)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Failed to create LLM connector: {e}"}

    # Validate datasets
    datasets = payload.get("datasets", [])
    if not datasets:
        return {"success": False, "error": "No datasets provided"}

    if len(datasets) < 2:
        return {
            "success": False,
            "error": "BestOf requires at least 2 datasets with different assistant_ids",
        }

    # Get unique assistant IDs
    assistant_ids = {d.get("assistant_id") for d in datasets}
    if len(assistant_ids) < 2:
        return {
            "success": False,
            "error": "BestOf requires datasets from at least 2 different assistants",
        }

    config = payload.get("config", {})

    # Run BestOf metric
    try:
        metrics = BestOf.run(
            lambda: PayloadRetriever(payload),
            model=model,
            use_structured_output=config.get("use_structured_output", True),
            criteria=config.get("criteria", "Overall response quality"),
            verbose=config.get("verbose", False),
        )
    except Exception as e:
        return {"success": False, "error": f"BestOf evaluation failed: {e}"}

    if not metrics:
        return {"success": False, "error": "No metrics produced"}

    # Extract results
    result = metrics[0]
    return {
        "success": True,
        "winner": result.bestof_winner_id,
        "contestants": list(assistant_ids),
        "total_rounds": max(c.round for c in result.bestof_contests),
        "contests": [
            {
                "round": c.round,
                "left": c.left_id,
                "right": c.right_id,
                "winner": c.winner_id,
                "confidence": c.confidence,
                "verdict": c.verdict,
                "reasoning": c.reasoning,
            }
            for c in result.bestof_contests
        ],
    }
