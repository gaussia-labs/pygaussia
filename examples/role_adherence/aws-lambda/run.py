"""Gaussia RoleAdherence Lambda business logic.

Evaluates whether an AI assistant adheres to its defined role across conversation turns.
"""

import importlib
import os
from typing import Any

from gaussia.core import Retriever
from gaussia.metrics.role_adherence import LLMJudgeStrategy, RoleAdherence
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
