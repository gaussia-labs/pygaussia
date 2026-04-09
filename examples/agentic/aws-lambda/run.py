"""Gaussia Agentic Lambda business logic.

Evaluates complete agent conversations using pass@K and tool correctness metrics.
A conversation is correct only if ALL its interactions are correct.
"""

import importlib
import os
from typing import Any

from gaussia.core import Retriever
from gaussia.metrics.agentic import Agentic
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
    """Run Agentic metric on payload datasets.

    Evaluates complete agent conversations. Each Dataset represents one complete conversation.
    A conversation is correct only if ALL its interactions are correct.

    Args:
        payload: Request JSON body with connector, datasets (conversations) and config

    Returns:
        dict: Agentic evaluation results with pass@K, pass^K, and tool correctness per conversation

    Example payload (2 different conversations):
        {
            "connector": {
                "class_path": "langchain_groq.chat_models.ChatGroq",
                "params": {
                    "model": "llama-3.3-70b-versatile",
                    "api_key": "your-api-key",
                    "temperature": 0.0
                }
            },
            "datasets": [
                {
                    "session_id": "conversation_001",
                    "assistant_id": "agent_v1",
                    "language": "english",
                    "context": "Math calculator conversation",
                    "conversation": [
                        {
                            "qa_id": "q1_interaction1",
                            "query": "What is 5 + 3?",
                            "assistant": "The result is 8.",
                            "ground_truth_assistant": "5 + 3 equals 8",
                            "agentic": {
                                "tools_used": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "result": 8,
                                        "step": 1
                                    }
                                ],
                                "final_answer_uses_tools": true
                            },
                            "ground_truth_agentic": {
                                "expected_tools": [
                                    {
                                        "tool_name": "calculator",
                                        "parameters": {"operation": "add", "a": 5, "b": 3},
                                        "step": 1
                                    }
                                ],
                                "tool_sequence_matters": false
                            }
                        },
                        {
                            "qa_id": "q1_interaction2",
                            "query": "What is 10 * 2?",
                            "assistant": "10 times 2 is 20.",
                            "ground_truth_assistant": "20"
                        }
                    ]
                },
                {
                    "session_id": "conversation_002",
                    "assistant_id": "agent_v1",
                    "language": "english",
                    "context": "Simple Q&A conversation",
                    "conversation": [
                        {
                            "qa_id": "q2_interaction1",
                            "query": "What is the capital of France?",
                            "assistant": "The capital of France is Paris.",
                            "ground_truth_assistant": "Paris"
                        }
                    ]
                }
            ],
            "config": {
                "threshold": 0.7,
                "tool_threshold": 0.75,
                "tool_weights": {
                    "selection": 0.25,
                    "parameters": 0.25,
                    "sequence": 0.25,
                    "utilization": 0.25
                },
                "use_structured_output": true,
                "verbose": false,
                "k": 3
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

    # Each dataset represents one complete conversation
    # A conversation is correct only if ALL its interactions are correct

    config = payload.get("config", {})

    # Run Agentic metric
    k_value = config.get("k")
    if not k_value:
        return {"success": False, "error": "config.k is required"}

    try:
        metrics = Agentic.run(
            lambda: PayloadRetriever(payload),
            model=model,
            k=k_value,
            threshold=config.get("threshold", 0.7),
            tool_threshold=config.get("tool_threshold", 1.0),
            tool_weights=config.get("tool_weights"),
            use_structured_output=config.get("use_structured_output", True),
            verbose=config.get("verbose", False),
        )
    except Exception as e:
        return {"success": False, "error": f"Agentic evaluation failed: {e}"}

    if not metrics:
        return {"success": False, "error": "No metrics produced"}

    # Extract per-conversation results — pass@K is computed per conversation
    results = []
    for metric in metrics:
        result = {
            "session_id": metric.session_id,
            "total_interactions": metric.total_interactions,
            "correct_interactions": metric.correct_interactions,
            "is_fully_correct": metric.is_fully_correct,
            "threshold": metric.threshold,
            "correctness_scores": [round(s, 3) for s in metric.correctness_scores],
            "correct_indices": metric.correct_indices,
            "k": metric.k,
            "pass_at_k": round(metric.pass_at_k, 4),
            "pass_pow_k": round(metric.pass_pow_k, 4),
        }

        if metric.tool_correctness_scores:
            result["tool_correctness_scores"] = [
                {
                    "tool_selection_correct": round(tc.tool_selection_correct, 3),
                    "parameter_accuracy": round(tc.parameter_accuracy, 3),
                    "sequence_correct": round(tc.sequence_correct, 3),
                    "result_utilization": round(tc.result_utilization, 3),
                    "overall_correctness": round(tc.overall_correctness, 3),
                    "is_correct": tc.is_correct,
                    "reasoning": tc.reasoning,
                }
                if tc
                else None
                for tc in metric.tool_correctness_scores
            ]

        results.append(result)

    n_conversations = len(metrics)
    c_conversations = sum(1 for m in metrics if m.is_fully_correct)

    return {
        "success": True,
        "per_conversation_metrics": results,
        "summary": {
            "total_conversations": n_conversations,
            "fully_correct_conversations": c_conversations,
            "conversation_success_rate": round(c_conversations / n_conversations, 4),
            "k": k_value,
        },
    }
