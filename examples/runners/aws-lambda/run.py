"""Gaussia Runners Lambda business logic.

Executes test batches against AI systems and collects responses for evaluation.
Supports both Alquimia API and direct LLM testing via dynamic connectors.
"""

import asyncio
import importlib
import os
import time
from typing import Any

from gaussia.runners import AlquimiaRunner
from gaussia.schemas import Batch, Dataset


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


class LLMRunner:
    """Runner that executes tests directly against LangChain LLMs."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def run_batch(self, batch: Batch, session_id: str, context: str = "") -> tuple[Batch, bool, float]:
        """Execute single test case against the LLM."""
        start = time.time()

        try:
            # Build prompt with context if available
            if context:
                prompt = f"Context:\n{context}\n\nQuestion: {batch.query}"
            else:
                prompt = batch.query

            # Invoke the LLM
            response = await self.llm.ainvoke(prompt)

            # Extract content from response
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            execution_time = (time.time() - start) * 1000
            updated_batch = batch.model_copy(update={"assistant": content})
            return updated_batch, True, execution_time

        except Exception as e:
            execution_time = (time.time() - start) * 1000
            updated_batch = batch.model_copy(update={"assistant": f"[ERROR] {e}"})
            return updated_batch, False, execution_time

    async def run_dataset(self, dataset: Dataset) -> tuple[Dataset, dict]:
        """Execute all batches in dataset."""
        updated_batches = []
        successes = failures = 0
        total_time = 0.0

        for batch in dataset.conversation:
            updated, success, time_ms = await self.run_batch(batch, dataset.session_id, dataset.context)
            updated_batches.append(updated)
            total_time += time_ms
            if success:
                successes += 1
            else:
                failures += 1

        updated_dataset = dataset.model_copy(update={"conversation": updated_batches})

        summary = {
            "session_id": dataset.session_id,
            "total_batches": len(dataset.conversation),
            "successes": successes,
            "failures": failures,
            "total_execution_time_ms": total_time,
            "avg_batch_time_ms": total_time / len(dataset.conversation) if dataset.conversation else 0,
        }

        return updated_dataset, summary


def run(payload: dict) -> dict[str, Any]:
    """Run tests against an AI system.

    Supports two modes:
    1. Alquimia mode: Use config with base_url, api_key, agent_id
    2. LLM mode: Use connector with class_path and params

    Args:
        payload: Request JSON body with datasets and config/connector

    Returns:
        dict: Test results with responses from the AI system

    Example payload (Alquimia mode):
        {
            "datasets": [...],
            "config": {
                "base_url": "https://api.alquimia.ai",
                "api_key": "your-alquimia-api-key",
                "agent_id": "your-agent-id",
                "channel_id": "your-channel-id"
            }
        }

    Example payload (LLM mode):
        {
            "datasets": [...],
            "connector": {
                "class_path": "langchain_groq.chat_models.ChatGroq",
                "params": {
                    "model": "qwen/qwen3-32b",
                    "api_key": "your-api-key"
                }
            }
        }
    """
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async runner implementation."""
    config = payload.get("config", {})
    connector_config = payload.get("connector", {})

    # Load datasets from payload
    raw_datasets = payload.get("datasets", [])
    if not raw_datasets:
        return {"success": False, "error": "No datasets provided"}

    datasets = []
    for data in raw_datasets:
        datasets.append(Dataset.model_validate(data))

    # Determine which runner to use
    if connector_config:
        # LLM mode - use dynamic connector
        try:
            llm = create_llm_connector(connector_config)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to create LLM connector: {e}"}

        runner = LLMRunner(llm)
    else:
        # Alquimia mode - use AlquimiaRunner
        base_url = config.get("base_url") or os.environ.get("ALQUIMIA_BASE_URL")
        api_key = config.get("api_key") or os.environ.get("ALQUIMIA_API_KEY")
        agent_id = config.get("agent_id")
        channel_id = config.get("channel_id")

        if not base_url:
            return {"success": False, "error": "No base_url provided"}
        if not api_key:
            return {"success": False, "error": "No api_key provided"}
        if not agent_id:
            return {"success": False, "error": "No agent_id provided"}

        runner = AlquimiaRunner(
            base_url=base_url,
            api_key=api_key,
            agent_id=agent_id,
            channel_id=channel_id,
            api_version=config.get("api_version", ""),
        )

    # Run all datasets
    results = []
    summaries = []

    for dataset in datasets:
        updated_dataset, summary = await runner.run_dataset(dataset)
        results.append(updated_dataset.model_dump())
        summaries.append(summary)

    return {
        "success": True,
        "datasets": results,
        "summaries": summaries,
        "total_datasets": len(results),
    }
