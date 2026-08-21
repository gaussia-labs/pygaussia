"""Gaussia Generators Lambda business logic.

Generates synthetic test datasets from context documents using LLMs.
"""

import asyncio
import importlib
import os
import tempfile
from typing import Any


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


def run(payload: dict) -> dict[str, Any]:
    """Generate synthetic test datasets.

    Args:
        payload: Request JSON body with connector, context and config

    Returns:
        dict: Generated datasets

    Example payload:
        {
            "connector": {
                "class_path": "langchain_groq.chat_models.ChatGroq",
                "params": {
                    "model": "qwen/qwen3-32b",
                    "api_key": "your-api-key",
                    "temperature": 0.7
                }
            },
            "context": "# Knowledge Base\n\nYour markdown content...",
            "config": {
                "assistant_id": "my-assistant",
                "num_queries": 3,
                "language": "english",
                "conversation_mode": false,
                "max_chunk_size": 2000,
                "min_chunk_size": 200,
                "seed_examples": ["Example question 1?", "Example question 2?"]
            }
        }
    """
    return asyncio.get_event_loop().run_until_complete(_async_run(payload))


async def _async_run(payload: dict) -> dict[str, Any]:
    """Async generator implementation."""
    from gaussia.generators import BaseGenerator, LocalMarkdownLoader

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

    config = payload.get("config", {})

    # Initialize generator
    generator = BaseGenerator(model=model, use_structured_output=True)

    # Create context loader
    loader = LocalMarkdownLoader(
        max_chunk_size=config.get("max_chunk_size", 2000),
        min_chunk_size=config.get("min_chunk_size", 200),
    )

    # Get context from payload
    context_content = payload.get("context", "")
    assistant_id = config.get("assistant_id", "test-assistant")

    if not context_content:
        return {"success": False, "error": "No context provided"}

    # Write context to temp file for loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(context_content)
        temp_path = f.name

    try:
        datasets = await generator.generate_dataset(
            context_loader=loader,
            source=temp_path,
            assistant_id=assistant_id,
            num_queries_per_chunk=config.get("num_queries", 3),
            language=config.get("language", "english"),
            seed_examples=config.get("seed_examples"),
            conversation_mode=config.get("conversation_mode", False),
        )
    finally:
        # Cleanup temp file
        os.unlink(temp_path)

    return {
        "success": True,
        "datasets": [d.model_dump() for d in datasets],
        "total_datasets": len(datasets),
        "total_batches": sum(len(d.conversation) for d in datasets),
    }
