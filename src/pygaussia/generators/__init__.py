"""Generators module for Gaussia.

Provides synthetic dataset generators for creating test datasets
from context documents to evaluate AI assistants.

The BaseGenerator class uses LangChain's BaseChatModel interface, allowing
any LangChain-compatible model to be used for generation.

Chunk Selection Strategies:
- SequentialStrategy: Process all chunks in order (default behavior)
- RandomSamplingStrategy: Randomly sample chunks multiple times

Usage:
    ```python
    from langchain_openai import ChatOpenAI
    from pygaussia.generators import BaseGenerator, create_markdown_loader

    model = ChatOpenAI(model="gpt-4o-mini")
    generator = BaseGenerator(model=model)

    loader = create_markdown_loader()
    datasets = await generator.generate_dataset(
        context_loader=loader,
        source="./docs/knowledge_base.md",
        assistant_id="my-assistant",
    )
    ```
"""

from loguru import logger

from pygaussia.schemas.generators import (
    BaseChunkSelectionStrategy,
    BaseContextLoader,
    BaseGenerator,
    Chunk,
    ConversationTurn,
    GeneratedConversationOutput,
    GeneratedQueriesOutput,
    GeneratedQuery,
)

from .context_loaders import LocalMarkdownLoader
from .strategies import RandomSamplingStrategy, SequentialStrategy


def create_markdown_loader(
    max_chunk_size: int = 2000,
    min_chunk_size: int = 200,
    overlap: int = 100,
    header_levels: list[int] | None = None,
) -> LocalMarkdownLoader:
    """Create a local markdown context loader.

    Args:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap: Overlap between size-based chunks
        header_levels: Header levels to split on

    Returns:
        Configured loader instance.
    """
    logger.info("Creating local markdown loader")
    return LocalMarkdownLoader(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=overlap,
        header_levels=header_levels,
    )


__all__ = [
    "BaseGenerator",
    "BaseContextLoader",
    "BaseChunkSelectionStrategy",
    "Chunk",
    "GeneratedQuery",
    "GeneratedQueriesOutput",
    "ConversationTurn",
    "GeneratedConversationOutput",
    "LocalMarkdownLoader",
    "SequentialStrategy",
    "RandomSamplingStrategy",
    "create_markdown_loader",
]
