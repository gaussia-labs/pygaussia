"""Chunk selection strategies for Gaussia generators.

This module provides strategies for selecting and grouping chunks
during dataset generation.

Available strategies:
- SequentialStrategy: Process all chunks in order (default)
- RandomSamplingStrategy: Randomly sample chunks multiple times

Example:
    >>> from pygaussia.generators.strategies import RandomSamplingStrategy
    >>> strategy = RandomSamplingStrategy(num_samples=5, chunks_per_sample=3)
    >>> datasets = await generator.generate_dataset(
    ...     context_loader=loader,
    ...     source="docs.md",
    ...     assistant_id="test",
    ...     selection_strategy=strategy,
    ... )
"""

from .base import BaseChunkSelectionStrategy
from .random_sampling import RandomSamplingStrategy
from .sequential import SequentialStrategy

__all__ = [
    "BaseChunkSelectionStrategy",
    "RandomSamplingStrategy",
    "SequentialStrategy",
]
