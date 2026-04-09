"""Random sampling chunk selection strategy."""

import random
from collections.abc import Iterator

from gaussia.schemas.generators import Chunk

from .base import BaseChunkSelectionStrategy


class RandomSamplingStrategy(BaseChunkSelectionStrategy):
    """Randomly sample chunks multiple times.

    Creates multiple chunk groups by randomly selecting a subset of chunks.
    Each group becomes a separate dataset. Useful for:
    - Testing model performance on random context subsets
    - Creating diverse test scenarios from the same source
    - Reducing dataset size while maintaining coverage

    Args:
        num_samples: Number of random samples to generate (datasets).
        chunks_per_sample: Number of chunks to select per sample.
        seed: Random seed for reproducibility. If None, uses system random.
        with_replacement: If True, same chunk can appear multiple times
            within a single sample. Default is False.

    Example:
        >>> strategy = RandomSamplingStrategy(num_samples=3, chunks_per_sample=2)
        >>> chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
        >>> list(strategy.select(chunks))
        [[chunk2, chunk4], [chunk1, chunk5], [chunk3, chunk1]]  # 3 random samples
    """

    def __init__(
        self,
        num_samples: int = 5,
        chunks_per_sample: int = 3,
        seed: int | None = None,
        with_replacement: bool = False,
    ):
        """Initialize the random sampling strategy.

        Args:
            num_samples: Number of random samples to generate.
            chunks_per_sample: Number of chunks per sample.
            seed: Optional random seed for reproducibility.
            with_replacement: Allow same chunk multiple times per sample.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if chunks_per_sample < 1:
            raise ValueError("chunks_per_sample must be at least 1")

        self.num_samples = num_samples
        self.chunks_per_sample = chunks_per_sample
        self.seed = seed
        self.with_replacement = with_replacement

    def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        """Randomly sample chunks to create multiple groups.

        Args:
            chunks: All available chunks from the context loader.

        Yields:
            list[Chunk]: Random subsets of chunks.

        Note:
            If chunks_per_sample exceeds available chunks and
            with_replacement is False, will sample all available chunks.
        """
        if not chunks:
            return

        rng = random.Random(self.seed)
        k = min(self.chunks_per_sample, len(chunks))

        for _ in range(self.num_samples):
            if self.with_replacement:
                sample = rng.choices(chunks, k=self.chunks_per_sample)
            else:
                sample = rng.sample(chunks, k=k)
            yield sample

    def __repr__(self) -> str:
        return (
            f"RandomSamplingStrategy("
            f"num_samples={self.num_samples}, "
            f"chunks_per_sample={self.chunks_per_sample}, "
            f"seed={self.seed})"
        )


__all__ = ["RandomSamplingStrategy"]
