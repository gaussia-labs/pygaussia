"""Sequential chunk selection strategy."""

from collections.abc import Iterator

from gaussia.schemas.generators import Chunk

from .base import BaseChunkSelectionStrategy


class SequentialStrategy(BaseChunkSelectionStrategy):
    """Process all chunks sequentially in a single group.

    This is the default strategy that maintains backward compatibility
    with the original behavior. All chunks are yielded as a single group,
    resulting in one dataset containing queries from all chunks.

    Example:
        >>> strategy = SequentialStrategy()
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> list(strategy.select(chunks))
        [[chunk1, chunk2, chunk3]]  # Single group with all chunks
    """

    def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        """Yield all chunks as a single group.

        Args:
            chunks: All available chunks from the context loader.

        Yields:
            list[Chunk]: Single group containing all chunks.
        """
        if chunks:
            yield chunks


__all__ = ["SequentialStrategy"]
