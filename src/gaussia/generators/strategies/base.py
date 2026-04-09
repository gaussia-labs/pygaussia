"""Base class for chunk selection strategies."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from gaussia.schemas.generators import Chunk


class BaseChunkSelectionStrategy(ABC):
    """Abstract base class for chunk selection strategies.

    Strategies determine how chunks are selected and grouped for
    dataset generation. Each yielded group becomes a separate dataset.

    Example strategies:
        - Sequential: Process all chunks in order (single group)
        - Random Sampling: Randomly select k chunks, repeat n times
        - Clustering: Group related chunks together

    Example:
        >>> strategy = SequentialStrategy()
        >>> chunks = loader.load("docs.md")
        >>> for chunk_group in strategy.select(chunks):
        ...     dataset = await generator.process_group(chunk_group)
    """

    @abstractmethod
    def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
        """Select and group chunks for processing.

        Args:
            chunks: All available chunks from the context loader.

        Yields:
            list[Chunk]: Groups of chunks to process together.
                Each group becomes a separate dataset.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


__all__ = ["BaseChunkSelectionStrategy"]
