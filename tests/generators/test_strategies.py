"""Tests for chunk selection strategies."""

import pytest

from gaussia.generators.strategies import (
    BaseChunkSelectionStrategy,
    RandomSamplingStrategy,
    SequentialStrategy,
)
from gaussia.schemas.generators import Chunk


class TestSequentialStrategy:
    """Test suite for SequentialStrategy."""

    def test_sequential_yields_all_chunks_as_single_group(self, sample_chunks: list[Chunk]):
        """Test that sequential strategy yields all chunks as one group."""
        strategy = SequentialStrategy()
        groups = list(strategy.select(sample_chunks))

        assert len(groups) == 1
        assert groups[0] == sample_chunks

    def test_sequential_with_empty_list(self):
        """Test sequential strategy with empty chunk list."""
        strategy = SequentialStrategy()
        groups = list(strategy.select([]))

        assert len(groups) == 0

    def test_sequential_with_single_chunk(self, sample_chunk: Chunk):
        """Test sequential strategy with a single chunk."""
        strategy = SequentialStrategy()
        groups = list(strategy.select([sample_chunk]))

        assert len(groups) == 1
        assert groups[0] == [sample_chunk]


class TestRandomSamplingStrategy:
    """Test suite for RandomSamplingStrategy."""

    def test_random_sampling_basic(self, sample_chunks: list[Chunk]):
        """Test basic random sampling functionality."""
        strategy = RandomSamplingStrategy(
            num_samples=2,
            chunks_per_sample=1,
            seed=42,
        )
        groups = list(strategy.select(sample_chunks))

        assert len(groups) == 2
        for group in groups:
            assert len(group) == 1
            assert group[0] in sample_chunks

    def test_random_sampling_reproducibility(self, sample_chunks: list[Chunk]):
        """Test that same seed produces same results."""
        strategy1 = RandomSamplingStrategy(
            num_samples=3,
            chunks_per_sample=1,
            seed=42,
        )
        strategy2 = RandomSamplingStrategy(
            num_samples=3,
            chunks_per_sample=1,
            seed=42,
        )

        groups1 = list(strategy1.select(sample_chunks))
        groups2 = list(strategy2.select(sample_chunks))

        for g1, g2 in zip(groups1, groups2, strict=False):
            assert g1 == g2

    def test_random_sampling_different_seeds(self, sample_chunks: list[Chunk]):
        """Test that different seeds can produce different results."""
        # Create enough chunks to make collision unlikely
        many_chunks = [Chunk(content=f"Content {i}", chunk_id=f"chunk_{i}", metadata={}) for i in range(10)]

        strategy1 = RandomSamplingStrategy(
            num_samples=5,
            chunks_per_sample=2,
            seed=42,
        )
        strategy2 = RandomSamplingStrategy(
            num_samples=5,
            chunks_per_sample=2,
            seed=123,
        )

        groups1 = list(strategy1.select(many_chunks))
        groups2 = list(strategy2.select(many_chunks))

        # At least some groups should be different
        differences = sum(1 for g1, g2 in zip(groups1, groups2, strict=False) if g1 != g2)
        assert differences > 0

    def test_random_sampling_chunks_per_sample_exceeds_available(self):
        """Test behavior when chunks_per_sample exceeds available chunks."""
        chunks = [
            Chunk(content="Content 1", chunk_id="chunk_1", metadata={}),
            Chunk(content="Content 2", chunk_id="chunk_2", metadata={}),
        ]

        strategy = RandomSamplingStrategy(
            num_samples=2,
            chunks_per_sample=5,  # More than available
            seed=42,
        )
        groups = list(strategy.select(chunks))

        assert len(groups) == 2
        # Each group should have all available chunks (2, not 5)
        for group in groups:
            assert len(group) == 2

    def test_random_sampling_with_replacement(self):
        """Test sampling with replacement allows duplicates."""
        chunks = [
            Chunk(content="Content 1", chunk_id="chunk_1", metadata={}),
        ]

        strategy = RandomSamplingStrategy(
            num_samples=3,
            chunks_per_sample=2,
            with_replacement=True,
            seed=42,
        )
        groups = list(strategy.select(chunks))

        assert len(groups) == 3
        # With only 1 chunk and replacement=True, each sample should have 2 copies
        for group in groups:
            assert len(group) == 2
            assert all(chunk == chunks[0] for chunk in group)

    def test_random_sampling_without_replacement(self):
        """Test sampling without replacement has no duplicates within a sample."""
        chunks = [Chunk(content=f"Content {i}", chunk_id=f"chunk_{i}", metadata={}) for i in range(5)]

        strategy = RandomSamplingStrategy(
            num_samples=3,
            chunks_per_sample=3,
            with_replacement=False,
            seed=42,
        )
        groups = list(strategy.select(chunks))

        for group in groups:
            # No duplicates within a single sample
            chunk_ids = [c.chunk_id for c in group]
            assert len(chunk_ids) == len(set(chunk_ids))

    def test_random_sampling_empty_list(self):
        """Test random sampling with empty chunk list."""
        strategy = RandomSamplingStrategy(num_samples=3, chunks_per_sample=2)
        groups = list(strategy.select([]))

        assert len(groups) == 0

    def test_random_sampling_invalid_num_samples(self):
        """Test that num_samples must be at least 1."""
        with pytest.raises(ValueError, match="num_samples must be at least 1"):
            RandomSamplingStrategy(num_samples=0, chunks_per_sample=2)

    def test_random_sampling_invalid_chunks_per_sample(self):
        """Test that chunks_per_sample must be at least 1."""
        with pytest.raises(ValueError, match="chunks_per_sample must be at least 1"):
            RandomSamplingStrategy(num_samples=2, chunks_per_sample=0)

    def test_random_sampling_repr(self):
        """Test string representation of RandomSamplingStrategy."""
        strategy = RandomSamplingStrategy(
            num_samples=5,
            chunks_per_sample=3,
            seed=42,
        )
        repr_str = repr(strategy)

        assert "RandomSamplingStrategy" in repr_str
        assert "num_samples=5" in repr_str
        assert "chunks_per_sample=3" in repr_str
        assert "seed=42" in repr_str


class TestBaseChunkSelectionStrategy:
    """Test suite for BaseChunkSelectionStrategy interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseChunkSelectionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChunkSelectionStrategy()

    def test_custom_strategy_implementation(self, sample_chunks: list[Chunk]):
        """Test implementing a custom strategy."""
        from collections.abc import Iterator

        class EvenChunksStrategy(BaseChunkSelectionStrategy):
            """Custom strategy that selects every other chunk."""

            def select(self, chunks: list[Chunk]) -> Iterator[list[Chunk]]:
                yield chunks[::2]  # Every other chunk

        strategy = EvenChunksStrategy()
        groups = list(strategy.select(sample_chunks))

        assert len(groups) == 1
        # With 2 chunks, should get only the first one
        assert len(groups[0]) == 1
        assert groups[0][0] == sample_chunks[0]
