"""Mock retriever implementations for testing."""

from gaussia.core.retriever import Retriever
from gaussia.schemas.common import Dataset
from tests.fixtures.mock_data import (
    create_agentic_dataset,
    create_bestof_dataset,
    create_bias_dataset,
    create_context_dataset,
    create_conversational_dataset,
    create_emotional_dataset,
    create_multiple_datasets,
    create_regulatory_dataset,
    create_sample_dataset,
    create_toxicity_dataset,
    create_vision_dataset,
)


class MockRetriever(Retriever):
    """Mock retriever that returns predefined test data."""

    def __init__(self, datasets: list[Dataset] = None, **kwargs):
        """
        Initialize mock retriever with optional predefined datasets.

        Args:
            datasets: List of Dataset objects to return. If None, uses default sample dataset.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self._datasets = datasets

    def load_dataset(self) -> list[Dataset]:
        """Load mock dataset."""
        if self._datasets is not None:
            return self._datasets
        return [create_sample_dataset()]


class EmptyRetriever(Retriever):
    """Mock retriever that returns empty dataset list."""

    def load_dataset(self) -> list[Dataset]:
        """Return empty dataset list."""
        return []


class SingleDatasetRetriever(Retriever):
    """Mock retriever that returns a single dataset."""

    def load_dataset(self) -> list[Dataset]:
        """Return single dataset."""
        return [create_sample_dataset()]


class MultipleDatasetRetriever(Retriever):
    """Mock retriever that returns multiple datasets."""

    def load_dataset(self) -> list[Dataset]:
        """Return multiple datasets."""
        return create_multiple_datasets()


class EmotionalDatasetRetriever(Retriever):
    """Mock retriever for Humanity metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return emotional dataset."""
        return [create_emotional_dataset()]


class ConversationalDatasetRetriever(Retriever):
    """Mock retriever for Conversational metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return conversational dataset."""
        return [create_conversational_dataset()]


class BiasDatasetRetriever(Retriever):
    """Mock retriever for Bias metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return bias testing dataset."""
        return [create_bias_dataset()]


class ToxicityDatasetRetriever(Retriever):
    """Mock retriever for Toxicity metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return toxicity testing dataset."""
        return [create_toxicity_dataset()]


class ContextDatasetRetriever(Retriever):
    """Mock retriever for Context metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return context testing dataset."""
        return [create_context_dataset()]


class BestOfDatasetRetriever(Retriever):
    """Mock retriever for BestOf metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return best-of testing dataset."""
        return create_bestof_dataset()


class AgenticDatasetRetriever(Retriever):
    """Mock retriever for Agentic metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return agentic testing dataset with K responses."""
        return create_agentic_dataset()


class VisionDatasetRetriever(Retriever):
    """Mock retriever for Vision metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return vision testing dataset."""
        return [create_vision_dataset()]


class RegulatoryDatasetRetriever(Retriever):
    """Mock retriever for Regulatory metric testing."""

    def load_dataset(self) -> list[Dataset]:
        """Return regulatory testing dataset."""
        return [create_regulatory_dataset()]


class ErrorRetriever(Retriever):
    """Mock retriever that raises an error."""

    def load_dataset(self) -> list[Dataset]:
        """Raise an error."""
        raise ValueError("Simulated retriever error")
