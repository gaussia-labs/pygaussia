"""SimilarityScorer abstract base class."""

from abc import ABC, abstractmethod


class SimilarityScorer(ABC):
    """Abstract interface for computing semantic similarity between two texts."""

    @abstractmethod
    def calculate(self, assistant: str, ground_truth: str) -> float:
        """Return a similarity score between 0.0 and 1.0."""
        ...
