"""Reranker abstract base class for document scoring."""

from abc import ABC, abstractmethod


class Reranker(ABC):
    """Abstract base class for document reranking models.

    Implementations score the relevance of documents against a query.
    """

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score the relevance of each document to a query.

        Args:
            query: The query text.
            documents: List of document texts to score against the query.

        Returns:
            List of relevance scores, one per document.
        """
        ...
