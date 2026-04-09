"""Embedder abstract base class for text encoding."""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedding models.

    Implementations encode text into dense vector representations.
    Override encode_query for models that distinguish between
    document and query encoding (e.g., instruction-prefixed models).
    """

    @abstractmethod
    def encode(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into embedding vectors.

        Args:
            sentences: List of texts to encode.

        Returns:
            Array of shape (n_sentences, embedding_dim).
        """
        ...

    def encode_query(self, sentences: list[str]) -> np.ndarray:
        """Encode query sentences.

        Override for models with query-specific encoding (e.g., instruction prefix).
        Defaults to standard encode.

        Args:
            sentences: List of query texts to encode.

        Returns:
            Array of shape (n_sentences, embedding_dim).
        """
        return self.encode(sentences)
