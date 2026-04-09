"""Cosine similarity scorer using a pluggable Embedder."""

from gaussia.core.embedder import Embedder
from gaussia.core.similarity_scorer import SimilarityScorer
from gaussia.utils.math import cosine_similarity


class CosineSimilarity(SimilarityScorer):
    """Computes semantic similarity using cosine distance between embeddings."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def calculate(self, assistant: str, ground_truth: str) -> float:
        embeddings = self._embedder.encode([assistant, ground_truth])
        return cosine_similarity(embeddings[0], embeddings[1])
