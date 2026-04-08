"""SentenceTransformer-based embedder implementation."""

import numpy as np
from sentence_transformers import SentenceTransformer

from pygaussia.core.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a SentenceTransformers model.

    Args:
        model_name: Name or path of the SentenceTransformer model.
        batch_size: Batch size for encoding.
        normalize_embeddings: Whether to L2-normalize output embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ):
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                sentences,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )
        )
