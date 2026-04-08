"""Embedding-based group extractor."""

import numpy as np

from pygaussia.core.embedder import Embedder
from pygaussia.core.extractor import BaseGroupExtractor
from pygaussia.schemas.toxicity import GroupDetection


class EmbeddingGroupExtractor(BaseGroupExtractor):
    """Detects whether a text mentions each group using embedding cosine similarity
    against per-group prototype phrases.

    - Precompute prototype embeddings per group.
    - Encode each text once.
    - For each group: score = max cosine(text, prototypes[group])
    - present = score >= threshold[group] (or default_threshold)

    Important: when normalize_embeddings=True, vectors are L2-normalized so dot product == cosine.

    Args:
        embedder: Embedder instance for encoding text.
        group_prototypes: Dict mapping group name to list of prototype phrases.
        thresholds: Optional dict of per-group detection thresholds.
        default_threshold: Default threshold for groups without a specific one.
        normalize_embeddings: Whether to L2-normalize embeddings after encoding.
    """

    def __init__(
        self,
        embedder: Embedder,
        group_prototypes: dict[str, list[str]],
        thresholds: dict[str, float] | None = None,
        default_threshold: float = 0.50,
        normalize_embeddings: bool = True,
    ):
        if not group_prototypes:
            raise ValueError("group_prototypes must be non-empty.")
        for g, ps in group_prototypes.items():
            if not ps:
                raise ValueError(f"group_prototypes['{g}'] is empty; each group needs at least 1 prototype.")

        self._embedder = embedder
        self.group_prototypes = group_prototypes
        self.thresholds = thresholds or {}
        self.default_threshold = float(default_threshold)
        self.normalize_embeddings = bool(normalize_embeddings)

        self._proto_embs: dict[str, np.ndarray] = {}
        for g, protos in self.group_prototypes.items():
            embs = self._encode(protos)
            if embs.ndim != 2:
                raise ValueError(f"Prototype embeddings for group '{g}' must be 2D, got shape={embs.shape}")
            self._proto_embs[g] = embs

    def _encode(self, texts: list[str]) -> np.ndarray:
        embs = self._embedder.encode(texts)
        embs = np.asarray(embs)

        if self.normalize_embeddings:
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            embs = embs / norms

        return embs

    def detect_one(self, text: str) -> dict[str, GroupDetection]:
        """Detect group mentions in a single text.

        Args:
            text: The text to analyze.

        Returns:
            Dict mapping group names to detection results.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        e = self._encode([text])[0]

        results: dict[str, GroupDetection] = {}
        for g, P in self._proto_embs.items():
            sims = P @ e
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            thr = float(self.thresholds.get(g, self.default_threshold))
            results[g] = GroupDetection(
                present=best_sim >= thr,
                score=best_sim,
                best_prototype=self.group_prototypes[g][best_idx],
                best_prototype_index=best_idx,
            )
        return results

    def detect_batch(self, texts: list[str]) -> list[dict[str, GroupDetection]]:
        """Detect group mentions in a batch of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of dicts, one per text, mapping group names to detection results.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a list[str]")

        E = self._encode(texts)
        out: list[dict[str, GroupDetection]] = []

        for i in range(E.shape[0]):
            e = E[i]
            row: dict[str, GroupDetection] = {}
            for g, P in self._proto_embs.items():
                sims = P @ e
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                thr = float(self.thresholds.get(g, self.default_threshold))
                row[g] = GroupDetection(
                    present=best_sim >= thr,
                    score=best_sim,
                    best_prototype=self.group_prototypes[g][best_idx],
                    best_prototype_index=best_idx,
                )
            out.append(row)

        return out
