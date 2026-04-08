"""Vision metrics: Similarity and Hallucination for VLM evaluation."""

from abc import abstractmethod

from tqdm.auto import tqdm

from pygaussia.core import Gaussia, Retriever, SimilarityScorer
from pygaussia.schemas import Batch
from pygaussia.schemas.vision import (
    VisionHallucinationInteraction,
    VisionHallucinationMetric,
    VisionSimilarityInteraction,
    VisionSimilarityMetric,
)

_DEFAULT_MODEL = "all-mpnet-base-v2"
_DEFAULT_THRESHOLD = 0.75


def _default_scorer() -> SimilarityScorer:
    from pygaussia.embedders import SentenceTransformerEmbedder
    from pygaussia.scorers import CosineSimilarity

    return CosineSimilarity(SentenceTransformerEmbedder(model=_DEFAULT_MODEL))


class _VisionBase(Gaussia):
    """Shared base for vision metrics.

    Compares VLM free-text descriptions against human ground truth
    using a pluggable SimilarityScorer.

    Expected Batch fields:
        assistant              — VLM free-text description of the scene
        ground_truth_assistant — human description of what actually happened
    """

    def __init__(
        self,
        retriever: type[Retriever],
        scorer: SimilarityScorer | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self._scorer = scorer if scorer is not None else _default_scorer()
        self._threshold = threshold
        self._session_data: dict[str, dict] = {}

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if session_id not in self._session_data:
            self._session_data[session_id] = {"assistant_id": assistant_id, "interactions": []}

        for interaction in tqdm(batch, desc=session_id, unit="frame"):
            self.logger.debug(f"QA ID: {interaction.qa_id}")
            similarity = self._scorer.calculate(interaction.assistant, interaction.ground_truth_assistant)
            self._session_data[session_id]["interactions"].append(
                {"qa_id": interaction.qa_id, "similarity_score": round(similarity, 4)}
            )

    @abstractmethod
    def on_process_complete(self):
        raise NotImplementedError


class VisionSimilarity(_VisionBase):
    """Measures how accurately the VLM describes scenes compared to human ground truth.

    Uses a SimilarityScorer to compare the VLM description against the human-annotated
    ground truth. A score of 1.0 means the descriptions are semantically identical;
    0.0 means they are completely unrelated.
    """

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            raw = data["interactions"]
            scores = [i["similarity_score"] for i in raw]
            mean = round(sum(scores) / len(scores), 4)
            min_s = round(min(scores), 4)
            max_s = round(max(scores), 4)
            summary = (
                f"The model's descriptions have an average similarity of {mean:.0%} with the ground truth "
                f"(min: {min_s:.0%}, max: {max_s:.0%}) across {len(scores)} frames."
            )
            self.metrics.append(
                VisionSimilarityMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    mean_similarity=mean,
                    min_similarity=min_s,
                    max_similarity=max_s,
                    summary=summary,
                    interactions=[VisionSimilarityInteraction(**i) for i in raw],
                )
            )


class VisionHallucination(_VisionBase):
    """Measures how often the VLM describes scenes that differ significantly from reality.

    A frame is considered a hallucination when the similarity score between the VLM
    description and the ground truth falls below the configured threshold.
    """

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            raw = data["interactions"]
            n_frames = len(raw)
            interactions = [
                VisionHallucinationInteraction(
                    qa_id=i["qa_id"],
                    similarity_score=i["similarity_score"],
                    is_hallucination=i["similarity_score"] < self._threshold,
                )
                for i in raw
            ]
            n_hallucinations = sum(1 for i in interactions if i.is_hallucination)
            rate = round(n_hallucinations / n_frames, 4) if n_frames > 0 else 0.0
            summary = (
                f"The model hallucinated in {n_hallucinations} of {n_frames} frames "
                f"({rate:.0%}). A frame is considered a hallucination when similarity "
                f"with the ground truth falls below {self._threshold}."
            )
            self.metrics.append(
                VisionHallucinationMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    hallucination_rate=rate,
                    n_hallucinations=n_hallucinations,
                    n_frames=n_frames,
                    threshold=self._threshold,
                    summary=summary,
                    interactions=interactions,
                )
            )
