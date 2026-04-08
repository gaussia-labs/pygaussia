"""Pydantic schemas for vision metrics."""

from pydantic import BaseModel

from pygaussia.schemas.metrics import BaseMetric


class VisionSimilarityInteraction(BaseModel):
    qa_id: str
    similarity_score: float


class VisionSimilarityMetric(BaseMetric):
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    summary: str
    interactions: list[VisionSimilarityInteraction]

    def display(self) -> None:
        """Print a human-readable summary of the similarity results."""
        print(f"Session: {self.session_id}  |  Assistant: {self.assistant_id}")
        print(self.summary)
        print()
        for i in self.interactions:
            print(f"  {i.qa_id}  similarity={i.similarity_score:.2f}")
        print()


class VisionHallucinationInteraction(BaseModel):
    qa_id: str
    similarity_score: float
    is_hallucination: bool


class VisionHallucinationMetric(BaseMetric):
    hallucination_rate: float
    n_hallucinations: int
    n_frames: int
    threshold: float
    summary: str
    interactions: list[VisionHallucinationInteraction]

    def display(self) -> None:
        """Print a human-readable summary of the hallucination results."""
        print(f"Session: {self.session_id}  |  Assistant: {self.assistant_id}")
        print(self.summary)
        print()
        for i in self.interactions:
            label = "HALLUCINATION" if i.is_hallucination else "ok"
            print(f"  {i.qa_id}  similarity={i.similarity_score:.2f}  {label}")
        print()
