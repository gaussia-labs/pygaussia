"""BestOf metric schemas."""

from pydantic import BaseModel

from .metrics import BaseMetric


class BestOfContest(BaseModel):
    """
    A single contest within the best-of tournament bracket.
    """

    round: int
    left_id: str
    right_id: str
    winner_id: str
    confidence: float | None = None
    verdict: str | None = None
    reasoning: dict | None = None
    thinkings: str | None = None


class BestOfMetric(BaseMetric):
    """
    Best-of tournament metric capturing the final winner and all contests that led to it.
    """

    bestof_winner_id: str
    bestof_contests: list[BestOfContest]
    qa_id: str
