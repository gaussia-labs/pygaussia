"""Role adherence metric schemas."""

from pydantic import BaseModel

from .metrics import BaseMetric


class RoleAdherenceTurn(BaseModel):
    qa_id: str
    adherence_score: float
    adherent: bool
    reason: str | None = None


class RoleAdherenceMetric(BaseMetric):
    """Session-level role adherence metric aggregating per-turn adherence scores."""

    n_turns: int
    role_adherence: float
    role_adherence_ci_low: float | None = None
    role_adherence_ci_high: float | None = None
    adherent: bool
    turns: list[RoleAdherenceTurn]
