"""Role adherence metric schemas."""

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class RoleAdherenceJudgeOutput(BaseModel):
    """Structured-output schema for the role adherence judge."""

    adherent: bool = Field(
        description="True if the assistant's response adheres to its role (YES), "
        "False if it violates the role (NO)."
    )


class RoleAdherenceTurn(BaseModel):
    qa_id: str
    adherence_score: float
    adherent: bool


class RoleAdherenceMetric(BaseMetric):
    """Session-level role adherence metric aggregating per-turn adherence scores."""

    n_turns: int
    role_adherence: float
    role_adherence_ci_low: float | None = None
    role_adherence_ci_high: float | None = None
    adherent: bool
    turns: list[RoleAdherenceTurn]
