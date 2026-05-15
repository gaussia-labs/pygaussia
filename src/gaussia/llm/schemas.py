"""Pydantic schemas for LLM structured outputs."""

from pydantic import BaseModel, ConfigDict, Field


class ContextJudgeOutput(BaseModel):
    """Structured output for context evaluation."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1, description="Context alignment score (0-1)")
    insight: str = Field(description="Insight about context compliance")


class ConversationalJudgeOutput(BaseModel):
    """Structured output for conversational evaluation."""

    model_config = ConfigDict(extra="forbid")

    memory: float = Field(ge=0, le=10, description="Memory recall score (0-10)")
    language: float = Field(ge=0, le=10, description="Language appropriateness score (0-10)")
    insight: str = Field(description="Overall insight about conversation quality")
    quality_maxim: float = Field(ge=0, le=10, description="Grice quality maxim score (0-10)")
    quantity_maxim: float = Field(ge=0, le=10, description="Grice quantity maxim score (0-10)")
    relation_maxim: float = Field(ge=0, le=10, description="Grice relation maxim score (0-10)")
    manner_maxim: float = Field(ge=0, le=10, description="Grice manner maxim score (0-10)")
    sensibleness: float = Field(ge=0, le=10, description="Sensibleness score (0-10)")


class BestOfJudgeOutput(BaseModel):
    """Structured output for best-of evaluation."""

    model_config = ConfigDict(extra="forbid")

    winner: str = Field(description="Winner identifier or 'tie'")
    verdict: str = Field(description="Explanation of why this contestant won")
    confidence: float = Field(ge=0, le=1, description="Confidence in the decision (0-1)")
    reasoning: dict = Field(description="Strengths and weaknesses for each contestant")
