"""Agentic metric schemas."""

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class ToolCorrectnessScore(BaseModel):
    """
    Evaluation scores for tool usage correctness.

    Evaluates four aspects: tool selection (correct tools chosen), parameter accuracy
    (correct parameters passed), sequence (correct order if required), and utilization
    (tool results used in final answer). Overall score is weighted average.
    """

    tool_selection_correct: float = Field(ge=0.0, le=1.0)
    parameter_accuracy: float = Field(ge=0.0, le=1.0)
    sequence_correct: float = Field(ge=0.0, le=1.0)
    result_utilization: float = Field(ge=0.0, le=1.0)
    overall_correctness: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    reasoning: str | None = None


class AgenticMetric(BaseMetric):
    """
    Metric for evaluating complete agent conversations with pass@K and tool correctness.

    Evaluates conversations as complete units where a conversation is correct only if
    ALL its interactions are correct. This measures the agent's capability to maintain
    fully correct conversations.

    pass@K: Probability of ≥1 correct conversation when attempting k different conversations (0.0-1.0).
    pass^K: Probability of k consecutive correct conversations (0.0-1.0).
    tool_correctness: Optional evaluation of tool usage quality per interaction.
    """

    session_id: str  # Unique conversation ID
    total_interactions: int  # Number of interactions in the conversation
    correct_interactions: int  # Number of correct interactions
    is_fully_correct: bool  # True if ALL interactions are correct
    threshold: float  # Threshold for answer correctness
    correctness_scores: list[float]  # Score per interaction
    correct_indices: list[int]  # Indices of correct interactions
    tool_correctness_scores: list[ToolCorrectnessScore | None] = []  # Tool scores per interaction
    k: int = 0
    pass_at_k: float = 0.0
    pass_at_k_ci_low: float | None = None
    pass_at_k_ci_high: float | None = None
    pass_pow_k: float = 0.0
    pass_pow_k_ci_low: float | None = None
    pass_pow_k_ci_high: float | None = None
