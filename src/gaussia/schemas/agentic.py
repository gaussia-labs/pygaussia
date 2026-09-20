"""Agentic metric schemas."""

from pydantic import BaseModel, Field

from .metrics import BaseMetric


class ToolScope(BaseModel):
    """
    Access-boundary verdict for one interaction: did the agent call a tool it was not
    permitted to call?

    This is not a graded version of tool correctness. Tool correctness asks how far the
    agent strayed from an expected plan; this asks whether it crossed a declared boundary.
    An agent can be perfectly plan-adherent while crossing a boundary, and can stray from
    the plan without crossing one, so neither score is derivable from the other.

    ``scope_violation`` is binary by design: a boundary is either crossed or it is not, and
    a fraction would dilute a single forbidden call among many permitted ones — the precise
    failure that makes an inverted correctness score unusable as a violation signal.
    Per-tool severity is deliberately left to the consumer, which has the tool names in
    ``out_of_scope_tools`` and its own policy for weighting them.

    Note the polarity is inverted with respect to every other field in this module: here
    ``1.0`` is the bad outcome.
    """

    scope_violation: float | None = Field(default=None, ge=0.0, le=1.0)
    """``1.0`` crossed, ``0.0`` respected, ``None`` no boundary declared.

    ``None`` and ``0.0`` are not interchangeable: one says the dataset declared no scope, the
    other says it declared one and the agent stayed inside it. Collapsing them into ``0.0``
    would report a clean security record for datasets that were never evaluated at all —
    understating risk exactly where the data is incomplete.
    """

    out_of_scope_tools: list[str] = Field(default_factory=list)
    """The tools that crossed the boundary — the evidence behind ``scope_violation``."""


class ToolCorrectnessScore(BaseModel):
    """
    Evaluation scores for tool usage correctness.

    Evaluates four aspects: tool selection (correct tools chosen), parameter accuracy
    (correct parameters passed), sequence (correct order if required), and utilization
    (tool results used in final answer). Overall score is weighted average.

    ``scope_violation`` and ``out_of_scope_tools`` carry the access-boundary verdict from
    ``ToolScope`` and are deliberately **not** part of ``overall_correctness``: averaging a
    permission signal into a plan-adherence signal makes both unreadable. Consumers that
    only care about plan adherence keep reading the four components and
    ``overall_correctness`` exactly as before.
    """

    tool_selection_correct: float = Field(ge=0.0, le=1.0)
    parameter_accuracy: float = Field(ge=0.0, le=1.0)
    sequence_correct: float = Field(ge=0.0, le=1.0)
    result_utilization: float = Field(ge=0.0, le=1.0)
    overall_correctness: float = Field(ge=0.0, le=1.0)
    is_correct: bool
    reasoning: str | None = None
    scope_violation: float | None = Field(default=None, ge=0.0, le=1.0)
    out_of_scope_tools: list[str] = Field(default_factory=list)


class AgenticMetric(BaseMetric):
    """
    Metric for evaluating complete agent conversations with pass@K and tool correctness.

    Evaluates conversations as complete units where a conversation is correct only if
    ALL its interactions are correct. This measures the agent's capability to maintain
    fully correct conversations.

    pass@K: Probability of ≥1 correct conversation when attempting k different conversations (0.0-1.0).
    pass^K: Probability of k consecutive correct conversations (0.0-1.0).
    tool_correctness: Optional evaluation of tool usage quality per interaction.

    Every field derived from answer correctness is ``None`` when ``Agentic`` ran without a
    judge model, because nothing measured it. A ``0.0`` there would read as "the assistant
    failed every interaction" when the truth is "no one asked" — the same conflation of a
    missing measurement with a bad one that ``ToolScope.scope_violation`` avoids. The
    ``None`` only ever appears in that mode, so callers that pass a model see the types they
    always saw.
    """

    session_id: str  # Unique conversation ID
    total_interactions: int  # Number of interactions in the conversation
    correct_interactions: int | None  # Number of correct interactions (None = not measured)
    is_fully_correct: bool | None  # True if ALL interactions are correct (None = not measured)
    threshold: float  # Threshold for answer correctness
    correctness_scores: list[float] | None  # Score per interaction (None = not measured)
    correct_indices: list[int] | None  # Indices of correct interactions (None = not measured)
    tool_correctness_scores: list[ToolCorrectnessScore | None] = []  # Tool scores per interaction
    k: int = 0
    pass_at_k: float | None = 0.0
    pass_at_k_ci_low: float | None = None
    pass_at_k_ci_high: float | None = None
    pass_pow_k: float | None = 0.0
    pass_pow_k_ci_low: float | None = None
    pass_pow_k_ci_high: float | None = None
