"""Shared schemas for prompt optimization results."""

from pydantic import BaseModel


class FailingExample(BaseModel):
    query: str
    context: str
    expected: str
    actual: str
    score: float


class CandidateResult(BaseModel):
    prompt: str
    score: float


class IterationResult(BaseModel):
    iteration: int
    best_prompt: str
    best_score: float
    candidates: list[CandidateResult]
    failing_examples: list[FailingExample]


class OptimizationResult(BaseModel):
    optimized_prompt: str
    initial_score: float
    final_score: float
    iterations_run: int
    n_examples: int
    history: list[IterationResult] = []


class Demo(BaseModel):
    query: str
    response: str


class TrialResult(BaseModel):
    trial: int
    instruction_idx: int
    demo_set_idx: int
    score: float


class MIPROv2Result(OptimizationResult):
    optimized_instruction: str
    demos: list[Demo] = []
    trials: list[TrialResult] = []
