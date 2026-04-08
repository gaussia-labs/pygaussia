"""Protocols defining the interfaces required by prompt optimizers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Executor(Protocol):
    """Runs the AI system under optimization with a given system prompt."""

    def __call__(self, prompt: str, query: str, context: str) -> str: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores an AI response against the expected output. Returns a float in [0.0, 1.0]."""

    def __call__(self, actual: str, expected: str, query: str, context: str) -> float: ...
