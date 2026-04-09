"""Pre-built evaluators for the GEPA prompt optimizer."""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


class LLMEvaluator:
    """Evaluates AI responses using an LLM judge with a natural language criteria.

    Drop-in replacement for a custom evaluator function. The user describes
    what a good response looks like in plain language — no scoring code needed.

    Usage:
        evaluator = LLMEvaluator(
            model=ChatGroq(model="llama-3.3-70b-versatile"),
            criteria="The response must answer using only the provided context, without hallucinating.",
        )
    """

    def __init__(self, model: BaseChatModel, criteria: str):
        self._model = model
        self._criteria = criteria

    def __call__(self, actual: str, expected: str, query: str, context: str) -> float:
        system = (
            f"You are a response quality evaluator.\n\n"
            f"Criteria: {self._criteria}\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            f"Expected response: {expected}\n"
            f"Actual response: {actual}\n\n"
            'Score the response. Return ONLY a JSON object like: {"score": 0.85}'
        )
        response = self._model.invoke([
            SystemMessage(content=system),
            HumanMessage(content="Provide the score."),
        ])
        return self._parse_score(str(response.content))

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.0
