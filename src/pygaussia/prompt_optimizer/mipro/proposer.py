"""Instruction proposal and demo bootstrapping for MIPROv2."""

import random

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pygaussia.prompt_optimizer.mipro.prompts import INSTRUCTION_PROPOSAL_SYSTEM, INSTRUCTION_PROPOSAL_USER
from pygaussia.prompt_optimizer.schemas import Demo

_TIPS = [
    "Be explicit about what the model should NOT do.",
    "Specify the expected output format and structure.",
    "Define the tone and communication style.",
    "Emphasize grounding responses in the provided context only.",
    "Prioritize conciseness — fewer words, more precision.",
    "Address edge cases and ambiguous or out-of-scope queries.",
    "Make the constraints actionable and specific.",
    "Emphasize what differentiates an excellent response from a mediocre one.",
    "Focus on the user's underlying need, not just the literal question.",
    "Be explicit about when to say you don't know.",
]


class _InstructionVariants(BaseModel):
    instructions: list[str] = Field(description="List of system prompt variants, one per focus area")


class InstructionProposer:
    def __init__(
        self,
        model: BaseChatModel,
        seed_prompt: str,
        objective: str,
        num_candidates: int,
        tips: list[str] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        self._model = model
        self._seed_prompt = seed_prompt
        self._objective = objective
        self._num_candidates = num_candidates
        self._tips = tips or _TIPS
        self._system_prompt = system_prompt or INSTRUCTION_PROPOSAL_SYSTEM
        self._user_prompt = user_prompt or INSTRUCTION_PROPOSAL_USER

    def propose(self) -> list[str]:
        tips = [self._tips[i % len(self._tips)] for i in range(self._num_candidates)]
        tips_text = "\n".join(f"{i + 1}. {tip}" for i, tip in enumerate(tips))
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=self._user_prompt.format(
                    seed_prompt=self._seed_prompt,
                    objective=self._objective,
                    n=self._num_candidates,
                    tips=tips_text,
                )
            ),
        ]
        structured = self._model.with_structured_output(_InstructionVariants)
        result: _InstructionVariants = structured.invoke(messages)  # type: ignore[assignment]
        return result.instructions[: self._num_candidates]


class DemoBootstrapper:
    def __init__(self, dataset: list, num_demo_sets: int, max_demos_per_set: int, random_seed: int):
        self._dataset = dataset
        self._num_demo_sets = num_demo_sets
        self._max_demos_per_set = max_demos_per_set
        self._rng = random.Random(random_seed)

    def bootstrap(self) -> list[list[Demo]]:
        all_demos = [
            Demo(query=batch.query, response=batch.ground_truth_assistant)
            for dataset in self._dataset
            for batch in dataset.conversation
        ]
        n = min(self._max_demos_per_set, len(all_demos))
        return [self._rng.sample(all_demos, n) for _ in range(self._num_demo_sets)]
