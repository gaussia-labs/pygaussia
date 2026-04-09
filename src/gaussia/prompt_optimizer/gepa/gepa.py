"""GEPA: Generative Evolutionary Prompt Adaptation optimizer."""

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from gaussia.prompt_optimizer.base import BaseOptimizer
from gaussia.prompt_optimizer.gepa.prompts import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT
from gaussia.prompt_optimizer.protocols import Evaluator, Executor
from gaussia.prompt_optimizer.schemas import (
    CandidateResult,
    FailingExample,
    IterationResult,
    OptimizationResult,
)

if TYPE_CHECKING:
    from gaussia.core.retriever import Retriever


class _CandidatePrompts(BaseModel):
    candidates: list[str] = Field(description="List of improved system prompt candidates")


class GEPAOptimizer(BaseOptimizer):
    """Prompt optimizer using GEPA (Generative Evolutionary Prompt Adaptation).

    Iteratively evaluates a system prompt against a dataset, identifies failures,
    and generates improved candidates via an LLM until the score converges or
    the iteration budget is exhausted.

    Usage:
        result = GEPAOptimizer.run(
            retriever=MyRetriever,
            model=ChatGroq(model="llama-3.3-70b-versatile"),
            seed_prompt="You are a helpful assistant.",
            executor=lambda prompt, query, context: my_agent.call(prompt, query, context),
            objective="Answer questions accurately using only the provided context.",
        )
        print(result.optimized_prompt)

    For structured/deterministic evaluation, pass a custom evaluator:
        result = GEPAOptimizer.run(..., evaluator=my_evaluator_fn)
    """

    def __init__(
        self,
        retriever: type["Retriever"],
        model: BaseChatModel,
        seed_prompt: str,
        objective: str,
        executor: Executor | None = None,
        evaluator: Evaluator | None = None,
        iterations: int = 5,
        candidates_per_iteration: int = 3,
        failure_threshold: float = 0.6,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        from gaussia.prompt_optimizer.evaluators import LLMEvaluator

        self.model = model
        self.seed_prompt = seed_prompt
        self.objective = objective
        self.executor = executor or self._default_executor()
        self.evaluator = evaluator or LLMEvaluator(model=model, criteria=objective)
        self.iterations = iterations
        self.candidates_per_iteration = candidates_per_iteration
        self.failure_threshold = failure_threshold

    def _default_executor(self) -> Executor:
        model = self.model

        def _execute(prompt: str, query: str, context: str) -> str:
            system = f"{prompt}\n\n{context}" if context else prompt
            response = model.invoke([SystemMessage(content=system), HumanMessage(content=query)])
            return str(response.content)

        return _execute

    def _evaluate_prompt(self, prompt: str) -> tuple[float, list[FailingExample]]:
        scores: list[float] = []
        failing: list[FailingExample] = []

        for dataset in self.dataset:
            for batch in dataset.conversation:
                actual = self.executor(prompt, batch.query, dataset.context)
                score = self.evaluator(actual, batch.ground_truth_assistant, batch.query, dataset.context)
                scores.append(score)

                if score < self.failure_threshold:
                    failing.append(
                        FailingExample(
                            query=batch.query,
                            context=dataset.context,
                            expected=batch.ground_truth_assistant,
                            actual=actual,
                            score=score,
                        )
                    )

        aggregate = sum(scores) / len(scores) if scores else 0.0
        return aggregate, failing

    def _generate_candidates(self, current_prompt: str, failing: list[FailingExample]) -> list[str]:
        examples_text = self._format_failing_examples(failing)
        messages = [
            SystemMessage(content=GENERATION_SYSTEM_PROMPT),
            HumanMessage(
                content=GENERATION_USER_PROMPT.format(
                    objective=self.objective,
                    current_prompt=current_prompt,
                    failing_examples=examples_text,
                    n=self.candidates_per_iteration,
                )
            ),
        ]
        structured_model = self.model.with_structured_output(_CandidatePrompts)
        result: _CandidatePrompts = structured_model.invoke(messages)  # type: ignore[assignment]
        return result.candidates[: self.candidates_per_iteration]

    def _format_failing_examples(self, failing: list[FailingExample]) -> str:
        lines = []
        for i, ex in enumerate(failing[:10], 1):
            lines.append(
                f"Example {i}:\n"
                f"  Query: {ex.query}\n"
                f"  Expected: {ex.expected}\n"
                f"  Actual: {ex.actual}\n"
                f"  Score: {ex.score:.2f}"
            )
        return "\n\n".join(lines)

    def _optimize(self) -> OptimizationResult:
        n_examples = sum(len(d.conversation) for d in self.dataset)
        print(f"Evaluating seed prompt on {n_examples} examples...")
        current_score, failing = self._evaluate_prompt(self.seed_prompt)
        initial_score = current_score
        current_prompt = self.seed_prompt
        history: list[IterationResult] = []

        with tqdm(total=self.iterations, desc="Optimizing prompt") as pbar:
            for i in range(self.iterations):
                if not failing:
                    break

                raw_candidates = self._generate_candidates(current_prompt, failing)
                evaluated = [(cp, *self._evaluate_prompt(cp)) for cp in raw_candidates]

                candidates = [CandidateResult(prompt=cp, score=score) for cp, score, _ in evaluated]
                best_cp, best_score, best_failing = max(evaluated, key=lambda x: x[1])

                history.append(
                    IterationResult(
                        iteration=i + 1,
                        best_prompt=best_cp,
                        best_score=best_score,
                        candidates=candidates,
                        failing_examples=failing,
                    )
                )

                pbar.update(1)
                pbar.set_postfix({"best": f"{best_score:.2f}", "failing": len(best_failing)})

                if best_score > current_score:
                    current_prompt = best_cp
                    current_score = best_score
                    failing = best_failing
                else:
                    break

        return OptimizationResult(
            optimized_prompt=current_prompt,
            initial_score=initial_score,
            final_score=current_score,
            iterations_run=len(history),
            n_examples=n_examples,
            history=history,
        )
