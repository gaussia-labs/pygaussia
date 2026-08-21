"""MIPROv2: Multiprompt Instruction PRoposal Optimizer v2."""

import random
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm.auto import tqdm

from gaussia.prompt_optimizer.base import BaseOptimizer
from gaussia.prompt_optimizer.mipro.proposer import DemoBootstrapper, InstructionProposer
from gaussia.prompt_optimizer.protocols import Evaluator, Executor
from gaussia.prompt_optimizer.schemas import Demo, MIPROv2Result, TrialResult

if TYPE_CHECKING:
    from gaussia.core.retriever import Retriever
    from gaussia.schemas.common import Batch


class MIPROv2Optimizer(BaseOptimizer):
    """Prompt optimizer using MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2).

    Combines instruction proposal with few-shot demo bootstrapping, then uses
    Bayesian Optimization (Optuna/TPE) to find the best (instruction, demo_set) combination.

    Usage:
        result = MIPROv2Optimizer.run(
            retriever=MyRetriever,
            model=ChatGroq(model="llama-3.3-70b-versatile"),
            seed_prompt="You are a helpful assistant.",
            objective="Answer questions using only the provided context.",
        )
        print(result.optimized_prompt)
    """

    def __init__(
        self,
        retriever: type["Retriever"],
        model: BaseChatModel,
        seed_prompt: str,
        objective: str,
        executor: Executor | None = None,
        evaluator: Evaluator | None = None,
        num_candidates: int = 10,
        num_trials: int = 20,
        minibatch_size: int = 25,
        max_demos_per_set: int = 3,
        num_demo_sets: int = 5,
        random_seed: int = 42,
        tips: list[str] | None = None,
        instruction_proposal_system: str | None = None,
        instruction_proposal_user: str | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        from gaussia.prompt_optimizer.evaluators import LLMEvaluator

        self._model = model
        self._seed_prompt = seed_prompt
        self._objective = objective
        self._executor = executor or self._default_executor()
        self._evaluator = evaluator or LLMEvaluator(model=model, criteria=objective)
        self._num_candidates = num_candidates
        self._num_trials = num_trials
        self._minibatch_size = minibatch_size
        self._max_demos_per_set = max_demos_per_set
        self._num_demo_sets = num_demo_sets
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        self._tips = tips
        self._instruction_proposal_system = instruction_proposal_system
        self._instruction_proposal_user = instruction_proposal_user

    def _default_executor(self) -> Executor:
        model = self._model

        def _execute(prompt: str, query: str, context: str) -> str:
            system = f"{prompt}\n\n{context}" if context else prompt
            return str(model.invoke([SystemMessage(content=system), HumanMessage(content=query)]).content)

        return _execute

    def _collect_examples(self) -> list[tuple[str, "Batch"]]:
        return [
            (dataset.context, batch)
            for dataset in self.dataset
            for batch in dataset.conversation
        ]

    def _score_examples(self, prompt: str, examples: list[tuple[str, "Batch"]]) -> float:
        scores = [
            self._evaluator(
                self._executor(prompt, batch.query, context),
                batch.ground_truth_assistant,
                batch.query,
                context,
            )
            for context, batch in examples
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _evaluate_prompt(self, prompt: str, minibatch: bool = False) -> float:
        examples = self._collect_examples()
        if minibatch and len(examples) > self._minibatch_size:
            examples = self._rng.sample(examples, self._minibatch_size)
        return self._score_examples(prompt, examples)

    def _build_prompt(self, instruction: str, demos: list[Demo]) -> str:
        if not demos:
            return instruction
        examples = "\n\n".join(f"User: {d.query}\nAssistant: {d.response}" for d in demos)
        return f"{instruction}\n\nExamples:\n{examples}"

    def _optimize(self) -> MIPROv2Result:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as err:
            raise ImportError(
                "MIPROv2Optimizer requires optuna. Install it with: pip install optuna"
            ) from err

        n_examples = sum(len(d.conversation) for d in self.dataset)
        print(f"Evaluating seed prompt on {n_examples} examples...")
        initial_score = self._evaluate_prompt(self._seed_prompt)

        print(f"Generating {self._num_candidates} instruction candidates...")
        instructions = InstructionProposer(
            model=self._model,
            seed_prompt=self._seed_prompt,
            objective=self._objective,
            num_candidates=self._num_candidates,
            tips=self._tips,
            system_prompt=self._instruction_proposal_system,
            user_prompt=self._instruction_proposal_user,
        ).propose()

        demo_sets = DemoBootstrapper(
            dataset=self.dataset,
            num_demo_sets=self._num_demo_sets,
            max_demos_per_set=self._max_demos_per_set,
            random_seed=self._random_seed,
        ).bootstrap()

        trial_results: list[TrialResult] = []

        with tqdm(total=self._num_trials, desc="Evaluating trials") as pbar:

            def objective_fn(trial: "optuna.Trial") -> float:
                instruction_idx = trial.suggest_int("instruction_idx", 0, len(instructions) - 1)
                demo_set_idx = trial.suggest_int("demo_set_idx", 0, len(demo_sets) - 1)
                prompt = self._build_prompt(instructions[instruction_idx], demo_sets[demo_set_idx])
                score = self._evaluate_prompt(prompt, minibatch=True)
                trial_results.append(
                    TrialResult(
                        trial=trial.number,
                        instruction_idx=instruction_idx,
                        demo_set_idx=demo_set_idx,
                        score=score,
                    )
                )
                best = max(t.score for t in trial_results)
                pbar.set_postfix({"best": f"{best:.2f}"})
                pbar.update(1)
                return score

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self._random_seed),
            )
            study.optimize(objective_fn, n_trials=self._num_trials)

        best = study.best_trial
        best_instruction = instructions[best.params["instruction_idx"]]
        best_demos = demo_sets[best.params["demo_set_idx"]]

        return MIPROv2Result(
            optimized_prompt=self._build_prompt(best_instruction, best_demos),
            optimized_instruction=best_instruction,
            initial_score=initial_score,
            final_score=study.best_value,
            iterations_run=len(study.trials),
            n_examples=n_examples,
            demos=best_demos,
            trials=trial_results,
        )
