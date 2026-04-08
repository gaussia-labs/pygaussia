"""Context metric for evaluating AI response alignment with provided context."""

from langchain_core.language_models.chat_models import BaseChatModel

from pygaussia.core import Gaussia, Retriever
from pygaussia.llm import ContextJudgeOutput, Judge
from pygaussia.llm.prompts import (
    context_reasoning_system_prompt,
    context_reasoning_system_prompt_observation,
)
from pygaussia.schemas import Batch
from pygaussia.schemas.context import ContextInteraction, ContextMetric
from pygaussia.statistical import FrequentistMode, StatisticalMode


class Context(Gaussia):
    """Metric for evaluating how well AI responses align with provided context.

    Accumulates per-interaction context_awareness scores across the session and
    emits one session-level ContextMetric in on_process_complete(). The score is
    aggregated using the configured StatisticalMode — frequentist returns a
    weighted mean, Bayesian returns a bootstrapped credible interval.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        statistical_mode: Statistical computation mode (defaults to FrequentistMode)
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        **kwargs: Additional arguments passed to Gaussia base class
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        statistical_mode: StatisticalMode | None = None,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.model = model
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self._session_data: dict[str, dict] = {}

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            strict=self.strict,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )

        if session_id not in self._session_data:
            self._session_data[session_id] = {
                "assistant_id": assistant_id,
                "batches": [],
                "scores": [],
                "interactions": [],
            }

        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            data = {"context": context, "assistant_answer": interaction.assistant}

            if interaction.observation:
                reasoning, result = judge.check(
                    context_reasoning_system_prompt_observation,
                    interaction.query,
                    {"observation": interaction.observation, **data},
                    output_schema=ContextJudgeOutput,
                )
            else:
                reasoning, result = judge.check(
                    context_reasoning_system_prompt,
                    interaction.query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                    output_schema=ContextJudgeOutput,
                )

            if result is None:
                raise ValueError(f"[FAIR FORGE/CONTEXT] No valid response from judge for QA ID: {interaction.qa_id}")

            score = float(result["score"] if isinstance(result, dict) else result.score)
            insight = result["insight"] if isinstance(result, dict) else result.insight
            self.logger.debug(f"Context insight: {insight}")
            if reasoning:
                self.logger.debug(f"Context reasoning: {reasoning}")
            self.logger.debug(f"Context awareness: {score}")
            self._session_data[session_id]["batches"].append(interaction)
            self._session_data[session_id]["scores"].append(score)
            self._session_data[session_id]["interactions"].append(
                ContextInteraction(qa_id=interaction.qa_id, context_awareness=score)
            )

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            batches = data["batches"]
            weights = self._resolve_weights(batches)

            mean, ci_low, ci_high = self._aggregate_scores(data["scores"], batches, weights, self.statistical_mode)

            metric = ContextMetric(
                session_id=session_id,
                assistant_id=data["assistant_id"],
                n_interactions=len(batches),
                context_awareness=mean,
                context_awareness_ci_low=ci_low,
                context_awareness_ci_high=ci_high,
                interactions=data["interactions"],
            )
            self.metrics.append(metric)
