"""Conversational metric for evaluating dialogue quality using Grice's maxims."""

from langchain_core.language_models.chat_models import BaseChatModel

from pygaussia.core import Gaussia, Retriever
from pygaussia.llm import ConversationalJudgeOutput, Judge
from pygaussia.llm.prompts import (
    conversational_reasoning_system_prompt,
    conversational_reasoning_system_prompt_observation,
)
from pygaussia.schemas import Batch
from pygaussia.schemas.conversational import ConversationalInteraction, ConversationalMetric, ConversationalScore
from pygaussia.statistical import FrequentistMode, StatisticalMode


class Conversational(Gaussia):
    """Metric for evaluating conversational quality using Grice's maxims.

    Accumulates per-interaction judge scores across the session and emits one
    session-level ConversationalMetric in on_process_complete(). Each dimension
    (memory, language, quality_maxim, etc.) is aggregated using the configured
    StatisticalMode — frequentist returns a weighted mean, Bayesian returns a
    bootstrapped credible interval.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        statistical_mode: Statistical computation mode (defaults to FrequentistMode)
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        **kwargs: Additional arguments passed to Gaussia base class
    """

    _DIMENSIONS = (
        "memory",
        "language",
        "quality_maxim",
        "quantity_maxim",
        "relation_maxim",
        "manner_maxim",
        "sensibleness",
    )

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

    def _extract_scores(self, result) -> dict[str, float]:
        if isinstance(result, dict):
            return {dim: float(result[dim]) for dim in self._DIMENSIONS}
        return {dim: float(getattr(result, dim)) for dim in self._DIMENSIONS}

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
        )

        if session_id not in self._session_data:
            self._session_data[session_id] = {
                "assistant_id": assistant_id,
                "batches": [],
                "dimension_scores": {dim: [] for dim in self._DIMENSIONS},
                "interactions": [],
            }

        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            data = {"preferred_language": language, "assistant_answer": interaction.assistant}

            if interaction.observation:
                reasoning, result = judge.check(
                    conversational_reasoning_system_prompt_observation,
                    interaction.query,
                    {"observation": interaction.observation, **data},
                    output_schema=ConversationalJudgeOutput,
                )
            else:
                reasoning, result = judge.check(
                    conversational_reasoning_system_prompt,
                    interaction.query,
                    {"ground_truth_assistant": interaction.assistant, **data},
                    output_schema=ConversationalJudgeOutput,
                )

            if result is None:
                raise ValueError(
                    f"[FAIR FORGE/CONVERSATIONAL] No valid response from judge for QA ID: {interaction.qa_id}"
                )

            insight = result["insight"] if isinstance(result, dict) else result.insight
            self.logger.debug(f"Conversational insight: {insight}")
            if reasoning:
                self.logger.debug(f"Conversational reasoning: {reasoning}")
            scores = self._extract_scores(result)
            self._session_data[session_id]["batches"].append(interaction)
            for dim, score in scores.items():
                self._session_data[session_id]["dimension_scores"][dim].append(score)
            self._session_data[session_id]["interactions"].append(
                ConversationalInteraction(qa_id=interaction.qa_id, **scores)
            )

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            batches = data["batches"]
            weights = self._resolve_weights(batches)

            dimension_metrics = {}
            for dim in self._DIMENSIONS:
                mean, ci_low, ci_high = self._aggregate_scores(
                    data["dimension_scores"][dim], batches, weights, self.statistical_mode
                )
                dimension_metrics[dim] = ConversationalScore(mean=mean, ci_low=ci_low, ci_high=ci_high)

            metric = ConversationalMetric(
                session_id=session_id,
                assistant_id=data["assistant_id"],
                n_interactions=len(batches),
                conversational_memory=dimension_metrics["memory"],
                conversational_language=dimension_metrics["language"],
                conversational_quality_maxim=dimension_metrics["quality_maxim"],
                conversational_quantity_maxim=dimension_metrics["quantity_maxim"],
                conversational_relation_maxim=dimension_metrics["relation_maxim"],
                conversational_manner_maxim=dimension_metrics["manner_maxim"],
                conversational_sensibleness=dimension_metrics["sensibleness"],
                interactions=data["interactions"],
            )
            self.metrics.append(metric)
