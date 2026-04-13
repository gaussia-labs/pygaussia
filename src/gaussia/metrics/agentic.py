"""Agentic metric for evaluating agent responses with pass@K and tool correctness."""

from collections import defaultdict
from typing import Any

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from gaussia.core import Gaussia, Retriever
from gaussia.llm import Judge
from gaussia.schemas import Batch
from gaussia.schemas.agentic import AgenticMetric, ToolCorrectnessScore
from gaussia.statistical import FrequentistMode, StatisticalMode


class AnswerCorrectnessOutput(BaseModel):
    """Structured output for answer correctness evaluation."""

    model_config = ConfigDict(extra="forbid")

    correctness_score: float = Field(ge=0.0, le=1.0, description="Correctness score (0.0-1.0)")
    reasoning: str = Field(description="Brief explanation of the evaluation")


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k: probability of ≥1 correct conversation in k independent attempts.

    Uses the Bernoulli model: p = c/n is the estimated success rate from evaluation,
    and 1 - (1-p)^k is the probability of at least one success in k independent attempts.

    Args:
        n: Total conversations evaluated
        c: Fully correct conversations
        k: Number of independent attempts (not bounded by n)

    Returns:
        Probability between 0.0 and 1.0
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - (1.0 - c / n) ** k


def pass_pow_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass^k: probability of k consecutive correct conversations.

    Uses p = c/n as the estimated success rate: (c/n)^k is the probability
    that k independent attempts are all correct.

    Args:
        n: Total conversations evaluated
        c: Fully correct conversations
        k: Number of consecutive attempts (not bounded by n)

    Returns:
        Probability between 0.0 and 1.0
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    return (c / n) ** k


class Agentic(Gaussia):
    """
    Agentic metric for evaluating complete agent conversations with pass@K/pass^K formulas.

    Evaluates conversations as complete units where a conversation is correct only if ALL
    its interactions are correct. This measures the agent's capability to maintain fully
    correct multi-turn conversations.

    Metrics:
    - pass@K: Probability of ≥1 correct conversation when attempting K different conversations (0.0-1.0)
    - pass^K: Probability of K consecutive correct conversations (0.0-1.0)
    - Tool Correctness: Evaluates correct tool usage per interaction (selection, parameters, sequence, utilization)

    Uses an LLM judge for answer correctness, and direct dictionary comparison for tool correctness.

    Formulas:
        pass@k = 1 - (1 - p)^k  # Prob. of ≥1 correct conversation, p = c/n
        pass^k = p^k             # Prob. of all k conversations correct

    Where:
        n = total conversations evaluated
        c = fully correct conversations (all interactions correct)
        p = c/n, estimated success rate
        k = number of conversation attempts (required, user-specified)

    Args:
        retriever: Retriever class for loading datasets (each Dataset = 1 conversation)
        model: LangChain BaseChatModel instance for evaluation
        k: Number of independent attempts for pass@K/pass^K computation (required)
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        threshold: Similarity threshold for answer correctness (default: 0.7)
        tool_threshold: Threshold for tool correctness (default: 1.0)
        tool_weights: Weights for tool correctness components (default: 0.25 each)
        **kwargs: Additional arguments passed to Gaussia base class

    Example:
        >>> from langchain_groq import ChatGroq
        >>> model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        >>> results = Agentic.run(MyRetriever, model=model, k=3, threshold=0.8)
        >>> for r in results:
        ...     print(f"{r.session_id}: pass@3={r.pass_at_k:.3f}, pass^3={r.pass_pow_k:.3f}")
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        k: int,
        use_structured_output: bool = True,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        threshold: float = 0.7,
        tool_threshold: float = 1.0,
        tool_weights: dict[str, float] | None = None,
        statistical_mode: StatisticalMode | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)

        self.model = model
        self.k = k
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.threshold = threshold
        self.tool_threshold = tool_threshold

        if tool_weights is None:
            tool_weights = {
                "selection": 0.25,
                "parameters": 0.25,
                "sequence": 0.25,
                "utilization": 0.25,
            }
        self.tool_weights = tool_weights
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()

        self.logger.info(f"Initialized Agentic metric with model: {model.__class__.__name__}")
        self.logger.info(f"Thresholds - Answer: {threshold}, Tool: {tool_threshold}")
        self.logger.info(f"Statistical mode: {self.statistical_mode.get_result_type()}")

    @classmethod
    def run(cls, retriever: type[Retriever], **kwargs) -> list[AgenticMetric]:
        return cls(retriever, **kwargs)._process()

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        """Process batch - actual evaluation happens in _process() by grouping qa_ids."""
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}, Assistant: {assistant_id}")

    def _evaluate_answer_correctness(self, judge: Judge, query: str, answer: str, ground_truth: str) -> float:
        """Evaluate answer correctness using LLM judge. Returns score 0.0-1.0."""
        system_prompt = """You are a STRICT evaluator. Your task is to determine if an agent's answer is correct compared to the ground truth.

**Agent's Answer:** {answer}

**Ground Truth:** {ground_truth}

Evaluate the correctness with STRICT criteria:

1. **Factual Accuracy** (most important): Is the core information factually correct?
2. **Precision**: Spelling errors, typos, or incorrect formatting should be penalized
3. **Completeness**: Does it answer what was asked?
4. **Format**: Natural language variations are acceptable ONLY if facts are perfect

IMPORTANT SCORING RULES:
- 1.0: Identical or perfectly correct with natural rephrasing (same facts, perfect spelling)
- 0.85-0.95: Correct facts with slightly more verbose explanation
- 0.65-0.75: Correct core fact BUT has typo/spelling error (e.g., "Poris" instead of "Paris")
- 0.5-0.65: Mostly correct but missing important details
- 0.3-0.5: Partially correct with significant errors
- 0.0-0.3: Wrong answer or completely incorrect

**Typo/Spelling Penalty**:
- Single character typo in short answer: MAX score 0.75
- Multiple typos: MAX score 0.5
- Wrong word entirely: score below 0.3

**Wrong Answer**: Factually incorrect information must score below 0.3

Examples:
- Q: "Capital of France?", A: "Paris", GT: "Paris" → 1.0 (perfect match)
- Q: "Capital of France?", A: "The capital of France is Paris", GT: "Paris" → 0.95 (correct, verbose)
- Q: "Capital of France?", A: "Poris", GT: "Paris" → 0.7 (TYPO PENALTY - core fact known but misspelled)
- Q: "Capital of France?", A: "Pariis", GT: "Paris" → 0.7 (TYPO PENALTY)
- Q: "Capital of France?", A: "Lyon", GT: "Paris" → 0.0 (completely wrong city)
- Q: "Capital of France?", A: "London", GT: "Paris" → 0.0 (wrong country)
"""

        data = {"answer": answer, "ground_truth": ground_truth}

        try:
            _reasoning, result = judge.check(system_prompt, query, data, output_schema=AnswerCorrectnessOutput)

            self.logger.debug(f"Judge returned - reasoning: {_reasoning[:100] if _reasoning else 'None'}...")

            if result is None:
                self.logger.error("❌ Judge returned None - no valid JSON found in response")
                return 0.0

            self.logger.debug(f"✓ Judge result type: {type(result)}")
            self.logger.debug(f"✓ Judge result content: {result}")

            if isinstance(result, dict):
                score = float(result.get("correctness_score", 0.0))
                self.logger.debug(f"✓ Extracted score from dict: {score}")
                if score == 0.0 and "correctness_score" not in result:
                    self.logger.warning(f"⚠️  Dict missing 'correctness_score' key. Keys: {list(result.keys())}")
                return score

            score = float(result.correctness_score)
            self.logger.debug(f"✓ Extracted score from object: {score}")
            return score

        except Exception:
            self.logger.exception("❌ Error evaluating answer correctness")
            return 0.0

    def _evaluate_tool_correctness(
        self, agentic: dict[str, Any], ground_truth_agentic: dict[str, Any]
    ) -> ToolCorrectnessScore:
        """Evaluate tool usage correctness by comparing selection, parameters, sequence, and utilization."""
        tools_used = agentic.get("tools_used", [])
        final_answer_uses_tools = agentic.get("final_answer_uses_tools", False)
        expected_tools = ground_truth_agentic.get("expected_tools", [])
        sequence_matters = ground_truth_agentic.get("tool_sequence_matters", True)

        reasoning_parts = []

        used_tool_names = {tool.get("tool_name") for tool in tools_used}
        expected_tool_names = {tool.get("tool_name") for tool in expected_tools}

        if expected_tool_names == used_tool_names:
            tool_selection = 1.0
            reasoning_parts.append("✓ Tool selection: correct")
        elif used_tool_names.issubset(expected_tool_names):
            tool_selection = len(used_tool_names) / len(expected_tool_names)
            missing = expected_tool_names - used_tool_names
            reasoning_parts.append(f"⚠ Tool selection: missing {missing}")
        elif expected_tool_names.issubset(used_tool_names):
            tool_selection = len(expected_tool_names) / len(used_tool_names)
            extra = used_tool_names - expected_tool_names
            reasoning_parts.append(f"⚠ Tool selection: extra tools {extra}")
        else:
            overlap = len(used_tool_names.intersection(expected_tool_names))
            total = len(used_tool_names.union(expected_tool_names))
            tool_selection = overlap / total if total > 0 else 0.0
            reasoning_parts.append(f"✗ Tool selection: used {used_tool_names}, expected {expected_tool_names}")

        used_tools_map: dict[str, list] = defaultdict(list)
        expected_tools_map: dict[str, list] = defaultdict(list)

        for tool in tools_used:
            used_tools_map[tool.get("tool_name")].append(tool)
        for tool in expected_tools:
            expected_tools_map[tool.get("tool_name")].append(tool)

        param_matches = []
        for tool_name in expected_tool_names:
            if tool_name not in used_tools_map:
                param_matches.append(0.0)
                continue

            used_list = used_tools_map[tool_name]
            expected_list = expected_tools_map[tool_name]

            for exp_tool in expected_list:
                exp_params = exp_tool.get("parameters", {})
                used_tool = used_list[0] if used_list else {}
                used_params = used_tool.get("parameters", {})

                if exp_params == used_params:
                    param_matches.append(1.0)
                else:
                    all_keys = set(exp_params.keys()).union(used_params.keys())
                    matching = sum(1 for k in all_keys if exp_params.get(k) == used_params.get(k))
                    param_matches.append(matching / len(all_keys) if all_keys else 0.0)

        parameter_accuracy = sum(param_matches) / len(param_matches) if param_matches else 0.0

        if parameter_accuracy == 1.0:
            reasoning_parts.append("✓ Parameters: correct")
        elif parameter_accuracy > 0.7:
            reasoning_parts.append(f"⚠ Parameters: mostly correct ({parameter_accuracy:.2f})")
        else:
            reasoning_parts.append(f"✗ Parameters: incorrect ({parameter_accuracy:.2f})")

        if not sequence_matters:
            sequence_correct = 1.0
            reasoning_parts.append("✓ Sequence: not required")
        else:
            sequence_matches = []
            for tool_name in expected_tool_names:
                if tool_name not in used_tools_map:
                    sequence_matches.append(0.0)
                    continue

                used_list = used_tools_map[tool_name]
                expected_list = expected_tools_map[tool_name]

                for i, exp_tool in enumerate(expected_list):
                    exp_step = exp_tool.get("step")
                    used_tool = used_list[i] if i < len(used_list) else None
                    used_step = used_tool.get("step") if used_tool else None

                    if exp_step == used_step:
                        sequence_matches.append(1.0)
                    else:
                        sequence_matches.append(0.0)

            sequence_correct = sum(sequence_matches) / len(sequence_matches) if sequence_matches else 0.0

            if sequence_correct == 1.0:
                reasoning_parts.append("✓ Sequence: correct")
            else:
                reasoning_parts.append(f"✗ Sequence: incorrect ({sequence_correct:.2f})")

        if final_answer_uses_tools:
            result_utilization = 1.0
            reasoning_parts.append("✓ Utilization: tools used in answer")
        else:
            result_utilization = 0.0
            reasoning_parts.append("✗ Utilization: tools not used in answer")

        overall = (
            self.tool_weights["selection"] * tool_selection
            + self.tool_weights["parameters"] * parameter_accuracy
            + self.tool_weights["sequence"] * sequence_correct
            + self.tool_weights["utilization"] * result_utilization
        )

        is_correct = overall >= self.tool_threshold

        reasoning = "; ".join(reasoning_parts)

        return ToolCorrectnessScore(
            tool_selection_correct=tool_selection,
            parameter_accuracy=parameter_accuracy,
            sequence_correct=sequence_correct,
            result_utilization=result_utilization,
            overall_correctness=overall,
            is_correct=is_correct,
            reasoning=reasoning,
        )

    def _process(self) -> list[AgenticMetric]:
        """Evaluate each conversation (dataset) as a complete unit."""
        from gaussia.schemas.common import Dataset

        datasets = [d for d in self.dataset if isinstance(d, Dataset)]
        self.logger.info(f"[Agentic] Evaluating {len(datasets)} conversations")

        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            strict=self.strict,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )

        for dataset_idx, dataset in enumerate(datasets, 1):
            total_interactions = len(dataset.conversation)
            self.logger.info(
                f"[Agentic] Evaluating conversation {dataset_idx}/{len(datasets)}: "
                f"{dataset.session_id} ({total_interactions} interactions)"
            )

            correctness_scores: list[float] = []
            correct_indices: list[int] = []
            tool_correctness_scores: list[ToolCorrectnessScore | None] = []

            # Evaluate each interaction in the conversation
            for i, batch in enumerate(dataset.conversation):
                self.logger.debug(f"  Interaction {i + 1}/{total_interactions} (qa_id: {batch.qa_id})")

                # Evaluate answer correctness
                score = self._evaluate_answer_correctness(
                    judge=judge,
                    query=batch.query,
                    answer=batch.assistant,
                    ground_truth=batch.ground_truth_assistant,
                )

                correctness_scores.append(score)

                if score >= self.threshold:
                    correct_indices.append(i)
                    self.logger.debug(f"    Answer score: {score:.3f} ✅ CORRECT")
                else:
                    self.logger.debug(f"    Answer score: {score:.3f} ❌ INCORRECT")

                # Evaluate tool correctness if applicable
                if batch.ground_truth_agentic:
                    if batch.agentic and batch.agentic.get("tools_used"):
                        tool_correctness = self._evaluate_tool_correctness(
                            agentic=batch.agentic, ground_truth_agentic=batch.ground_truth_agentic
                        )
                        tool_correctness_scores.append(tool_correctness)
                        self.logger.debug(
                            f"    Tool correctness: {tool_correctness.overall_correctness:.3f}, "
                            f"Correct={tool_correctness.is_correct}"
                        )
                    else:
                        tool_correctness_scores.append(None)
                        self.logger.debug("    No tools used")
                else:
                    tool_correctness_scores.append(None)

            # Determine if conversation is fully correct
            correct_interactions = len(correct_indices)
            is_fully_correct = correct_interactions == total_interactions

            status = (
                "✅ FULLY CORRECT" if is_fully_correct else f"❌ PARTIAL ({correct_interactions}/{total_interactions})"
            )
            self.logger.info(f"  Conversation result: {status}")

            p_result = self.statistical_mode.rate_estimation(correct_interactions, total_interactions)

            if self.statistical_mode.get_result_type() == "point_estimate":
                metric = AgenticMetric(
                    session_id=dataset.session_id,
                    assistant_id=dataset.assistant_id,
                    total_interactions=total_interactions,
                    correct_interactions=correct_interactions,
                    is_fully_correct=is_fully_correct,
                    threshold=self.threshold,
                    correctness_scores=correctness_scores,
                    correct_indices=correct_indices,
                    tool_correctness_scores=tool_correctness_scores if tool_correctness_scores else [],
                    k=self.k,
                    pass_at_k=pass_at_k(total_interactions, correct_interactions, self.k),
                    pass_pow_k=pass_pow_k(total_interactions, correct_interactions, self.k),
                )
            else:
                p_result_dict = p_result
                p_samples = p_result_dict["samples"]  # type: ignore[index]
                pass_at_k_samples = 1.0 - (1.0 - p_samples) ** self.k
                pass_pow_k_samples = p_samples**self.k

                alpha = (1.0 - getattr(self.statistical_mode, "ci_level", 0.95)) / 2.0
                metric = AgenticMetric(
                    session_id=dataset.session_id,
                    assistant_id=dataset.assistant_id,
                    total_interactions=total_interactions,
                    correct_interactions=correct_interactions,
                    is_fully_correct=is_fully_correct,
                    threshold=self.threshold,
                    correctness_scores=correctness_scores,
                    correct_indices=correct_indices,
                    tool_correctness_scores=tool_correctness_scores if tool_correctness_scores else [],
                    k=self.k,
                    pass_at_k=float(np.mean(pass_at_k_samples)),
                    pass_at_k_ci_low=float(np.quantile(pass_at_k_samples, alpha)),
                    pass_at_k_ci_high=float(np.quantile(pass_at_k_samples, 1.0 - alpha)),
                    pass_pow_k=float(np.mean(pass_pow_k_samples)),
                    pass_pow_k_ci_low=float(np.quantile(pass_pow_k_samples, alpha)),
                    pass_pow_k_ci_high=float(np.quantile(pass_pow_k_samples, 1.0 - alpha)),
                )

            self.metrics.append(metric)

        fully_correct_count = sum(1 for m in self.metrics if m.is_fully_correct)
        self.logger.info(
            f"[Agentic] Completed evaluation. "
            f"{fully_correct_count}/{len(self.metrics)} conversations fully correct "
            f"({fully_correct_count/len(self.metrics)*100:.1f}%)"
        )
        return self.metrics
