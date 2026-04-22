"""Role adherence metric for evaluating whether an AI assistant adheres to its defined role."""

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

from gaussia.core import Gaussia, Retriever
from gaussia.llm import Judge, RoleAdherenceJudgeOutput
from gaussia.llm.prompts import (
    role_adherence_binary_system_prompt,
    role_adherence_continuous_system_prompt,
)
from gaussia.schemas import Batch, IterationLevel
from gaussia.schemas.role_adherence import RoleAdherenceMetric, RoleAdherenceTurn
from gaussia.statistical import FrequentistMode, StatisticalMode


class ScoringStrategy(ABC):
    """Abstract strategy for scoring per-turn role adherence."""

    @abstractmethod
    def score(
        self,
        turn: Batch,
        history: list[Batch],
        chatbot_role: str,
    ) -> tuple[float, str | None]:
        """Evaluate a single turn against the role definition.

        Args:
            turn: The current interaction to evaluate.
            history: All prior turns in the session (T<i).
            chatbot_role: The role definition string R.

        Returns:
            Tuple of (score ∈ [0,1], optional_reason).
        """


class LLMJudgeStrategy(ScoringStrategy):
    """Scoring strategy using an LLM judge to evaluate role adherence per turn.

    The judge reads the role definition directly and evaluates adherence without
    requiring ground-truth references. Supports binary classification (default)
    and continuous scoring modes.

    Args:
        model: LangChain BaseChatModel instance for evaluation.
        binary: If True, judge is prompted to return 0 or 1. If False, returns [0,1].
        use_structured_output: If True, use LangChain's with_structured_output().
        strict: Strict mode for structured output parsing.
        bos_json_clause: Opening marker for JSON blocks.
        eos_json_clause: Closing marker for JSON blocks.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        model: BaseChatModel,
        binary: bool = True,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        verbose: bool = False,
    ):
        self.model = model
        self.binary = binary
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.verbose = verbose
        self._prompt = role_adherence_binary_system_prompt if binary else role_adherence_continuous_system_prompt

    def score(self, turn: Batch, history: list[Batch], chatbot_role: str) -> tuple[float, str | None]:
        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            strict=self.strict,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )
        _, result = judge.check(
            self._prompt,
            turn.query,
            {
                "chatbot_role": chatbot_role,
                "history": self._format_history(history),
                "query": turn.query,
                "assistant_response": turn.assistant,
            },
            output_schema=RoleAdherenceJudgeOutput,
        )

        if result is None:
            raise ValueError(f"[GAUSSIA/ROLE_ADHERENCE] No valid response from judge for QA ID: {turn.qa_id}")

        score = float(result["score"] if isinstance(result, dict) else result.score)
        reason = result["reason"] if isinstance(result, dict) else result.reason
        return score, reason

    def _format_history(self, history: list[Batch]) -> str:
        if not history:
            return "No prior conversation."
        lines = []
        for turn in history:
            lines.append(f"User: {turn.query}")
            lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(lines)


class RoleAdherence(Gaussia):
    """Metric for evaluating whether an AI assistant adheres to its defined role across conversation turns.

    Implements RoleAdherence(R, T) = (1/n) Σᵢ adhere(tᵢ, T<i, R) from the Gaussia role adherence paper.
    Each turn is evaluated in context of the full prior conversation history T<i and the role definition R.

    Requires `chatbot_role` to be set on the Dataset objects returned by the retriever.
    Does not support STREAM_BATCHES iteration level.

    Args:
        retriever: Retriever class for loading datasets.
        scoring_strategy: Strategy object that scores each turn (e.g. LLMJudgeStrategy).
        statistical_mode: Statistical computation mode (defaults to FrequentistMode).
        binary: If True, per-turn scores are binarized using threshold. Session score = proportion adherent.
                If False, per-turn continuous scores are averaged.
        strict_mode: If True, the session is adherent only if all turns are adherent.
        threshold: Minimum score to classify a turn (or session) as adherent.
        include_reason: If True, include the judge's justification in per-turn output.
        **kwargs: Additional arguments passed to Gaussia base class.
    """

    def __init__(
        self,
        retriever: type[Retriever],
        scoring_strategy: ScoringStrategy,
        statistical_mode: StatisticalMode | None = None,
        binary: bool = True,
        strict_mode: bool = False,
        threshold: float = 0.5,
        include_reason: bool = False,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)

        if self.level == IterationLevel.STREAM_BATCHES:
            raise ValueError(
                "RoleAdherence does not support STREAM_BATCHES iteration level. "
                "Use FULL_DATASET or STREAM_SESSIONS so that chatbot_role is available per session."
            )

        self.scoring_strategy = scoring_strategy
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()
        self.binary = binary
        self.strict_mode = strict_mode
        self.threshold = threshold
        self.include_reason = include_reason
        self._current_chatbot_role: str = ""
        self._session_data: dict[str, dict] = {}

    def _process_dataset(self, data):
        for element in data:
            self._current_chatbot_role = element.chatbot_role or ""
            self.batch(
                session_id=element.session_id,
                context=element.context,
                assistant_id=element.assistant_id,
                batch=element.conversation,
                language=element.language,
            )

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if session_id not in self._session_data:
            self._session_data[session_id] = {
                "assistant_id": assistant_id,
                "batches": [],
                "scores": [],
                "turns": [],
            }

        history: list[Batch] = []
        for turn in batch:
            self.logger.debug(f"QA ID: {turn.qa_id}")

            raw_score, reason = self.scoring_strategy.score(turn, list(history), self._current_chatbot_role)
            history.append(turn)

            adherent_turn = raw_score >= self.threshold
            stored_score = float(adherent_turn) if self.binary else raw_score

            self.logger.debug(f"Role adherence score: {stored_score}, adherent: {adherent_turn}")

            self._session_data[session_id]["batches"].append(turn)
            self._session_data[session_id]["scores"].append(stored_score)
            self._session_data[session_id]["turns"].append(
                RoleAdherenceTurn(
                    qa_id=turn.qa_id,
                    adherence_score=stored_score,
                    adherent=adherent_turn,
                    reason=reason if self.include_reason else None,
                )
            )

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            batches = data["batches"]
            scores = data["scores"]
            turns = data["turns"]
            weights = self._resolve_weights(batches)

            mean, ci_low, ci_high = self._aggregate_scores(scores, batches, weights, self.statistical_mode)

            adherent = all(t.adherent for t in turns) if self.strict_mode else mean >= self.threshold

            self.metrics.append(
                RoleAdherenceMetric(
                    session_id=session_id,
                    assistant_id=data["assistant_id"],
                    n_turns=len(batches),
                    role_adherence=mean,
                    role_adherence_ci_low=ci_low,
                    role_adherence_ci_high=ci_high,
                    adherent=adherent,
                    turns=turns,
                )
            )
