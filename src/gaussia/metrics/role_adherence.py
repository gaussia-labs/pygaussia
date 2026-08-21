"""Role adherence metric for evaluating whether an AI assistant adheres to its defined role."""

import warnings
from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

from gaussia.core import Gaussia, Retriever
from gaussia.core.exceptions import LogprobsNotSupportedError
from gaussia.llm import Judge
from gaussia.llm.prompts import role_adherence_judge_system_prompt
from gaussia.schemas import Batch, IterationLevel
from gaussia.schemas.role_adherence import RoleAdherenceJudgeOutput, RoleAdherenceMetric, RoleAdherenceTurn
from gaussia.statistical import FrequentistMode, StatisticalMode


class ScoringStrategy(ABC):
    """Abstract strategy for scoring per-turn role adherence.

    Concrete implementations score a single turn against the role definition.
    Current implementations are `LLMJudgeStrategy` (logprob-based) and
    `StructuredOutputJudgeStrategy` (structured-output). Deterministic
    strategies (e.g. embedding similarity, rule-based) are tracked as future
    work — see the paper for evaluated deterministic baselines.
    """

    @abstractmethod
    def score(
        self,
        turn: Batch,
        history: list[Batch],
        chatbot_role: str,
    ) -> float:
        """Evaluate a single turn against the role definition.

        Args:
            turn: The current interaction to evaluate.
            history: All prior turns in the session (T<i).
            chatbot_role: The role definition string R.

        Returns:
            Score in [0, 1] where 1 means full adherence.
        """


class LLMScoringStrategy(ScoringStrategy):
    """Shared base for LLM-judge scoring strategies.

    Holds the model and the prompt-data assembly (history formatting and
    template variables) common to every LLM-judge strategy.
    """

    def __init__(self, model: BaseChatModel, verbose: bool = False):
        self.model = model
        self.verbose = verbose

    def _judge_data(self, turn: Batch, history: list[Batch], chatbot_role: str) -> dict:
        return {
            "chatbot_role": chatbot_role,
            "history": self._format_history(history),
            "query": turn.query,
            "assistant_response": turn.assistant,
        }

    @staticmethod
    def _format_history(history: list[Batch]) -> str:
        if not history:
            return "No prior conversation."
        lines = []
        for turn in history:
            lines.append(f"User: {turn.query}")
            lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(lines)


class StructuredOutputJudgeStrategy(LLMScoringStrategy):
    """Scoring strategy that uses an LLM judge with structured output.

    Asks the judge for a structured YES/NO verdict via `Judge.check()` and
    maps it to a binary {0.0, 1.0} adherence score. Works with any provider
    (no logprobs required), so it doubles as the fallback for
    `LLMJudgeStrategy` when logprobs are unavailable.

    Args:
        model: LangChain BaseChatModel.
        verbose: Enable verbose logging on the underlying Judge.
    """

    def score(self, turn: Batch, history: list[Batch], chatbot_role: str) -> float:
        judge = Judge(model=self.model, use_structured_output=True, verbose=self.verbose)
        _, result = judge.check(
            role_adherence_judge_system_prompt,
            turn.query,
            self._judge_data(turn, history, chatbot_role),
            output_schema=RoleAdherenceJudgeOutput,
        )
        return 1.0 if isinstance(result, RoleAdherenceJudgeOutput) and result.adherent else 0.0


class LLMJudgeStrategy(LLMScoringStrategy):
    """Scoring strategy that uses an LLM judge with logprob-based scoring.

    Asks the judge a binary YES/NO question and derives a continuous
    [0, 1] adherence score from the first-token logprobs, via
    `Judge.check_logprob_binary()`.

    Args:
        model: LangChain BaseChatModel. Providers that expose logprobs
            (OpenAI, Azure OpenAI, Ollama, LiteLLM, HF TGI via BaseChatOpenAI)
            use the logprob path directly. Providers that do not
            (Anthropic/Gemini/Bedrock) fall back to `fallback` if one is
            given; otherwise the first invocation raises
            LogprobsNotSupportedError.
        temperature: Forwarded to the judge. Default 1.0 follows the
            paper to preserve first-token distribution calibration. Pass
            None to inherit the model's own configured temperature.
        top_logprobs: Number of top tokens to retrieve per position. Default
            10 matches the paper.
        verbose: Enable verbose logging on the underlying Judge.
        fallback: Strategy to use when the provider does not expose logprobs.
            Pass `StructuredOutputJudgeStrategy(model)` to degrade gracefully
            instead of raising. Once the fallback triggers, it is reused for
            the rest of the run to avoid repeating the failed logprob call.
    """

    def __init__(
        self,
        model: BaseChatModel,
        temperature: float | None = 1.0,
        top_logprobs: int = 10,
        verbose: bool = False,
        fallback: ScoringStrategy | None = None,
    ):
        super().__init__(model, verbose)
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.fallback = fallback
        self._fell_back = False

    def score(self, turn: Batch, history: list[Batch], chatbot_role: str) -> float:
        if self._fell_back and self.fallback is not None:
            return self.fallback.score(turn, history, chatbot_role)

        judge = Judge(model=self.model, verbose=self.verbose)
        try:
            score, _ = judge.check_logprob_binary(
                role_adherence_judge_system_prompt,
                turn.query,
                self._judge_data(turn, history, chatbot_role),
                top_logprobs=self.top_logprobs,
                temperature=self.temperature,
            )
        except LogprobsNotSupportedError:
            if self.fallback is None:
                raise
            warnings.warn(
                f"Provider {type(self.model).__name__} does not expose logprobs; "
                f"falling back to structured-output scoring for the rest of this run.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fell_back = True
            return self.fallback.score(turn, history, chatbot_role)
        return score


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
                If False, the raw continuous logprob score is averaged.
        strict_mode: If True, the session is adherent only if all turns are adherent.
        threshold: Minimum score to classify a turn (or session) as adherent.
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

            raw_score = self.scoring_strategy.score(turn, list(history), self._current_chatbot_role)
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
