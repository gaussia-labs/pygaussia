"""Role adherence metric for evaluating whether an AI assistant adheres to its defined role."""

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

from gaussia.core import Gaussia, Retriever
from gaussia.llm import Judge
from gaussia.llm.prompts import role_adherence_judge_system_prompt
from gaussia.schemas import Batch, IterationLevel
from gaussia.schemas.role_adherence import RoleAdherenceMetric, RoleAdherenceTurn
from gaussia.statistical import FrequentistMode, StatisticalMode


class ScoringStrategy(ABC):
    """Abstract strategy for scoring per-turn role adherence.

    Concrete implementations score a single turn against the role definition.
    The current implementation is `LLMJudgeStrategy` (LLM-as-judge with
    logprob-based scoring). Deterministic strategies (e.g. embedding similarity,
    rule-based) are tracked as future work — see the paper for evaluated
    deterministic baselines.
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


class LLMJudgeStrategy(ScoringStrategy):
    """Scoring strategy that uses an LLM judge with logprob-based scoring.

    Asks the judge a binary YES/NO question and derives a continuous
    [0, 1] adherence score from the first-token logprobs, via
    `Judge.check_logprob_binary()`.

    Args:
        model: LangChain BaseChatModel. Must be a provider that exposes
            logprobs (OpenAI, Azure OpenAI, Ollama, LiteLLM, HF TGI via
            BaseChatOpenAI). Anthropic/Gemini/Bedrock will raise
            LogprobsNotSupportedError on first invocation.
        temperature: Forwarded to the judge. Default 1.0 follows the
            paper to preserve first-token distribution calibration. Pass
            None to inherit the model's own configured temperature.
        top_logprobs: Number of top tokens to retrieve per position. Default
            10 matches the paper.
        verbose: Enable verbose logging on the underlying Judge.
    """

    def __init__(
        self,
        model: BaseChatModel,
        temperature: float | None = 1.0,
        top_logprobs: int = 10,
        verbose: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.verbose = verbose

    def score(self, turn: Batch, history: list[Batch], chatbot_role: str) -> float:
        judge = Judge(model=self.model, verbose=self.verbose)
        score, _ = judge.check_logprob_binary(
            role_adherence_judge_system_prompt,
            turn.query,
            {
                "chatbot_role": chatbot_role,
                "history": self._format_history(history),
                "query": turn.query,
                "assistant_response": turn.assistant,
            },
            top_logprobs=self.top_logprobs,
            temperature=self.temperature,
        )
        return score

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
