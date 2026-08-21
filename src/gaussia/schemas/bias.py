"""Bias metric schemas."""

from abc import ABC, abstractmethod
from enum import StrEnum
from functools import partial
from typing import Any

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from .metrics import BaseMetric

LOGPROB_VERDICT_METHOD = "logprob-last-verdict-token"
LOGPROB_SOFTMAX_METHOD = "logprob-verdict-softmax"
SAMPLED_ANSWER_METHOD = "sampled-answer"


class GuardianBias(BaseModel):
    """
    A data model that represents the result of a bias detection analysis.

    Attributes:
        is_biased (bool): Indicates whether bias was detected in the interaction
        attribute (str): The specific attribute that was analyzed for bias
        certainty (Optional[float]): P(violation) for this interaction, read from the
            guardian's own verdict distribution. None when the provider exposed no
            distribution to read — a score is then unavailable, not certain.
        method (Optional[str]): How certainty was obtained, so a real reading is
            distinguishable from its absence. One of the module's method constants.
    """

    is_biased: bool
    attribute: str
    certainty: float | None
    method: str | None = None


class ProtectedAttribute(BaseModel):
    """
    Protected attributes for bias detection.
    """

    class Attribute(StrEnum):
        age = "age"
        gender = "gender"
        race = "race"
        religion = "religion"
        nationality = "nationality"
        sexual_orientation = "sexual_orientation"

    attribute: Attribute
    description: str


class BiasMetric(BaseMetric):
    """
    Bias metric for evaluating the bias of the assistant's responses.
    """

    class AttributeBiasRate(BaseModel):
        protected_attribute: str
        n_samples: int
        k_biased: int
        rate: float
        ci_low: float | None = None
        ci_high: float | None = None

    class GuardianInteraction(GuardianBias):
        qa_id: str

    attribute_rates: list[AttributeBiasRate]
    guardian_interactions: dict[str, list[GuardianInteraction]]


class LLMGuardianProviderInfer(BaseModel):
    """Result from an LLM guardian provider inference.

    Attributes:
        is_bias (bool): The guardian's verdict.
        probability (Optional[float]): P(violation), or None when the provider returned
            no distribution. Never defaulted: a placeholder would be indistinguishable
            from a confident reading.
        method (Optional[str]): How probability was obtained.
    """

    is_bias: bool
    probability: float | None
    method: str | None = None


class LLMGuardianProvider(ABC):
    """Abstract base class for LLM guardian providers."""

    def __init__(
        self,
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        api_key: str | None = None,
        url: str | None = None,
        temperature: float = 0.0,
        safe_token: str = "No",
        unsafe_token: str = "Yes",
        max_tokens: int = 5,
        logprobs: bool = True,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.url = url
        self.temperature = temperature
        self.safe_token = safe_token
        self.unsafe_token = unsafe_token
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.logprobs = logprobs

    @abstractmethod
    def infer(self, prompt: partial) -> LLMGuardianProviderInfer:
        raise NotImplementedError("Subclass must implement this method")


class GuardianLLMConfig(BaseModel):
    """Configuration for LLM-based guardians."""

    model: str
    tokenizer_model: str | None = None
    api_key: str | None = None
    url: str | None = None
    temperature: float
    logprobs: bool = False
    chat_completions: bool = False
    provider: type[LLMGuardianProvider]
    overrides: dict[str, Any] = Field(default_factory=dict)
