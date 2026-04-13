from functools import partial
from typing import Optional

from transformers import AutoTokenizer

from gaussia.core import Guardian
from gaussia.schemas.bias import GuardianBias, GuardianLLMConfig, ProtectedAttribute


class IBMGranite(Guardian):
    """
    IBM Granite is a guardian that uses the IBM Granite model to detect bias in AI responses.

    This guardian analyzes conversations between users and AI assistants to identify potential
    biases related to protected attributes. It uses IBM's Granite model to make bias assessments
    and provides certainty scores for its determinations.

    Args:
        protected_attributes (list[ProtectedAttribute]): List of attributes to check for bias against
        config (GuardianLLMConfig): Configuration for the IBM Granite model including API settings
        **kwargs: Additional keyword arguments passed to the parent Guardian class

    Attributes:
        config (GuardianLLMConfig): The configuration used for the IBM Granite model
        provider: The configured model provider instance for making inference calls
    """

    def __init__(self, config: GuardianLLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.provider = config.provider(
            model=config.model,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            api_key=config.api_key,
            url=config.url,
            temperature=config.temperature,
            safe_token="No",
            logprobs=config.logprobs,
            unsafe_token="Yes",
            overrides=config.overrides,
        )

    def is_biased(
        self, question: str, answer: str, attribute: ProtectedAttribute, context: str | None = None
    ) -> GuardianBias:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = partial(
            self.provider.tokenizer.apply_chat_template,  # type: ignore[attr-defined]
            conversation=messages,
            guardian_config={
                "risk_name": attribute.attribute.value,
                "risk_definition": attribute.description,
            },
            tokenize=False,
            add_generation_prompt=True,
        )
        infer = self.provider.infer(prompt)
        return GuardianBias(is_biased=infer.is_bias, attribute=attribute.attribute.value, certainty=infer.probability)


class LLamaGuard(Guardian):
    """
    LLamaGuard is a guardian that uses the LLamaGuard model to detect bias in AI responses.

    This guardian implements bias detection using Meta's LLamaGuard model, which is specifically
    designed for content safety and bias detection. It analyzes conversations and provides
    assessments of potential biases related to protected attributes.

    Args:
        protected_attributes (list[ProtectedAttribute]): List of attributes to check for bias against
        config (GuardianLLMConfig): Configuration for the LLamaGuard model including API settings
        **kwargs: Additional keyword arguments passed to the parent Guardian class

    Attributes:
        config (GuardianLLMConfig): The configuration used for the LLamaGuard model
        provider: The configured model provider instance for making inference calls
    """

    def __init__(self, config: GuardianLLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.provider = config.provider(
            model=config.model,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            api_key=config.api_key,
            url=config.url,
            temperature=config.temperature,
            safe_token="safe",
            unsafe_token="unsafe",
            logprobs=config.logprobs,
            chat_completions=True,
            overrides=config.overrides,
        )

    def is_biased(
        self, question: str, answer: str, attribute: ProtectedAttribute, context: str | None = None
    ) -> GuardianBias:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        prompt = partial(
            self.tokenizer.apply_chat_template,
            conversation=messages,  # type: ignore[arg-type]
            categories={"S1": f"{attribute.attribute.value}.\n{attribute.description}"},
        )
        infer = self.provider.infer(prompt)
        return GuardianBias(is_biased=infer.is_bias, attribute=attribute.attribute.value, certainty=infer.probability)
