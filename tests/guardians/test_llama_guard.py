"""Tests for LLamaGuard guardian."""

from functools import partial
from unittest.mock import MagicMock, patch

from gaussia.guardians import LLamaGuard
from gaussia.schemas.bias import GuardianBias, GuardianLLMConfig, LLMGuardianProviderInfer, ProtectedAttribute


def _make_config(mock_provider_class: MagicMock) -> GuardianLLMConfig:
    return GuardianLLMConfig.model_construct(
        model="meta-llama/llama-guard-test",
        temperature=0.0,
        provider=mock_provider_class,
        api_key=None,
        url=None,
        logprobs=False,
        overrides={},
    )


def _make_attribute() -> ProtectedAttribute:
    return ProtectedAttribute(
        attribute=ProtectedAttribute.Attribute.race,
        description="Bias related to race or ethnicity",
    )


class TestLLamaGuardInit:
    @patch("gaussia.guardians.AutoTokenizer")
    def test_tokenizer_loaded_from_config_model(self, mock_auto_tokenizer):
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        LLamaGuard(config=config)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("meta-llama/llama-guard-test")

    @patch("gaussia.guardians.AutoTokenizer")
    def test_provider_instantiated_with_chat_completions_true(self, mock_auto_tokenizer):
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        LLamaGuard(config=config)

        _, kwargs = mock_provider_class.call_args
        assert kwargs["chat_completions"] is True

    @patch("gaussia.guardians.AutoTokenizer")
    def test_provider_instantiated_with_safe_unsafe_tokens(self, mock_auto_tokenizer):
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        LLamaGuard(config=config)

        _, kwargs = mock_provider_class.call_args
        assert kwargs["safe_token"] == "safe"
        assert kwargs["unsafe_token"] == "unsafe"

    @patch("gaussia.guardians.AutoTokenizer")
    def test_provider_receives_tokenizer_from_pretrained(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        LLamaGuard(config=config)

        _, kwargs = mock_provider_class.call_args
        assert kwargs["tokenizer"] is mock_tokenizer


class TestLLamaGuardIsBiased:
    def _make_guardian(self) -> tuple[LLamaGuard, MagicMock]:
        mock_provider_instance = MagicMock()
        mock_provider_class = MagicMock(return_value=mock_provider_instance)

        with patch("gaussia.guardians.AutoTokenizer"):
            guardian = LLamaGuard(config=_make_config(mock_provider_class))

        return guardian, mock_provider_instance

    def test_returns_guardian_bias_instance(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert isinstance(result, GuardianBias)

    def test_is_biased_reflects_provider_infer_result(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=True, probability=0.92)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert result.is_biased is True
        assert result.certainty == 0.92

    def test_not_biased_reflects_provider_infer_result(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.03)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert result.is_biased is False
        assert result.certainty == 0.03

    def test_attribute_value_set_on_result(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)
        attribute = _make_attribute()

        result = guardian.is_biased("question", "answer", attribute)

        assert result.attribute == attribute.attribute.value

    def test_provider_infer_called_with_partial(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)

        guardian.is_biased("question", "answer", _make_attribute())

        mock_provider.infer.assert_called_once()
        prompt_arg = mock_provider.infer.call_args[0][0]
        assert isinstance(prompt_arg, partial)

    def test_messages_use_content_list_format(self):
        """LLamaGuard wraps content in [{"type": "text", "text": ...}] — distinto a IBMGranite."""
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)

        guardian.is_biased("my question", "my answer", _make_attribute())

        prompt_partial: partial = mock_provider.infer.call_args[0][0]
        conversation = prompt_partial.keywords["conversation"]
        assert isinstance(conversation[0]["content"], list)
        assert conversation[0]["content"][0]["type"] == "text"
        assert conversation[0]["content"][0]["text"] == "my question"
        assert conversation[1]["content"][0]["text"] == "my answer"

    def test_categories_include_attribute_value_and_description(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)
        attribute = _make_attribute()

        guardian.is_biased("question", "answer", attribute)

        prompt_partial: partial = mock_provider.infer.call_args[0][0]
        categories = prompt_partial.keywords["categories"]
        assert attribute.attribute.value in categories["S1"]
        assert attribute.description in categories["S1"]
