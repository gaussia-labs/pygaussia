"""Tests for IBMGranite guardian."""

from functools import partial
from unittest.mock import MagicMock, patch

from gaussia.guardians import IBMGranite
from gaussia.schemas.bias import GuardianBias, GuardianLLMConfig, LLMGuardianProviderInfer, ProtectedAttribute


def _make_config(mock_provider_class: MagicMock) -> GuardianLLMConfig:
    return GuardianLLMConfig.model_construct(
        model="ibm/granite-guardian-test",
        temperature=0.0,
        provider=mock_provider_class,
        api_key=None,
        url=None,
        logprobs=False,
        overrides={},
    )


def _make_attribute() -> ProtectedAttribute:
    return ProtectedAttribute(
        attribute=ProtectedAttribute.Attribute.gender,
        description="Bias related to gender identity",
    )


class TestIBMGraniteInit:
    @patch("gaussia.guardians.AutoTokenizer")
    def test_tokenizer_loaded_from_config_model(self, mock_auto_tokenizer):
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        IBMGranite(config=config)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("ibm/granite-guardian-test")

    @patch("gaussia.guardians.AutoTokenizer")
    def test_provider_instantiated_with_safe_no_unsafe_yes(self, mock_auto_tokenizer):
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        IBMGranite(config=config)

        _, kwargs = mock_provider_class.call_args
        assert kwargs["safe_token"] == "No"
        assert kwargs["unsafe_token"] == "Yes"

    @patch("gaussia.guardians.AutoTokenizer")
    def test_provider_receives_tokenizer_from_pretrained(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_provider_class = MagicMock()
        config = _make_config(mock_provider_class)

        IBMGranite(config=config)

        _, kwargs = mock_provider_class.call_args
        assert kwargs["tokenizer"] is mock_tokenizer


class TestIBMGraniteIsBiased:
    def _make_guardian(self) -> tuple[IBMGranite, MagicMock]:
        mock_provider_instance = MagicMock()
        mock_provider_class = MagicMock(return_value=mock_provider_instance)

        with patch("gaussia.guardians.AutoTokenizer"):
            guardian = IBMGranite(config=_make_config(mock_provider_class))

        return guardian, mock_provider_instance

    def test_returns_guardian_bias_instance(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert isinstance(result, GuardianBias)

    def test_is_biased_reflects_provider_infer_result(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=True, probability=0.95)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert result.is_biased is True
        assert result.certainty == 0.95

    def test_not_biased_reflects_provider_infer_result(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.05)

        result = guardian.is_biased("question", "answer", _make_attribute())

        assert result.is_biased is False
        assert result.certainty == 0.05

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

    def test_guardian_config_includes_attribute_risk_name(self):
        guardian, mock_provider = self._make_guardian()
        mock_provider.infer.return_value = LLMGuardianProviderInfer(is_bias=False, probability=0.1)
        attribute = _make_attribute()

        guardian.is_biased("question", "answer", attribute)

        prompt_partial: partial = mock_provider.infer.call_args[0][0]
        assert prompt_partial.keywords["guardian_config"]["risk_name"] == attribute.attribute.value
