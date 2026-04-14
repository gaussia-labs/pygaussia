"""Tests for HuggingFaceGuardianProvider — fixes 1 and 2."""

import sys
from unittest.mock import MagicMock, patch

from gaussia.guardians.llms.providers import HuggingFaceGuardianProvider


class TestHuggingFaceProviderInit:
    @patch("gaussia.guardians.llms.providers.AutoTokenizer")
    def test_tokenizer_loaded_from_model_name(self, mock_auto_tokenizer):
        """Fix 1: tokenizer is constructed from the model name, not left as None."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        provider = HuggingFaceGuardianProvider(model="test-model")

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
        assert provider.tokenizer is mock_tokenizer

    @patch("gaussia.guardians.llms.providers.AutoTokenizer")
    def test_tokenizer_is_not_none_after_init(self, mock_auto_tokenizer):
        """Regression: tokenizer must never be None after construction."""
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        provider = HuggingFaceGuardianProvider(model="test-model")

        assert provider.tokenizer is not None


class TestHuggingFaceProviderInfer:
    def _make_provider(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        return HuggingFaceGuardianProvider(model="test-model")

    def _make_mock_torch(self):
        mock_torch = MagicMock()
        # torch.no_grad() is used as a context manager
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        return mock_torch

    @patch("gaussia.guardians.llms.providers.AutoTokenizer")
    @patch("gaussia.guardians.llms.providers.AutoModelForCausalLM")
    def test_device_comes_from_model_parameters_not_model_name(self, mock_causal_lm_cls, mock_auto_tokenizer):
        """Fix 2: device is taken from next(model.parameters()).device, not self.model.device.

        self.model is a str. Calling .device on a str raises AttributeError.
        If infer() completes without AttributeError, the fix is confirmed.
        """
        mock_torch = self._make_mock_torch()

        mock_model = MagicMock()
        mock_causal_lm_cls.from_pretrained.return_value = mock_model

        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        provider = self._make_provider(mock_auto_tokenizer)
        provider._parse_output = MagicMock(return_value=(False, 0.5))

        mock_prompt = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = provider.infer(mock_prompt)

        assert result.is_bias is False
        assert result.probability == 0.5

    @patch("gaussia.guardians.llms.providers.AutoTokenizer")
    @patch("gaussia.guardians.llms.providers.AutoModelForCausalLM")
    def test_infer_reads_device_from_model_parameters(self, mock_causal_lm_cls, mock_auto_tokenizer):
        """Fix 2: verifies model.parameters() is actually called to obtain the device."""
        mock_torch = self._make_mock_torch()

        mock_model = MagicMock()
        mock_causal_lm_cls.from_pretrained.return_value = mock_model

        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        provider = self._make_provider(mock_auto_tokenizer)
        provider._parse_output = MagicMock(return_value=(False, 0.5))

        with patch.dict(sys.modules, {"torch": mock_torch}):
            provider.infer(MagicMock())

        mock_model.parameters.assert_called_once()
