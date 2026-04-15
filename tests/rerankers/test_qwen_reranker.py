"""Tests for QwenReranker lazy initialization."""

from unittest.mock import MagicMock, patch

from gaussia.rerankers.qwen import QwenReranker


class TestQwenRerankerLazyInit:
    def test_tokenizer_is_none_before_first_access(self):
        reranker = QwenReranker(model_name="test-model")

        assert reranker._tokenizer is None

    def test_model_is_none_before_first_access(self):
        reranker = QwenReranker(model_name="test-model")

        assert reranker._model is None

    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_tokenizer_loaded_on_first_property_access(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        reranker = QwenReranker(model_name="test-model")
        result = reranker.tokenizer

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model", padding_side="left")
        assert result is mock_tokenizer

    @patch("gaussia.rerankers.qwen.AutoModelForCausalLM")
    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_model_loaded_on_first_property_access(self, mock_auto_tokenizer, mock_causal_lm):
        mock_model = MagicMock()
        mock_causal_lm.from_pretrained.return_value = mock_model

        reranker = QwenReranker(model_name="test-model")
        result = reranker.model

        mock_causal_lm.from_pretrained.assert_called_once()
        assert result is mock_model

    @patch("gaussia.rerankers.qwen.AutoModelForCausalLM")
    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_model_eval_called_on_load(self, mock_auto_tokenizer, mock_causal_lm):
        mock_model = MagicMock()
        mock_causal_lm.from_pretrained.return_value = mock_model

        reranker = QwenReranker(model_name="test-model")
        reranker.model

        mock_model.eval.assert_called_once()

    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_tokenizer_cached_after_first_access(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        reranker = QwenReranker(model_name="test-model")
        _ = reranker.tokenizer
        _ = reranker.tokenizer

        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @patch("gaussia.rerankers.qwen.AutoModelForCausalLM")
    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_model_cached_after_first_access(self, mock_auto_tokenizer, mock_causal_lm):
        mock_auto_model = MagicMock()
        mock_causal_lm.from_pretrained.return_value = mock_auto_model

        reranker = QwenReranker(model_name="test-model")
        _ = reranker.model
        _ = reranker.model

        mock_causal_lm.from_pretrained.assert_called_once()

    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_tokenizer_not_none_after_access(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        reranker = QwenReranker(model_name="test-model")
        _ = reranker.tokenizer

        assert reranker._tokenizer is not None

    @patch("gaussia.rerankers.qwen.AutoModelForCausalLM")
    @patch("gaussia.rerankers.qwen.AutoTokenizer")
    def test_model_not_none_after_access(self, mock_auto_tokenizer, mock_causal_lm):
        mock_causal_lm.from_pretrained.return_value = MagicMock()

        reranker = QwenReranker(model_name="test-model")
        _ = reranker.model

        assert reranker._model is not None
