"""Tests for QwenEmbedder lazy initialization."""

from unittest.mock import MagicMock, patch

from gaussia.embedders.qwen import QwenEmbedder


class TestQwenEmbedderLazyInit:
    def test_tokenizer_is_none_before_first_access(self):
        embedder = QwenEmbedder(model_name="test-model")

        assert embedder._tokenizer is None

    def test_model_is_none_before_first_access(self):
        embedder = QwenEmbedder(model_name="test-model")

        assert embedder._model is None

    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_tokenizer_loaded_on_first_property_access(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        embedder = QwenEmbedder(model_name="test-model")
        result = embedder.tokenizer

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model", padding_side="left")
        assert result is mock_tokenizer

    @patch("gaussia.embedders.qwen.AutoModel")
    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_model_loaded_on_first_property_access(self, mock_auto_tokenizer, mock_auto_model):
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        embedder = QwenEmbedder(model_name="test-model")
        result = embedder.model

        mock_auto_model.from_pretrained.assert_called_once()
        assert result is mock_model

    @patch("gaussia.embedders.qwen.AutoModel")
    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_model_eval_called_on_load(self, mock_auto_tokenizer, mock_auto_model):
        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        embedder = QwenEmbedder(model_name="test-model")
        embedder.model

        mock_model.eval.assert_called_once()

    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_tokenizer_cached_after_first_access(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        embedder = QwenEmbedder(model_name="test-model")
        _ = embedder.tokenizer
        _ = embedder.tokenizer

        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @patch("gaussia.embedders.qwen.AutoModel")
    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_model_cached_after_first_access(self, mock_auto_tokenizer, mock_auto_model):
        mock_auto_model.from_pretrained.return_value = MagicMock()

        embedder = QwenEmbedder(model_name="test-model")
        _ = embedder.model
        _ = embedder.model

        mock_auto_model.from_pretrained.assert_called_once()

    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_tokenizer_not_none_after_access(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()

        embedder = QwenEmbedder(model_name="test-model")
        _ = embedder.tokenizer

        assert embedder._tokenizer is not None

    @patch("gaussia.embedders.qwen.AutoModel")
    @patch("gaussia.embedders.qwen.AutoTokenizer")
    def test_model_not_none_after_access(self, mock_auto_tokenizer, mock_auto_model):
        mock_auto_model.from_pretrained.return_value = MagicMock()

        embedder = QwenEmbedder(model_name="test-model")
        _ = embedder.model

        assert embedder._model is not None
