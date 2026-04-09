"""Tests for Attribution Explainer."""

from unittest.mock import MagicMock, patch

import pytest

from gaussia.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)


class TestAttributionSchemas:
    """Test suite for attribution schemas."""

    def test_token_attribution_creation(self):
        """Test TokenAttribution creation."""
        attr = TokenAttribution(
            text="hello",
            score=0.8,
            position=0,
            normalized_score=0.9,
        )
        assert attr.text == "hello"
        assert attr.score == 0.8
        assert attr.position == 0
        assert attr.normalized_score == 0.9

    def test_token_attribution_without_normalized_score(self):
        """Test TokenAttribution without normalized_score."""
        attr = TokenAttribution(
            text="world",
            score=-0.5,
            position=1,
        )
        assert attr.text == "world"
        assert attr.score == -0.5
        assert attr.normalized_score is None

    def test_attribution_result_creation(self):
        """Test AttributionResult creation."""
        attributions = [
            TokenAttribution(text="hello", score=0.8, position=0),
            TokenAttribution(text="world", score=0.2, position=1),
        ]
        result = AttributionResult(
            prompt="Hello world",
            target="Greeting response",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        assert result.prompt == "Hello world"
        assert result.target == "Greeting response"
        assert result.method == AttributionMethod.LIME
        assert result.granularity == Granularity.WORD
        assert len(result.attributions) == 2

    def test_attribution_result_top_attributions(self):
        """Test top_attributions property."""
        attributions = [
            TokenAttribution(text="low", score=0.1, position=0),
            TokenAttribution(text="high", score=0.9, position=1),
            TokenAttribution(text="medium", score=0.5, position=2),
        ]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.SALIENCY,
            granularity=Granularity.TOKEN,
            attributions=attributions,
        )
        top = result.top_attributions
        assert top[0].text == "high"
        assert top[1].text == "medium"
        assert top[2].text == "low"

    def test_attribution_result_get_top_k(self):
        """Test get_top_k method."""
        attributions = [TokenAttribution(text=f"word_{i}", score=i * 0.1, position=i) for i in range(10)]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        top_3 = result.get_top_k(3)
        assert len(top_3) == 3
        assert top_3[0].text == "word_9"

    def test_attribution_result_to_dict_for_visualization(self):
        """Test to_dict_for_visualization method."""
        attributions = [
            TokenAttribution(text="a", score=0.5, position=0, normalized_score=0.7),
            TokenAttribution(text="b", score=0.3, position=1, normalized_score=0.4),
        ]
        result = AttributionResult(
            prompt="test",
            target="test",
            method=AttributionMethod.LIME,
            granularity=Granularity.WORD,
            attributions=attributions,
        )
        viz_dict = result.to_dict_for_visualization()
        assert viz_dict["tokens"] == ["a", "b"]
        assert viz_dict["scores"] == [0.5, 0.3]
        assert viz_dict["normalized_scores"] == [0.7, 0.4]

    def test_attribution_batch_result_iteration(self):
        """Test AttributionBatchResult iteration and indexing."""
        results = [
            AttributionResult(
                prompt=f"prompt_{i}",
                target=f"target_{i}",
                method=AttributionMethod.LIME,
                granularity=Granularity.WORD,
                attributions=[],
            )
            for i in range(3)
        ]
        batch = AttributionBatchResult(
            results=results,
            model_name="test-model",
            total_compute_time_seconds=1.5,
        )
        assert len(batch) == 3
        assert batch[0].prompt == "prompt_0"
        assert batch[1].prompt == "prompt_1"
        assert list(batch)[2].prompt == "prompt_2"


class TestAttributionMethod:
    """Test suite for AttributionMethod enum."""

    def test_gradient_methods(self):
        """Test gradient-based methods are available."""
        gradient_methods = [
            AttributionMethod.SALIENCY,
            AttributionMethod.INTEGRATED_GRADIENTS,
            AttributionMethod.GRADIENT_SHAP,
            AttributionMethod.SMOOTH_GRAD,
            AttributionMethod.SQUARE_GRAD,
            AttributionMethod.VAR_GRAD,
            AttributionMethod.INPUT_X_GRADIENT,
        ]
        for method in gradient_methods:
            assert isinstance(method.value, str)

    def test_perturbation_methods(self):
        """Test perturbation-based methods are available."""
        perturbation_methods = [
            AttributionMethod.LIME,
            AttributionMethod.KERNEL_SHAP,
            AttributionMethod.OCCLUSION,
            AttributionMethod.SOBOL,
        ]
        for method in perturbation_methods:
            assert isinstance(method.value, str)


class TestGranularity:
    """Test suite for Granularity enum."""

    def test_granularity_values(self):
        """Test granularity enum values."""
        assert Granularity.TOKEN.value == "token"
        assert Granularity.WORD.value == "word"
        assert Granularity.SENTENCE.value == "sentence"


class TestInterpretoResultParser:
    """Test suite for InterpretoResultParser."""

    def test_parse_object_with_tokens_attributions(self):
        """Test parsing object with tokens and attributions attributes."""
        from gaussia.explainability import InterpretoResultParser

        parser = InterpretoResultParser()

        mock_result = MagicMock()
        mock_result.tokens = ["a", "b", "c"]
        mock_result.attributions = [0.1, 0.5, 0.4]

        tokens, scores = parser.parse([mock_result])

        assert tokens == ["a", "b", "c"]
        assert scores == [0.1, 0.5, 0.4]

    def test_parse_dict_format(self):
        """Test parsing dict format."""
        from gaussia.explainability import InterpretoResultParser

        parser = InterpretoResultParser()

        mock_result = [{"tokens": ["x", "y"], "attributions": [0.3, 0.7]}]

        tokens, scores = parser.parse(mock_result)

        assert tokens == ["x", "y"]
        assert scores == [0.3, 0.7]

    def test_parse_empty_result(self):
        """Test parsing returns empty lists for unparseable input."""
        from gaussia.explainability import InterpretoResultParser

        parser = InterpretoResultParser()

        tokens, scores = parser.parse([None])

        assert tokens == []
        assert scores == []


class TestBaseAttributionMethod:
    """Test suite for BaseAttributionMethod class."""

    def test_method_classes_have_required_attributes(self):
        """Test all method classes have required class attributes."""
        from gaussia.explainability import (
            GradientShap,
            InputXGradient,
            IntegratedGradients,
            KernelShap,
            Lime,
            Occlusion,
            Saliency,
            SmoothGrad,
            Sobol,
            SquareGrad,
            VarGrad,
        )

        method_classes = [
            Saliency,
            IntegratedGradients,
            GradientShap,
            SmoothGrad,
            SquareGrad,
            VarGrad,
            InputXGradient,
            Lime,
            KernelShap,
            Occlusion,
            Sobol,
        ]

        for cls in method_classes:
            assert hasattr(cls, "name")
            assert hasattr(cls, "method_enum")
            assert isinstance(cls.name, str)
            assert len(cls.name) > 0


class TestAttributionExplainer:
    """Test suite for AttributionExplainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HuggingFace model."""
        model = MagicMock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    def test_explainer_initialization(self, mock_model, mock_tokenizer):
        """Test AttributionExplainer initialization."""
        from gaussia.explainability import AttributionExplainer, Lime

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            default_method=Lime,
            default_granularity=Granularity.WORD,
        )
        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
        assert explainer.default_method == Lime
        assert explainer.default_granularity == Granularity.WORD

    def test_explainer_initialization_defaults(self, mock_model, mock_tokenizer):
        """Test AttributionExplainer default parameters."""
        from gaussia.explainability import AttributionExplainer, Lime

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        assert explainer.default_method == Lime
        assert explainer.default_granularity == Granularity.WORD

    def test_explainer_accepts_custom_parser(self, mock_model, mock_tokenizer):
        """Test AttributionExplainer accepts custom result parser."""
        from gaussia.explainability import (
            AttributionExplainer,
            AttributionResultParser,
        )

        class CustomParser(AttributionResultParser):
            def parse(self, raw_result):
                return ["custom"], [1.0]

        parser = CustomParser()
        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            result_parser=parser,
        )
        assert explainer.result_parser == parser

    def test_build_attribution_result(self, mock_model, mock_tokenizer):
        """Test _build_attribution_result method."""
        from gaussia.explainability import AttributionExplainer, Lime

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        result = explainer._build_attribution_result(
            tokens=["a", "b", "c"],
            scores=[0.1, 0.5, 0.4],
            prompt="test prompt",
            target="test target",
            method=Lime,
            granularity=Granularity.WORD,
        )

        assert isinstance(result, AttributionResult)
        assert len(result.attributions) == 3
        assert result.attributions[0].text == "a"
        assert result.method == AttributionMethod.LIME

    def test_build_attribution_result_normalizes_scores(self, mock_model, mock_tokenizer):
        """Test that _build_attribution_result normalizes scores."""
        from gaussia.explainability import AttributionExplainer, Lime

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        result = explainer._build_attribution_result(
            tokens=["a", "b"],
            scores=[0.0, 1.0],
            prompt="test",
            target="test",
            method=Lime,
            granularity=Granularity.WORD,
        )

        # Min score should normalize to 0, max to 1
        assert result.attributions[0].normalized_score == 0.0
        assert result.attributions[1].normalized_score == 1.0


class TestConvenienceFunction:
    """Test suite for compute_attributions convenience function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    @patch("gaussia.explainability.attributions.Lime")
    def test_compute_attributions(
        self,
        mock_lime_class,
        mock_model,
        mock_tokenizer,
    ):
        """Test compute_attributions convenience function."""
        from gaussia.explainability import compute_attributions

        # Mock the Lime method
        mock_lime_instance = MagicMock()
        mock_lime_instance.compute.return_value = [{"tokens": ["test"], "attributions": [0.5]}]
        mock_lime_class.return_value = mock_lime_instance
        mock_lime_class.name = "lime"
        mock_lime_class.method_enum = AttributionMethod.LIME

        result = compute_attributions(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompt="Hello world",
            target="Response",
            method=mock_lime_class,
        )

        assert isinstance(result, AttributionResult)


class TestExplainMethod:
    """Test suite for explain method with mocked dependencies."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.config._name_or_path = "test-model"
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    def test_explain_uses_default_method(self, mock_model, mock_tokenizer):
        """Test explain uses default method when not specified."""
        from gaussia.explainability import AttributionExplainer, Saliency

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            default_method=Saliency,
        )

        # Verify default is set correctly
        assert explainer.default_method == Saliency

    def test_explain_batch_returns_batch_result(self, mock_model, mock_tokenizer):
        """Test explain_batch returns AttributionBatchResult."""
        from gaussia.explainability import AttributionExplainer

        # Create a mock parser that returns valid data
        mock_parser = MagicMock()
        mock_parser.parse.return_value = (["a", "b"], [0.5, 0.5])

        # Create a mock method class
        mock_method = MagicMock()
        mock_method.name = "mock"
        mock_method.method_enum = AttributionMethod.LIME
        mock_method_instance = MagicMock()
        mock_method_instance.compute.return_value = {}
        mock_method.return_value = mock_method_instance

        explainer = AttributionExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            default_method=mock_method,
            result_parser=mock_parser,
        )

        items = [
            ("prompt1", "target1"),
            ("prompt2", "target2"),
        ]

        result = explainer.explain_batch(items)

        assert isinstance(result, AttributionBatchResult)
        assert len(result) == 2
        assert result.model_name == "test-model"
