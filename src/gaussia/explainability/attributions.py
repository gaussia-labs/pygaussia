"""Attribution-based explainability for language models using interpreto."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from gaussia.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)
from gaussia.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


# =============================================================================
# Attribution Result Parser Interface
# =============================================================================


class AttributionResultParser(ABC):
    """
    Abstract base class for parsing attribution results from different sources.

    Implement this interface to support different attribution library output formats.
    """

    @abstractmethod
    def parse(self, raw_result: Any) -> tuple[list[str], list[float]]:
        """
        Parse raw attribution result into tokens and scores.

        Args:
            raw_result: Raw result from the attribution library

        Returns:
            Tuple of (tokens, scores) lists
        """


class InterpretoResultParser(AttributionResultParser):
    """Parser for interpreto library attribution results."""

    def parse(self, raw_result: Any) -> tuple[list[str], list[float]]:
        """Parse interpreto attribution result."""
        # interpreto returns a list, we take the first element
        attr_data = raw_result[0] if isinstance(raw_result, list) else raw_result

        tokens: list[str] = []
        scores: list[float] = []

        # Try attribute-based access patterns
        if hasattr(attr_data, "tokens") and hasattr(attr_data, "attributions"):
            tokens = list(attr_data.tokens)
            scores = list(attr_data.attributions)
        elif hasattr(attr_data, "words") and hasattr(attr_data, "scores"):
            tokens = list(attr_data.words)
            scores = list(attr_data.scores)
        elif hasattr(attr_data, "input_tokens") and hasattr(attr_data, "attribution_scores"):
            tokens = list(attr_data.input_tokens)
            scores = list(attr_data.attribution_scores)
        # Try dict-based access
        elif isinstance(attr_data, dict):
            tokens = list(attr_data.get("tokens") or attr_data.get("words") or [])
            scores = list(attr_data.get("attributions") or attr_data.get("scores") or [])
        # Try sequence-based access
        else:
            try:
                if len(attr_data) >= 2:
                    tokens, scores = list(attr_data[0]), list(attr_data[1])
            except (TypeError, IndexError):
                pass

        # Ensure scores are floats
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = [float(s) for s in scores]
        tokens = [str(t) for t in tokens]

        return tokens, scores


# =============================================================================
# Attribution Method Classes
# =============================================================================


class BaseAttributionMethod(ABC):
    """
    Abstract base class for attribution methods.

    Subclass this to add new attribution methods. The implementation uses
    interpreto internally but can be migrated to custom implementations.
    """

    # Class attributes to be set by subclasses
    name: str = ""
    method_enum: AttributionMethod = AttributionMethod.LIME  # Default, override in subclass

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        granularity: Granularity = Granularity.WORD,
        **kwargs: Any,
    ):
        """
        Initialize the attribution method.

        Args:
            model: HuggingFace PreTrainedModel
            tokenizer: HuggingFace PreTrainedTokenizer
            granularity: Granularity level for attributions
            **kwargs: Additional arguments for the underlying implementation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.granularity = granularity
        self._explainer = self._create_explainer(**kwargs)

    @abstractmethod
    def _create_explainer(self, **kwargs: Any) -> Any:
        """Create the underlying explainer instance."""

    def compute(self, prompt: str, target: str, max_length: int = 512) -> Any:
        """
        Compute attributions for the given prompt and target.

        Args:
            prompt: The formatted input prompt string
            target: The model's output to explain
            max_length: Maximum sequence length

        Returns:
            Raw attribution result from underlying library
        """
        return self._explainer(prompt, target, max_length=max_length)


def _get_interpreto_granularity(granularity: Granularity):
    """Convert Granularity enum to interpreto's Granularity."""
    from interpreto import Granularity as InterpretoGranularity

    mapping = {
        Granularity.TOKEN: InterpretoGranularity.TOKEN,
        Granularity.WORD: InterpretoGranularity.WORD,
        Granularity.SENTENCE: InterpretoGranularity.SENTENCE,
    }
    return mapping[granularity]


# Gradient-based methods


class Saliency(BaseAttributionMethod):
    """Saliency attribution method (Simonyan et al., 2013)."""

    name = "saliency"
    method_enum = AttributionMethod.SALIENCY

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import Saliency as InterpretoSaliency

        return InterpretoSaliency(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class IntegratedGradients(BaseAttributionMethod):
    """Integrated Gradients attribution method (Sundararajan et al., 2017)."""

    name = "integrated_gradients"
    method_enum = AttributionMethod.INTEGRATED_GRADIENTS

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import IntegratedGradients as InterpretoIG

        return InterpretoIG(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class GradientShap(BaseAttributionMethod):
    """GradientSHAP attribution method (Lundberg and Lee, 2017)."""

    name = "gradient_shap"
    method_enum = AttributionMethod.GRADIENT_SHAP

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import GradientShap as InterpretoGS

        return InterpretoGS(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class SmoothGrad(BaseAttributionMethod):
    """SmoothGrad attribution method (Smilkov et al., 2017)."""

    name = "smooth_grad"
    method_enum = AttributionMethod.SMOOTH_GRAD

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import SmoothGrad as InterpretoSG

        return InterpretoSG(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class SquareGrad(BaseAttributionMethod):
    """SquareGrad attribution method (Hooker et al., 2019)."""

    name = "square_grad"
    method_enum = AttributionMethod.SQUARE_GRAD

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import SquareGrad as InterpretoSqG

        return InterpretoSqG(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class VarGrad(BaseAttributionMethod):
    """VarGrad attribution method (Richter et al., 2020)."""

    name = "var_grad"
    method_enum = AttributionMethod.VAR_GRAD

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import VarGrad as InterpretoVG

        return InterpretoVG(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class InputXGradient(BaseAttributionMethod):
    """Input x Gradient attribution method (Simonyan et al., 2013)."""

    name = "input_x_gradient"
    method_enum = AttributionMethod.INPUT_X_GRADIENT

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import InputxGradient as InterpretoIxG

        return InterpretoIxG(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


# Perturbation-based methods


class Lime(BaseAttributionMethod):
    """LIME attribution method (Ribeiro et al., 2013)."""

    name = "lime"
    method_enum = AttributionMethod.LIME

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import Lime as InterpretoLime

        return InterpretoLime(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class KernelShap(BaseAttributionMethod):
    """KernelSHAP attribution method (Lundberg and Lee, 2017)."""

    name = "kernel_shap"
    method_enum = AttributionMethod.KERNEL_SHAP

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import KernelShap as InterpretoKS

        return InterpretoKS(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class Occlusion(BaseAttributionMethod):
    """Occlusion attribution method (Zeiler and Fergus, 2014)."""

    name = "occlusion"
    method_enum = AttributionMethod.OCCLUSION

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import Occlusion as InterpretoOcc

        return InterpretoOcc(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


class Sobol(BaseAttributionMethod):
    """Sobol attribution method (Fel et al., 2021)."""

    name = "sobol"
    method_enum = AttributionMethod.SOBOL

    def _create_explainer(self, **kwargs: Any) -> Any:
        from interpreto import Sobol as InterpretoSobol

        return InterpretoSobol(
            self.model,
            self.tokenizer,
            granularity=_get_interpreto_granularity(self.granularity),
            **kwargs,
        )


# =============================================================================
# Main Explainer Class
# =============================================================================


class AttributionExplainer:
    """
    Compute token/word/sentence attributions for language model responses.

    This class provides explainability for transformer-based language models
    by computing how much each input token contributes to the model's output.

    Note:
        This class expects pre-formatted prompts. Users are responsible for
        formatting their messages into the appropriate prompt format for their
        model (e.g., using tokenizer.apply_chat_template).

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from gaussia.explainability import AttributionExplainer, Lime, Granularity
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>>
        >>> # Format your prompt (model-specific)
        >>> messages = [{"role": "user", "content": "What is gravity?"}]
        >>> prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        >>>
        >>> explainer = AttributionExplainer(model, tokenizer)
        >>> result = explainer.explain(
        ...     prompt=prompt,
        ...     target="Gravity is the force that attracts objects toward each other.",
        ...     method=Lime
        ... )
        >>> print(result.get_top_k(5))
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        *,
        default_method: type[BaseAttributionMethod] = Lime,
        default_granularity: Granularity = Granularity.WORD,
        result_parser: AttributionResultParser | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the attribution explainer.

        Args:
            model: A HuggingFace PreTrainedModel (e.g., AutoModelForCausalLM)
            tokenizer: The corresponding tokenizer for the model
            default_method: Default attribution method class to use
            default_granularity: Default granularity level (token, word, or sentence)
            result_parser: Custom parser for attribution results (defaults to InterpretoResultParser)
            verbose: Enable verbose logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.default_method = default_method
        self.default_granularity = default_granularity
        self.result_parser = result_parser or InterpretoResultParser()
        self.logger = VerboseLogger(verbose)

        # Cache for method instances
        self._method_cache: dict[tuple[type[BaseAttributionMethod], Granularity], BaseAttributionMethod] = {}

    def _get_method_instance(
        self,
        method_class: type[BaseAttributionMethod],
        granularity: Granularity,
        **kwargs: Any,
    ) -> BaseAttributionMethod:
        """Get or create a method instance."""
        cache_key = (method_class, granularity)

        # Only use cache if no extra kwargs are provided
        if not kwargs and cache_key in self._method_cache:
            return self._method_cache[cache_key]

        self.logger.info(f"Creating {method_class.name} method with {granularity.value} granularity")

        instance = method_class(
            self.model,
            self.tokenizer,
            granularity=granularity,
            **kwargs,
        )

        if not kwargs:
            self._method_cache[cache_key] = instance

        return instance

    def _build_attribution_result(
        self,
        tokens: list[str],
        scores: list[float],
        prompt: str,
        target: str,
        method: type[BaseAttributionMethod],
        granularity: Granularity,
    ) -> AttributionResult:
        """Build AttributionResult from parsed tokens and scores."""
        # Normalize scores to [0, 1] range
        normalized: list[float] = []
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            if score_range > 0:
                normalized = [(s - min_score) / score_range for s in scores]
            else:
                normalized = [0.5] * len(scores)

        # Create TokenAttribution objects
        attributions = []
        for i, (token, score) in enumerate(zip(tokens, scores, strict=False)):
            norm_score = normalized[i] if normalized and i < len(normalized) else None
            attributions.append(
                TokenAttribution(
                    text=token,
                    score=score,
                    position=i,
                    normalized_score=norm_score,
                )
            )

        return AttributionResult(
            prompt=prompt,
            target=target,
            method=method.method_enum,
            granularity=granularity,
            attributions=attributions,
        )

    def explain(
        self,
        prompt: str,
        target: str,
        *,
        method: type[BaseAttributionMethod] | None = None,
        granularity: Granularity | None = None,
        max_length: int = 512,
        **method_kwargs: Any,
    ) -> AttributionResult:
        """
        Compute attributions for a model response given an input prompt.

        Args:
            prompt: The formatted input prompt string. Users should format this
                    according to their model's requirements (e.g., using
                    tokenizer.apply_chat_template).
            target: The model's response/output to explain
            method: Attribution method class to use (defaults to instance default)
            granularity: Granularity level (defaults to instance default)
            max_length: Maximum sequence length for attribution computation
            **method_kwargs: Additional kwargs passed to the attribution method

        Returns:
            AttributionResult containing token attributions and metadata
        """
        method = method or self.default_method
        granularity = granularity or self.default_granularity

        self.logger.info(f"Computing {method.name} attributions at {granularity.value} level")
        self.logger.debug(f"Prompt (first 200 chars): {prompt[:200]}...")

        # Get or create method instance
        method_instance = self._get_method_instance(method, granularity, **method_kwargs)

        # Compute attributions
        start_time = time.time()
        raw_result = method_instance.compute(prompt, target, max_length=max_length)
        compute_time = time.time() - start_time

        self.logger.info(f"Attribution computed in {compute_time:.2f}s")

        # Parse result using the parser interface
        tokens, scores = self.result_parser.parse(raw_result)

        if not tokens:
            self.logger.warning("No tokens extracted from attribution result")

        # Build and return result
        result = self._build_attribution_result(tokens, scores, prompt, target, method, granularity)
        result.metadata["compute_time_seconds"] = compute_time
        result.metadata["max_length"] = max_length

        return result

    def explain_batch(
        self,
        items: list[tuple[str, str]],
        *,
        method: type[BaseAttributionMethod] | None = None,
        granularity: Granularity | None = None,
        max_length: int = 512,
        **method_kwargs: Any,
    ) -> AttributionBatchResult:
        """
        Compute attributions for multiple prompt/target pairs.

        Args:
            items: List of (prompt, target) tuples. Prompts should be pre-formatted.
            method: Attribution method class to use
            granularity: Granularity level
            max_length: Maximum sequence length
            **method_kwargs: Additional kwargs for the method

        Returns:
            AttributionBatchResult containing all results
        """
        method = method or self.default_method
        granularity = granularity or self.default_granularity

        self.logger.info(f"Processing batch of {len(items)} items")
        start_time = time.time()

        results = []
        for prompt, target in items:
            result = self.explain(
                prompt=prompt,
                target=target,
                method=method,
                granularity=granularity,
                max_length=max_length,
                **method_kwargs,
            )
            results.append(result)

        total_time = time.time() - start_time

        # Get model name
        model_name = getattr(self.model.config, "_name_or_path", "unknown")

        return AttributionBatchResult(
            results=results,
            model_name=model_name,
            total_compute_time_seconds=total_time,
        )

    def visualize(
        self,
        result: AttributionResult,
        *,
        return_html: bool = False,
    ) -> str | None:
        """
        Display or return HTML visualization of attributions.

        Args:
            result: AttributionResult to visualize
            return_html: If True, return HTML string instead of displaying

        Returns:
            HTML string if return_html=True, otherwise None (displays in notebook)
        """
        from interpreto import AttributionVisualization

        # Create a mock attribution object that interpreto can visualize
        class _MockAttribution:
            def __init__(self, tokens: list[str], scores: list[float]):
                self.tokens = tokens
                self.attributions = scores

        mock_attr = _MockAttribution(
            tokens=[attr.text for attr in result.attributions],
            scores=[attr.score for attr in result.attributions],
        )

        viz = AttributionVisualization(mock_attr)

        if return_html:
            html_result: str = viz.to_html()
            return html_result

        viz.display()
        return None


def compute_attributions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    *,
    method: type[BaseAttributionMethod] = Lime,
    granularity: Granularity = Granularity.WORD,
    **kwargs: Any,
) -> AttributionResult:
    """
    Convenience function to compute attributions in one call.

    This is a simpler interface for one-off attribution computations.
    For repeated use, instantiate AttributionExplainer directly.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Pre-formatted prompt string
        target: Model's response to explain
        method: Attribution method class
        granularity: Granularity level
        **kwargs: Additional kwargs for explain()

    Returns:
        AttributionResult
    """
    explainer = AttributionExplainer(
        model=model,
        tokenizer=tokenizer,
        default_method=method,
        default_granularity=granularity,
    )
    return explainer.explain(prompt=prompt, target=target, **kwargs)
