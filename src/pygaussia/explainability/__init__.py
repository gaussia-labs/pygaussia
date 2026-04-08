"""
Explainability module for Gaussia.

This module provides token attribution analysis for language models using the
interpreto library. It helps understand which parts of the input contribute
most to the model's output.

Note:
    Users are responsible for formatting prompts according to their model's
    requirements (e.g., using tokenizer.apply_chat_template). This design
    choice keeps the explainability module focused on attribution computation
    and avoids coupling with specific LLM prompt formats.

Example:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from pygaussia.explainability import AttributionExplainer, Lime, Granularity
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>>
    >>> # Format prompt using tokenizer (model-specific)
    >>> messages = [{"role": "user", "content": "What is gravity?"}]
    >>> prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    >>>
    >>> explainer = AttributionExplainer(model, tokenizer)
    >>> result = explainer.explain(
    ...     prompt=prompt,
    ...     target="Gravity is the force of attraction between objects.",
    ...     method=Lime
    ... )
    >>> print(result.get_top_k(5))
"""

from pygaussia.explainability.attributions import (
    # Main classes
    AttributionExplainer,
    AttributionResultParser,
    BaseAttributionMethod,
    # Gradient-based methods
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    InterpretoResultParser,
    # Perturbation-based methods
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
    # Convenience function
    compute_attributions,
)
from pygaussia.schemas.explainability import (
    AttributionBatchResult,
    AttributionMethod,
    AttributionResult,
    Granularity,
    TokenAttribution,
)

__all__ = [
    # Main classes
    "AttributionExplainer",
    "BaseAttributionMethod",
    "AttributionResultParser",
    "InterpretoResultParser",
    # Convenience function
    "compute_attributions",
    # Gradient-based method classes
    "Saliency",
    "IntegratedGradients",
    "GradientShap",
    "SmoothGrad",
    "SquareGrad",
    "VarGrad",
    "InputXGradient",
    # Perturbation-based method classes
    "Lime",
    "KernelShap",
    "Occlusion",
    "Sobol",
    # Schemas
    "AttributionResult",
    "AttributionBatchResult",
    "TokenAttribution",
    # Enums
    "AttributionMethod",
    "Granularity",
]
