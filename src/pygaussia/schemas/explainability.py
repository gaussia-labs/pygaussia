"""Schemas for explainability module - Token attribution results."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttributionMethod(str, Enum):
    """Available attribution methods from interpreto library."""

    # Gradient-based methods
    SALIENCY = "saliency"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    SMOOTH_GRAD = "smooth_grad"
    SQUARE_GRAD = "square_grad"
    VAR_GRAD = "var_grad"
    INPUT_X_GRADIENT = "input_x_gradient"

    # Perturbation-based methods
    LIME = "lime"
    KERNEL_SHAP = "kernel_shap"
    OCCLUSION = "occlusion"
    SOBOL = "sobol"


class Granularity(str, Enum):
    """Granularity level for attributions."""

    TOKEN = "token"
    WORD = "word"
    SENTENCE = "sentence"


class TokenAttribution(BaseModel):
    """Attribution score for a single token or text unit."""

    text: str = Field(..., description="The token or text unit")
    score: float = Field(..., description="Attribution score (importance)")
    position: int = Field(..., description="Position in the sequence")
    normalized_score: float | None = Field(None, description="Score normalized to [0, 1] range")


class AttributionResult(BaseModel):
    """Complete attribution result for a single explanation."""

    prompt: str = Field(..., description="The input prompt")
    target: str = Field(..., description="The target/generated text being explained")
    method: AttributionMethod = Field(..., description="Attribution method used")
    granularity: Granularity = Field(..., description="Granularity level of attributions")
    attributions: list[TokenAttribution] = Field(..., description="List of token attributions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def top_attributions(self) -> list[TokenAttribution]:
        """Get attributions sorted by score (highest first)."""
        return sorted(self.attributions, key=lambda x: abs(x.score), reverse=True)

    def get_top_k(self, k: int = 10) -> list[TokenAttribution]:
        """Get top-k most important tokens/units."""
        return self.top_attributions[:k]

    def to_dict_for_visualization(self) -> dict[str, Any]:
        """Convert to dictionary format suitable for visualization."""
        return {
            "tokens": [attr.text for attr in self.attributions],
            "scores": [attr.score for attr in self.attributions],
            "normalized_scores": [attr.normalized_score for attr in self.attributions],
        }


class AttributionBatchResult(BaseModel):
    """Results from computing attributions on multiple inputs."""

    results: list[AttributionResult] = Field(..., description="List of attribution results")
    model_name: str = Field(..., description="Name/ID of the model used")
    total_compute_time_seconds: float | None = Field(None, description="Total computation time")

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx: int) -> AttributionResult:
        return self.results[idx]
