"""Frequentist statistical mode implementation."""

import numpy as np

from .base import StatisticalMode


class FrequentistMode(StatisticalMode):
    """Frequentist statistical computation - point estimates."""

    def distribution_divergence(
        self, observed: dict[str, int | float], reference: dict[str, float], divergence_type: str = "total_variation"
    ) -> float:
        """
        Total variation distance: 0.5 * sum(|p - q|)

        For frequentist mode, observed should already be proportions.
        """
        if divergence_type != "total_variation":
            raise NotImplementedError(f"Divergence type '{divergence_type}' not implemented")

        keys = set(observed.keys()) | set(reference.keys())
        divergence = 0.5 * sum(abs(float(observed.get(k, 0.0)) - float(reference.get(k, 0.0))) for k in keys)
        return float(divergence)

    def rate_estimation(self, successes: int, trials: int) -> float:
        """Simple proportion."""
        return float(successes / trials) if trials > 0 else 0.0

    def aggregate_metrics(self, metrics: dict[str, float], weights: dict[str, float]) -> float:
        """Weighted sum."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        return float(sum(metrics[k] * normalized_weights[k] for k in metrics))

    def dispersion_metric(self, values: dict[str, float], center: str = "mean") -> float:
        """Mean absolute deviation from center."""
        if not values:
            return 0.0

        vals = list(values.values())
        if center == "mean":
            center_val = sum(vals) / len(vals)
        elif center == "median":
            center_val = float(np.median(vals))
        else:
            raise ValueError(f"Unknown center type: {center}")

        return float(sum(abs(v - center_val) for v in vals) / len(vals))

    def get_result_type(self) -> str:
        return "point_estimate"
