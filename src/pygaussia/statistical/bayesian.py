"""Bayesian statistical mode implementation."""

from typing import Any

import numpy as np

from .base import StatisticalMode


class BayesianMode(StatisticalMode):
    """Bayesian statistical computation - full posterior distributions."""

    def __init__(
        self,
        mc_samples: int = 5000,
        ci_level: float = 0.95,
        dirichlet_prior: float = 1.0,
        beta_prior_a: float = 1.0,
        beta_prior_b: float = 1.0,
        rng_seed: int | None = 42,
    ):
        """
        Initialize Bayesian mode with prior parameters.

        Args:
            mc_samples: Number of Monte Carlo samples for posterior
            ci_level: Credible interval level (default: 0.95 for 95% CI)
            dirichlet_prior: Symmetric Dirichlet prior parameter (alpha0)
            beta_prior_a: Beta prior parameter a (for rate estimation)
            beta_prior_b: Beta prior parameter b (for rate estimation)
            rng_seed: Random seed for reproducibility
        """
        self.mc_samples = mc_samples
        self.ci_level = ci_level
        self.dirichlet_prior = dirichlet_prior
        self.beta_prior_a = beta_prior_a
        self.beta_prior_b = beta_prior_b
        self.rng = np.random.default_rng(rng_seed)

    def distribution_divergence(
        self, observed_counts: dict[str, int], reference: dict[str, float], divergence_type: str = "total_variation"
    ) -> dict[str, Any]:
        """
        Bayesian divergence with Dirichlet posterior.

        For Bayesian mode, observed should be counts (not proportions).

        Args:
            observed_counts: Counts per category (not proportions!)
            reference: Reference distribution
            divergence_type: Type of divergence (default: total_variation)

        Returns:
            Dict with 'mean', 'ci_low', 'ci_high', 'samples'
        """
        categories = list(observed_counts.keys())
        if not categories:
            return self._empty_summary()

        counts = np.array([observed_counts[k] for k in categories])
        ref = np.array([reference.get(k, 1.0 / len(categories)) for k in categories])

        # Normalize reference
        ref = ref / ref.sum()

        # Dirichlet posterior
        alpha_posterior = self.dirichlet_prior + counts
        samples = self.rng.dirichlet(alpha_posterior, size=self.mc_samples)

        # Compute divergence for each sample
        if divergence_type == "total_variation":
            divergences = 0.5 * np.sum(np.abs(samples - ref), axis=1)
        else:
            raise NotImplementedError(f"Divergence type '{divergence_type}' not implemented")

        return self._summarize(divergences)

    def rate_estimation(self, successes: int, trials: int) -> dict[str, Any]:
        """Beta-Binomial posterior for rate."""
        if trials <= 0:
            return self._empty_summary()

        a_posterior = self.beta_prior_a + successes
        b_posterior = self.beta_prior_b + max(trials - successes, 0)

        samples = self.rng.beta(a_posterior, b_posterior, size=self.mc_samples)
        return self._summarize(samples)

    def aggregate_metrics(self, metrics: dict[str, dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
        """Aggregate by combining samples with weights."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self._empty_summary()

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Combine samples
        aggregated_samples = np.zeros(self.mc_samples)
        for name, metric_result in metrics.items():
            weight = normalized_weights.get(name, 0.0)
            if "samples" in metric_result and len(metric_result["samples"]) == self.mc_samples:
                aggregated_samples += weight * metric_result["samples"]
            else:
                # Fallback if no samples or wrong size
                aggregated_samples += weight * metric_result.get("mean", 0.0)

        return self._summarize(aggregated_samples)

    def dispersion_metric(self, values: dict[str, dict[str, Any]], center: str = "mean") -> dict[str, Any]:
        """Compute dispersion from samples."""
        if not values:
            return self._empty_summary()

        # Stack all samples
        all_samples = []
        for v in values.values():
            if "samples" in v and len(v["samples"]) == self.mc_samples:
                all_samples.append(v["samples"])
            else:
                # Fallback to point estimate if no samples or wrong size
                all_samples.append(np.full(self.mc_samples, v.get("mean", 0.0)))

        all_samples = np.array(all_samples)

        # Compute center for each MC iteration
        if center == "mean":
            center_samples = all_samples.mean(axis=0)
        elif center == "median":
            center_samples = np.median(all_samples, axis=0)
        else:
            raise ValueError(f"Unknown center type: {center}")

        # Compute dispersion for each MC iteration
        dispersion_samples = np.mean(np.abs(all_samples - center_samples), axis=0)

        return self._summarize(dispersion_samples)

    def _summarize(self, samples: np.ndarray) -> dict[str, Any]:
        """Convert samples to summary statistics."""
        if samples.size == 0:
            return self._empty_summary()

        alpha = (1.0 - self.ci_level) / 2.0
        return {
            "mean": float(samples.mean()),
            "ci_low": float(np.quantile(samples, alpha)),
            "ci_high": float(np.quantile(samples, 1.0 - alpha)),
            "samples": samples,
        }

    def _empty_summary(self) -> dict[str, Any]:
        """Return empty summary for edge cases."""
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "samples": np.zeros(self.mc_samples)}

    def get_result_type(self) -> str:
        return "distribution"
