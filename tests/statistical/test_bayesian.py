"""Tests for Bayesian statistical mode."""

import numpy as np
import pytest

from pygaussia.statistical import BayesianMode


class TestBayesianMode:
    """Test suite for BayesianMode class."""

    def test_initialization_default_params(self):
        """Test BayesianMode initialization with default parameters."""
        mode = BayesianMode()
        assert mode.mc_samples == 5000
        assert mode.ci_level == 0.95
        assert mode.dirichlet_prior == 1.0
        assert mode.beta_prior_a == 1.0
        assert mode.beta_prior_b == 1.0

    def test_initialization_custom_params(self):
        """Test BayesianMode initialization with custom parameters."""
        mode = BayesianMode(
            mc_samples=10000, ci_level=0.90, dirichlet_prior=0.5, beta_prior_a=2.0, beta_prior_b=2.0, rng_seed=123
        )
        assert mode.mc_samples == 10000
        assert mode.ci_level == 0.90
        assert mode.dirichlet_prior == 0.5
        assert mode.beta_prior_a == 2.0
        assert mode.beta_prior_b == 2.0

    def test_get_result_type(self):
        """Test that result type is 'distribution'."""
        mode = BayesianMode()
        assert mode.get_result_type() == "distribution"

    def test_distribution_divergence_basic(self):
        """Test distribution divergence with basic counts."""
        mode = BayesianMode(mc_samples=1000, rng_seed=42)
        observed_counts = {"A": 30, "B": 70}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed_counts, reference)

        assert "mean" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert "samples" in result
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]
        assert len(result["samples"]) == 1000

    def test_distribution_divergence_uniform(self):
        """Test distribution divergence with uniform distribution."""
        mode = BayesianMode(mc_samples=1000, rng_seed=42)
        observed_counts = {"A": 50, "B": 50}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed_counts, reference)

        # With uniform observed and reference, divergence should be low
        assert result["mean"] < 0.2

    def test_distribution_divergence_empty_categories(self):
        """Test distribution divergence with empty categories."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)
        observed_counts = {}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed_counts, reference)

        assert result["mean"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0

    def test_distribution_divergence_missing_reference_key(self):
        """Test distribution divergence with missing reference key."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)
        observed_counts = {"A": 50, "B": 30, "C": 20}
        reference = {"A": 0.5, "B": 0.5}  # Missing 'C'

        result = mode.distribution_divergence(observed_counts, reference)

        assert "mean" in result
        assert result["mean"] >= 0

    def test_distribution_divergence_not_implemented_type(self):
        """Test distribution divergence raises error for unsupported divergence type."""
        mode = BayesianMode()
        observed_counts = {"A": 50, "B": 50}
        reference = {"A": 0.5, "B": 0.5}

        with pytest.raises(NotImplementedError):
            mode.distribution_divergence(observed_counts, reference, divergence_type="kl_divergence")

    def test_rate_estimation_basic(self):
        """Test rate estimation with basic counts."""
        mode = BayesianMode(mc_samples=1000, rng_seed=42)

        result = mode.rate_estimation(successes=70, trials=100)

        assert "mean" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert "samples" in result
        assert 0 <= result["mean"] <= 1
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    def test_rate_estimation_zero_trials(self):
        """Test rate estimation with zero trials."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        result = mode.rate_estimation(successes=0, trials=0)

        assert result["mean"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0

    def test_rate_estimation_negative_trials(self):
        """Test rate estimation with negative trials."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        result = mode.rate_estimation(successes=0, trials=-1)

        assert result["mean"] == 0.0

    def test_rate_estimation_all_successes(self):
        """Test rate estimation with all successes."""
        mode = BayesianMode(mc_samples=1000, rng_seed=42)

        result = mode.rate_estimation(successes=100, trials=100)

        assert result["mean"] > 0.9

    def test_rate_estimation_no_successes(self):
        """Test rate estimation with no successes."""
        mode = BayesianMode(mc_samples=1000, rng_seed=42)

        result = mode.rate_estimation(successes=0, trials=100)

        assert result["mean"] < 0.1

    def test_aggregate_metrics_basic(self):
        """Test aggregation of metrics."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        metrics = {
            "metric1": {"mean": 0.3, "samples": np.array([0.3] * 100)},
            "metric2": {"mean": 0.7, "samples": np.array([0.7] * 100)},
        }
        weights = {"metric1": 1.0, "metric2": 1.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert "mean" in result
        assert np.isclose(result["mean"], 0.5, atol=0.01)

    def test_aggregate_metrics_weighted(self):
        """Test aggregation of metrics with different weights."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        metrics = {
            "metric1": {"mean": 0.0, "samples": np.array([0.0] * 100)},
            "metric2": {"mean": 1.0, "samples": np.array([1.0] * 100)},
        }
        weights = {"metric1": 1.0, "metric2": 3.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert np.isclose(result["mean"], 0.75, atol=0.01)

    def test_aggregate_metrics_zero_weights(self):
        """Test aggregation with zero weights."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        metrics = {
            "metric1": {"mean": 0.5, "samples": np.array([0.5] * 100)},
        }
        weights = {"metric1": 0.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert result["mean"] == 0.0

    def test_aggregate_metrics_fallback_no_samples(self):
        """Test aggregation fallback when no samples present."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        metrics = {
            "metric1": {"mean": 0.5},  # No samples
            "metric2": {"mean": 0.5, "samples": np.array([0.5] * 100)},
        }
        weights = {"metric1": 1.0, "metric2": 1.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert "mean" in result

    def test_dispersion_metric_basic(self):
        """Test dispersion metric calculation."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        values = {
            "v1": {"mean": 0.2, "samples": np.array([0.2] * 100)},
            "v2": {"mean": 0.4, "samples": np.array([0.4] * 100)},
            "v3": {"mean": 0.6, "samples": np.array([0.6] * 100)},
        }

        result = mode.dispersion_metric(values, center="mean")

        assert "mean" in result
        assert result["mean"] >= 0

    def test_dispersion_metric_median_center(self):
        """Test dispersion metric with median center."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        values = {
            "v1": {"mean": 0.0, "samples": np.array([0.0] * 100)},
            "v2": {"mean": 0.5, "samples": np.array([0.5] * 100)},
            "v3": {"mean": 1.0, "samples": np.array([1.0] * 100)},
        }

        result = mode.dispersion_metric(values, center="median")

        assert "mean" in result
        assert result["mean"] >= 0

    def test_dispersion_metric_empty_values(self):
        """Test dispersion metric with empty values."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        result = mode.dispersion_metric({}, center="mean")

        assert result["mean"] == 0.0

    def test_dispersion_metric_invalid_center(self):
        """Test dispersion metric raises error for invalid center."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        values = {
            "v1": {"mean": 0.5, "samples": np.array([0.5] * 100)},
        }

        with pytest.raises(ValueError, match="Unknown center type"):
            mode.dispersion_metric(values, center="invalid")

    def test_dispersion_metric_fallback_no_samples(self):
        """Test dispersion metric fallback when no samples."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        values = {
            "v1": {"mean": 0.2},  # No samples
            "v2": {"mean": 0.8, "samples": np.array([0.8] * 100)},
        }

        result = mode.dispersion_metric(values, center="mean")

        assert "mean" in result

    def test_summarize_empty_samples(self):
        """Test _summarize with empty samples."""
        mode = BayesianMode(mc_samples=100, rng_seed=42)

        result = mode._summarize(np.array([]))

        assert result["mean"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0

    def test_summarize_normal_samples(self):
        """Test _summarize with normal samples."""
        mode = BayesianMode(mc_samples=1000, ci_level=0.95, rng_seed=42)
        samples = np.linspace(0, 1, 1000)

        result = mode._summarize(samples)

        assert np.isclose(result["mean"], 0.5, atol=0.01)
        assert result["ci_low"] < result["ci_high"]
        # With 95% CI, we expect ci_low ~= 0.025 and ci_high ~= 0.975
        assert np.isclose(result["ci_low"], 0.025, atol=0.01)
        assert np.isclose(result["ci_high"], 0.975, atol=0.01)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        mode1 = BayesianMode(mc_samples=100, rng_seed=42)
        mode2 = BayesianMode(mc_samples=100, rng_seed=42)

        observed = {"A": 30, "B": 70}
        reference = {"A": 0.5, "B": 0.5}

        result1 = mode1.distribution_divergence(observed, reference)
        result2 = mode2.distribution_divergence(observed, reference)

        assert np.isclose(result1["mean"], result2["mean"])

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        mode1 = BayesianMode(mc_samples=100, rng_seed=42)
        mode2 = BayesianMode(mc_samples=100, rng_seed=123)

        observed = {"A": 30, "B": 70}
        reference = {"A": 0.5, "B": 0.5}

        result1 = mode1.distribution_divergence(observed, reference)
        result2 = mode2.distribution_divergence(observed, reference)

        # Results should be different with different seeds
        # (though they could occasionally match by chance)
        assert not np.array_equal(result1["samples"], result2["samples"])
