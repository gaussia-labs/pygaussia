"""Tests for Frequentist statistical mode."""

import pytest

from gaussia.statistical import FrequentistMode


class TestFrequentistMode:
    """Test suite for FrequentistMode class."""

    def test_initialization(self):
        """Test FrequentistMode initialization."""
        mode = FrequentistMode()
        assert mode is not None

    def test_get_result_type(self):
        """Test that result type is 'point_estimate'."""
        mode = FrequentistMode()
        assert mode.get_result_type() == "point_estimate"

    def test_distribution_divergence_basic(self):
        """Test distribution divergence with basic proportions."""
        mode = FrequentistMode()
        observed = {"A": 0.3, "B": 0.7}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed, reference)

        assert isinstance(result, float)
        assert result == pytest.approx(0.2, abs=0.001)

    def test_distribution_divergence_identical(self):
        """Test distribution divergence with identical distributions."""
        mode = FrequentistMode()
        observed = {"A": 0.5, "B": 0.5}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed, reference)

        assert result == pytest.approx(0.0, abs=0.001)

    def test_distribution_divergence_max_different(self):
        """Test distribution divergence with maximally different distributions."""
        mode = FrequentistMode()
        observed = {"A": 1.0, "B": 0.0}
        reference = {"A": 0.0, "B": 1.0}

        result = mode.distribution_divergence(observed, reference)

        assert result == pytest.approx(1.0, abs=0.001)

    def test_distribution_divergence_missing_keys(self):
        """Test distribution divergence with missing keys."""
        mode = FrequentistMode()
        observed = {"A": 0.5, "B": 0.3, "C": 0.2}
        reference = {"A": 0.5, "B": 0.5}

        result = mode.distribution_divergence(observed, reference)

        assert isinstance(result, float)
        assert result >= 0

    def test_distribution_divergence_not_implemented_type(self):
        """Test distribution divergence raises error for unsupported type."""
        mode = FrequentistMode()
        observed = {"A": 0.5, "B": 0.5}
        reference = {"A": 0.5, "B": 0.5}

        with pytest.raises(NotImplementedError):
            mode.distribution_divergence(observed, reference, divergence_type="kl_divergence")

    def test_distribution_divergence_empty(self):
        """Test distribution divergence with empty distributions."""
        mode = FrequentistMode()
        observed = {}
        reference = {}

        result = mode.distribution_divergence(observed, reference)

        assert result == 0.0

    def test_rate_estimation_basic(self):
        """Test rate estimation with basic counts."""
        mode = FrequentistMode()

        result = mode.rate_estimation(successes=70, trials=100)

        assert isinstance(result, float)
        assert result == pytest.approx(0.7, abs=0.001)

    def test_rate_estimation_zero_trials(self):
        """Test rate estimation with zero trials."""
        mode = FrequentistMode()

        result = mode.rate_estimation(successes=0, trials=0)

        assert result == 0.0

    def test_rate_estimation_all_successes(self):
        """Test rate estimation with all successes."""
        mode = FrequentistMode()

        result = mode.rate_estimation(successes=100, trials=100)

        assert result == 1.0

    def test_rate_estimation_no_successes(self):
        """Test rate estimation with no successes."""
        mode = FrequentistMode()

        result = mode.rate_estimation(successes=0, trials=100)

        assert result == 0.0

    def test_aggregate_metrics_basic(self):
        """Test aggregation of metrics with equal weights."""
        mode = FrequentistMode()

        metrics = {"metric1": 0.3, "metric2": 0.7}
        weights = {"metric1": 1.0, "metric2": 1.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert isinstance(result, float)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_aggregate_metrics_weighted(self):
        """Test aggregation with different weights."""
        mode = FrequentistMode()

        metrics = {"metric1": 0.0, "metric2": 1.0}
        weights = {"metric1": 1.0, "metric2": 3.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert result == pytest.approx(0.75, abs=0.001)

    def test_aggregate_metrics_zero_weights(self):
        """Test aggregation with zero total weight."""
        mode = FrequentistMode()

        metrics = {"metric1": 0.5}
        weights = {"metric1": 0.0}

        result = mode.aggregate_metrics(metrics, weights)

        assert result == 0.0

    def test_dispersion_metric_mean_center(self):
        """Test dispersion metric with mean center."""
        mode = FrequentistMode()

        values = {"v1": 0.2, "v2": 0.4, "v3": 0.6}

        result = mode.dispersion_metric(values, center="mean")

        assert isinstance(result, float)
        # Mean is 0.4, deviations are |0.2-0.4|=0.2, |0.4-0.4|=0.0, |0.6-0.4|=0.2
        # MAD = (0.2 + 0.0 + 0.2) / 3 = 0.133...
        assert result == pytest.approx(0.133, abs=0.01)

    def test_dispersion_metric_median_center(self):
        """Test dispersion metric with median center."""
        mode = FrequentistMode()

        values = {"v1": 0.0, "v2": 0.5, "v3": 1.0}

        result = mode.dispersion_metric(values, center="median")

        assert isinstance(result, float)
        # Median is 0.5, deviations are |0-0.5|=0.5, |0.5-0.5|=0, |1-0.5|=0.5
        # MAD = (0.5 + 0 + 0.5) / 3 = 0.333...
        assert result == pytest.approx(0.333, abs=0.01)

    def test_dispersion_metric_empty_values(self):
        """Test dispersion metric with empty values."""
        mode = FrequentistMode()

        result = mode.dispersion_metric({}, center="mean")

        assert result == 0.0

    def test_dispersion_metric_invalid_center(self):
        """Test dispersion metric raises error for invalid center."""
        mode = FrequentistMode()

        values = {"v1": 0.5}

        with pytest.raises(ValueError, match="Unknown center type"):
            mode.dispersion_metric(values, center="invalid")

    def test_dispersion_metric_single_value(self):
        """Test dispersion metric with single value."""
        mode = FrequentistMode()

        values = {"v1": 0.5}

        result = mode.dispersion_metric(values, center="mean")

        # Single value, dispersion should be 0
        assert result == 0.0

    def test_dispersion_metric_uniform_values(self):
        """Test dispersion metric with all same values."""
        mode = FrequentistMode()

        values = {"v1": 0.5, "v2": 0.5, "v3": 0.5}

        result = mode.dispersion_metric(values, center="mean")

        assert result == 0.0
