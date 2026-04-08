"""Tests for Bias metric."""

import pytest

from pygaussia.core import Guardian
from pygaussia.metrics.bias import Bias
from pygaussia.schemas.bias import BiasMetric, GuardianBias, ProtectedAttribute
from pygaussia.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import MockRetriever


class MockGuardian(Guardian):
    """Mock Guardian for testing bias detection."""

    def __init__(self, always_biased=False, certainty=0.9, **kwargs):
        self.always_biased = always_biased
        self.certainty = certainty
        self.call_count = 0

    def is_biased(self, question, answer, attribute, context=None):
        """Return mock bias detection result."""
        self.call_count += 1
        return GuardianBias(is_biased=self.always_biased, attribute=attribute.attribute.value, certainty=self.certainty)


class MockGuardianAlternating(Guardian):
    """Mock Guardian that alternates between biased and not biased."""

    def __init__(self, **kwargs):
        self.call_count = 0

    def is_biased(self, question, answer, attribute, context=None):
        """Return alternating bias detection result."""
        self.call_count += 1
        is_biased = self.call_count % 2 == 0
        return GuardianBias(is_biased=is_biased, attribute=attribute.attribute.value, certainty=0.85)


class TestBiasMetric:
    """Test suite for Bias metric."""

    def test_initialization_basic(self):
        """Test Bias initialization with basic parameters."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian)

        assert bias.guardian is not None
        assert isinstance(bias.statistical_mode, FrequentistMode)
        assert len(bias.protected_attributes) == 5

    def test_initialization_custom_statistical_mode(self):
        """Test Bias initialization with custom statistical mode."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian, statistical_mode=BayesianMode())

        assert isinstance(bias.statistical_mode, BayesianMode)

    def test_protected_attributes_defined(self):
        """Test that all protected attributes are defined."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian)

        expected_attributes = {
            ProtectedAttribute.Attribute.gender,
            ProtectedAttribute.Attribute.race,
            ProtectedAttribute.Attribute.religion,
            ProtectedAttribute.Attribute.nationality,
            ProtectedAttribute.Attribute.sexual_orientation,
        }

        actual_attributes = {attr.attribute for attr in bias.protected_attributes}
        assert actual_attributes == expected_attributes

    def test_get_guardian_biased_attributes(self):
        """Test _get_guardian_biased_attributes method."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        result = bias._get_guardian_biased_attributes(
            batch=batches, attributes=bias.protected_attributes, context="test context"
        )

        assert len(result) == 5
        for attr in bias.protected_attributes:
            assert attr.attribute.value in result
            assert len(result[attr.attribute.value]) == 2

    def test_calculate_attribute_rates_frequentist(self):
        """Test _calculate_attribute_rates with frequentist mode."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian)

        biases_by_attributes = {}
        for attr in bias.protected_attributes:
            biases_by_attributes[attr.attribute.value] = [
                BiasMetric.GuardianInteraction(
                    qa_id="qa_001", is_biased=False, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_002", is_biased=False, attribute=attr.attribute.value, certainty=0.85
                ),
            ]

        result = bias._calculate_attribute_rates(biases_by_attributes)

        assert len(result) == 5
        for rate in result:
            assert isinstance(rate, BiasMetric.AttributeBiasRate)
            assert rate.n_samples == 2
            assert rate.k_biased == 0
            assert rate.rate == 0.0
            assert rate.ci_low is None
            assert rate.ci_high is None

    def test_calculate_attribute_rates_bayesian(self):
        """Test _calculate_attribute_rates with Bayesian mode."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian, statistical_mode=BayesianMode(mc_samples=100))

        biases_by_attributes = {}
        for attr in bias.protected_attributes:
            biases_by_attributes[attr.attribute.value] = [
                BiasMetric.GuardianInteraction(
                    qa_id="qa_001", is_biased=True, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_002", is_biased=False, attribute=attr.attribute.value, certainty=0.85
                ),
            ]

        result = bias._calculate_attribute_rates(biases_by_attributes)

        assert len(result) == 5
        for rate in result:
            assert isinstance(rate, BiasMetric.AttributeBiasRate)
            assert rate.ci_low is not None
            assert rate.ci_high is not None
            assert rate.ci_low <= rate.rate <= rate.ci_high

    def test_batch_processing(self):
        """Test batch method processes correctly."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian)

        batches = [
            create_sample_batch(qa_id="qa_001"),
            create_sample_batch(qa_id="qa_002"),
        ]

        bias.batch(session_id="test_session", context="test context", assistant_id="test_assistant", batch=batches)

        assert len(bias.metrics) == 1
        metric = bias.metrics[0]
        assert isinstance(metric, BiasMetric)
        assert metric.session_id == "test_session"
        assert metric.assistant_id == "test_assistant"
        assert len(metric.attribute_rates) == 5
        assert len(metric.guardian_interactions) == 5

    def test_batch_multiple_interactions(self):
        """Test batch method with multiple interactions."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian, verbose=False)

        batches = [create_sample_batch(qa_id=f"qa_{i}") for i in range(5)]

        bias.batch(session_id="test_session", context="test context", assistant_id="test_assistant", batch=batches)

        assert len(bias.metrics) == 1
        metric = bias.metrics[0]
        for attr_key, interactions in metric.guardian_interactions.items():
            assert len(interactions) == 5

    def test_batch_with_alternating_guardian(self):
        """Test batch with alternating biased/not biased responses."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardianAlternating)

        batches = [create_sample_batch(qa_id=f"qa_{i}") for i in range(4)]

        bias.batch(session_id="test_session", context="test context", assistant_id="test_assistant", batch=batches)

        metric = bias.metrics[0]
        for attr_key, interactions in metric.guardian_interactions.items():
            biased_count = sum(1 for i in interactions if i.is_biased)
            not_biased_count = sum(1 for i in interactions if not i.is_biased)
            assert biased_count + not_biased_count == 4

    def test_bias_rate_with_mixed_results(self):
        """Test bias rate calculation when some responses are biased."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardianAlternating)

        biases_by_attributes = {}
        for attr in bias.protected_attributes:
            biases_by_attributes[attr.attribute.value] = [
                BiasMetric.GuardianInteraction(
                    qa_id="qa_001", is_biased=True, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_002", is_biased=False, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_003", is_biased=True, attribute=attr.attribute.value, certainty=0.9
                ),
                BiasMetric.GuardianInteraction(
                    qa_id="qa_004", is_biased=False, attribute=attr.attribute.value, certainty=0.9
                ),
            ]

        result = bias._calculate_attribute_rates(biases_by_attributes)

        for rate in result:
            assert rate.n_samples == 4
            assert rate.k_biased == 2
            assert rate.rate == pytest.approx(0.5, abs=0.001)

    def test_verbose_mode(self):
        """Test that verbose mode doesn't break initialization."""
        bias = Bias(retriever=MockRetriever, guardian=MockGuardian, verbose=True)

        assert bias.verbose is True
