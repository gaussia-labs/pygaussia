"""Statistical mode abstract base class."""

from abc import ABC, abstractmethod
from typing import Any


class StatisticalMode(ABC):
    """
    Provides generic statistical primitives.
    Agnostic to domain-specific metrics (DR, DTO, ASB, etc.)

    This abstract class defines the interface for statistical computation modes.
    Implementations provide either point estimates (frequentist) or full
    distributions (Bayesian).
    """

    @abstractmethod
    def distribution_divergence(
        self, observed: dict[str, int | float], reference: dict[str, float], divergence_type: str = "total_variation"
    ) -> float | dict[str, Any]:
        """
        Compute divergence between two probability distributions.

        Args:
            observed: Observed distribution or counts {category: value}
            reference: Reference distribution {category: probability}
            divergence_type: Type of divergence measure (default: "total_variation")

        Returns:
            Frequentist: float (point estimate)
            Bayesian: dict with 'mean', 'samples', 'ci_low', 'ci_high'
        """
        raise NotImplementedError

    @abstractmethod
    def rate_estimation(self, successes: int, trials: int) -> float | dict[str, Any]:
        """
        Estimate a rate/proportion with uncertainty.

        Args:
            successes: Number of successes
            trials: Number of trials

        Returns:
            Frequentist: float (point estimate)
            Bayesian: dict with 'mean', 'samples', 'ci_low', 'ci_high'
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate_metrics(
        self, metrics: dict[str, float | dict[str, Any]], weights: dict[str, float]
    ) -> float | dict[str, Any]:
        """
        Aggregate multiple metrics with weights.

        Args:
            metrics: {metric_name: value} where value is from other primitives
            weights: {metric_name: weight}

        Returns:
            Frequentist: float (weighted sum)
            Bayesian: dict with 'mean', 'samples', 'ci_low', 'ci_high'
        """
        raise NotImplementedError

    @abstractmethod
    def dispersion_metric(
        self, values: dict[str, float | dict[str, Any]], center: str = "mean"
    ) -> float | dict[str, Any]:
        """
        Compute dispersion/spread of values around a center.
        Useful for metrics like DTO that measure deviation.

        Args:
            values: {group: value} where value can be point estimate or distribution
            center: Type of center to use ("mean" or "median")

        Returns:
            Frequentist: float (e.g., mean absolute deviation)
            Bayesian: dict with uncertainty
        """
        raise NotImplementedError

    @abstractmethod
    def get_result_type(self) -> str:
        """
        Return the type of result this mode produces.

        Returns:
            "point_estimate" for frequentist, "distribution" for Bayesian
        """
        raise NotImplementedError
