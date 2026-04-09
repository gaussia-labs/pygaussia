"""Statistical modes for Gaussia metrics."""

from .base import StatisticalMode
from .bayesian import BayesianMode
from .frequentist import FrequentistMode

__all__ = [
    "BayesianMode",
    "FrequentistMode",
    "StatisticalMode",
]
