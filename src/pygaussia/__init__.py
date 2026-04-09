"""
Gaussia: AI Evaluation Framework

A modular framework for measuring fairness, quality, and safety of AI assistant responses.
"""

from .core import (
    BaseGroupExtractor,
    Embedder,
    Gaussia,
    Guardian,
    Reranker,
    Retriever,
    ToxicityLoader,
)
from .schemas import (
    BaseMetric,
    Batch,
    Dataset,
)
from .statistical import BayesianMode, FrequentistMode, StatisticalMode

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Gaussia",
    "Retriever",
    "Guardian",
    "ToxicityLoader",
    "BaseGroupExtractor",
    "Embedder",
    "Reranker",
    "Batch",
    "Dataset",
    "BaseMetric",
    "StatisticalMode",
    "FrequentistMode",
    "BayesianMode",
]
