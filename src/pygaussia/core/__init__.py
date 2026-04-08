"""Core abstractions and base classes for Gaussia.

Pipeline components (document_retriever, contradiction_checker) require numpy and should
be imported directly:
    from pygaussia.core.document_retriever import DocumentRetriever, DocumentRetrieverConfig
    from pygaussia.core.contradiction_checker import ContradictionChecker
"""

from .base import Gaussia
from .embedder import Embedder
from .exceptions import (
    GaussiaError,
    GuardianError,
    LoaderError,
    MetricError,
    RetrieverError,
    StatisticalModeError,
)
from .extractor import BaseGroupExtractor
from .guardian import Guardian
from .loader import ToxicityLoader
from .reranker import Reranker
from .retriever import Retriever
from .sentiment import SentimentAnalyzer
from .similarity_scorer import SimilarityScorer

__all__ = [
    "BaseGroupExtractor",
    "Embedder",
    "Gaussia",
    "GaussiaError",
    "Guardian",
    "GuardianError",
    "LoaderError",
    "MetricError",
    "Reranker",
    "Retriever",
    "RetrieverError",
    "SentimentAnalyzer",
    "SimilarityScorer",
    "StatisticalModeError",
    "ToxicityLoader",
]
