"""Core abstractions and base classes for Gaussia.

Pipeline components (document_retriever, contradiction_checker) require numpy and should
be imported directly:
    from gaussia.core.document_retriever import DocumentRetriever, DocumentRetrieverConfig
    from gaussia.core.contradiction_checker import ContradictionChecker
"""

from .base import Gaussia
from .category_search import CategorySearch
from .embedder import Embedder
from .entity_enumerator import EntityEnumerator
from .exceptions import (
    GaussiaError,
    GuardianError,
    LoaderError,
    LogprobsExtractionError,
    LogprobsNotSupportedError,
    MetricError,
    RetrieverError,
    StatisticalModeError,
)
from .extractor import BaseGroupExtractor
from .fact_twister import FactTwister
from .grader import Grader
from .guardian import Guardian
from .hook_verifier import HookVerifier
from .loader import ToxicityLoader
from .on_profile_filter import OnProfileFilter
from .probe_engine import ProbeEngine
from .query_generator import QueryGenerator
from .realism_estimator import RealismEstimator
from .reranker import Reranker
from .retriever import Retriever
from .sentiment import SentimentAnalyzer
from .similarity_scorer import SimilarityScorer
from .target_assistant import TargetAssistant
from .transform import Transform

__all__ = [
    "BaseGroupExtractor",
    "CategorySearch",
    "Embedder",
    "EntityEnumerator",
    "FactTwister",
    "Gaussia",
    "GaussiaError",
    "Grader",
    "Guardian",
    "GuardianError",
    "HookVerifier",
    "LoaderError",
    "LogprobsExtractionError",
    "LogprobsNotSupportedError",
    "MetricError",
    "OnProfileFilter",
    "ProbeEngine",
    "QueryGenerator",
    "RealismEstimator",
    "Reranker",
    "Retriever",
    "RetrieverError",
    "SentimentAnalyzer",
    "SimilarityScorer",
    "StatisticalModeError",
    "TargetAssistant",
    "Transform",
]
