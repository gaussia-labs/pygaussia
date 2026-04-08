"""Corpus connectors for loading regulatory documents."""

from .base import CorpusConnector, RegulatoryDocument
from .local import LocalCorpusConnector

__all__ = [
    "CorpusConnector",
    "LocalCorpusConnector",
    "RegulatoryDocument",
]
