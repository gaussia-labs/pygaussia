"""Base corpus connector interface for regulatory compliance metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RegulatoryDocument:
    """A regulatory document loaded from the corpus."""

    text: str
    source: str


class CorpusConnector(ABC):
    """
    Abstract base class for loading regulatory corpus documents.

    Implementations provide access to markdown files from different storage backends.
    """

    @abstractmethod
    def load_documents(self) -> list[RegulatoryDocument]:
        """
        Load all regulatory documents from the corpus.

        Returns:
            List of RegulatoryDocument objects containing text and source metadata.
        """
        raise NotImplementedError
