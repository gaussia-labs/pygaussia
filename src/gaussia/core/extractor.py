"""Extractor abstract base class for feature extraction."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.toxicity import GroupDetection


class BaseGroupExtractor(ABC):
    """
    Abstract base class for group detection in text.

    This class provides a standardized interface for detecting whether a text
    mentions specific demographic or social groups.
    """

    @abstractmethod
    def detect_one(self, text: str) -> dict[str, "GroupDetection"]:
        """
        Detect group mentions in a single text.

        Args:
            text (str): The text to analyze.

        Returns:
            Dict[str, GroupDetection]: Dictionary mapping group names to detection results.
        """
        raise NotImplementedError("You should implement this method.")

    @abstractmethod
    def detect_batch(self, texts: list[str]) -> list[dict[str, "GroupDetection"]]:
        """
        Detect group mentions in a batch of texts.

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            List[Dict[str, GroupDetection]]: List of dictionaries, one per text,
                mapping group names to detection results.
        """
        raise NotImplementedError("You should implement this method.")
