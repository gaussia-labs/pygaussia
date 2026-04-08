"""Sentiment analyzer abstract base class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pygaussia.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from pygaussia.schemas.toxicity import SentimentScore


class SentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analysis models.

    This class provides a framework for implementing sentiment analysis models
    that return sentiment scores in a normalized range (typically [-1, 1]).

    The sentiment scores are used in the ASB (Associated Sentiment Bias) calculation
    to measure how sentiment varies across different demographic groups.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SentimentAnalyzer with a VerboseLogger.

        Args:
            **kwargs: Additional configuration parameters for specific implementations.
        """
        self.logger = VerboseLogger(kwargs.get("verbose", False))

    @abstractmethod
    def infer(self, text: str) -> "SentimentScore":
        """
        Analyze text and return a sentiment score.

        This method should be implemented by concrete SentimentAnalyzer classes
        to define their specific sentiment analysis logic.

        Args:
            text (str): The text to analyze for sentiment.

        Returns:
            SentimentScore: A SentimentScore object containing:
                - score: Sentiment score in range [-1, 1] where:
                    * -1 = most negative sentiment
                    * 0 = neutral sentiment
                    * 1 = most positive sentiment
                - confidence: Optional confidence/probability of the prediction
                - label: Optional categorical label (e.g., "positive", "negative", "neutral")

        Raises:
            NotImplementedError: If the concrete class does not implement this method.

        Example:
            >>> analyzer = ConcreteSentimentAnalyzer()
            >>> result = analyzer.infer("This is a great day!")
            >>> print(result.score)  # e.g., 0.85
            >>> print(result.label)  # e.g., "positive"
        """
        raise NotImplementedError("You should implement this method.")
