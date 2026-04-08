"""Loader abstract base classes for external data."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygaussia.schemas.toxicity import ToxicityDataset


class ToxicityLoader(ABC):
    """Abstract base class for loading toxicity datasets.

    This class serves as a template for implementing custom toxicity dataset loaders.
    It provides a standardized interface for loading toxicity-related datasets that
    can be used for training or evaluation of toxicity detection models.

    The class is designed to be extended by concrete implementations that handle
    specific data sources or formats. Each implementation must provide its own
    logic for loading and processing the toxicity data.

    Attributes:
        kwargs (dict): Configuration parameters passed during initialization.
            These parameters can be used by concrete implementations to customize
            the loading behavior.

    Example:
        To create a custom loader:
        ```python
        class CustomToxicityLoader(ToxicityLoader):
            def load(self, language: str) -> list[ToxicityDataset]:
                # Implementation specific to your data source
                pass
        ```
    """

    def __init__(self, **kwargs):
        """
        Initialize the ToxicityLoader with optional configuration parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        self.kwargs = kwargs

    @abstractmethod
    def load(self, language: str) -> list["ToxicityDataset"]:
        """Load and return a list of toxicity datasets.

        This method must be implemented by concrete subclasses to provide
        the actual dataset loading logic.

        Args:
            language (str): The language for which to load toxicity data.

        Returns:
            list[ToxicityDataset]: A list of loaded toxicity datasets.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError("You should implement this method.")
