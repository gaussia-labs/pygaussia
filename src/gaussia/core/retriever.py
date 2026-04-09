"""Retriever abstract base class for loading datasets."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.common import Dataset, IterationLevel, StreamedBatch


class Retriever(ABC):
    """
    Abstract base class for data retrieval from cold storage.

    This class serves as a template for implementing specific data retrieval strategies.
    Subclasses should implement the load_dataset method to fetch data from their respective storage systems.

    Attributes:
        kwargs (dict): Additional configuration parameters passed during initialization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Retriever with optional configuration parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        self.kwargs = kwargs

    @property
    def iteration_level(self) -> "IterationLevel":
        from gaussia.schemas.common import IterationLevel

        return IterationLevel.FULL_DATASET

    @abstractmethod
    def load_dataset(self) -> list["Dataset"] | Iterator["Dataset"] | Iterator["StreamedBatch"]:
        raise NotImplementedError("You should implement this method according to the type of storage you are using.")
