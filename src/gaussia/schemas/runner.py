"""Runner interfaces for Gaussia."""

from abc import ABC, abstractmethod
from typing import Any

from .common import Batch, Dataset


class BaseRunner(ABC):
    """
    Abstract base class for test runners that execute conversational batches and datasets.

    Runners are responsible for invoking AI systems (agents, models, APIs, etc.) with test queries
    and collecting responses for evaluation.
    """

    @abstractmethod
    async def run_batch(self, batch: Batch, session_id: str, **kwargs: Any) -> tuple[Batch, bool, float]:
        """
        Execute a single batch (test case) and return the updated batch with response.

        Args:
            batch: Batch object containing the query to execute
            session_id: Session identifier for conversation context
            **kwargs: Additional runner-specific arguments

        Returns:
            tuple: (updated_batch, success, execution_time_ms)
                - updated_batch: Batch object with assistant response filled in
                - success: Boolean indicating whether execution succeeded
                - execution_time_ms: Execution time in milliseconds
        """

    @abstractmethod
    async def run_dataset(self, dataset: Dataset, **kwargs: Any) -> tuple[Dataset, dict[str, Any]]:
        """
        Execute all batches in a dataset and return updated dataset with responses.

        Args:
            dataset: Dataset object containing conversation batches to execute
            **kwargs: Additional runner-specific arguments

        Returns:
            tuple: (updated_dataset, execution_summary)
                - updated_dataset: Dataset with all batches filled with responses
                - execution_summary: Dictionary containing execution statistics:
                    - session_id: str
                    - total_batches: int
                    - successes: int
                    - failures: int
                    - total_execution_time_ms: float
                    - avg_batch_time_ms: float
        """


__all__ = ["BaseRunner"]
