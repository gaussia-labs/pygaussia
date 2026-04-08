"""Gaussia base class for metrics."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from pygaussia.utils.logging import VerboseLogger

if TYPE_CHECKING:
    from pygaussia.core.retriever import Retriever
    from pygaussia.schemas.common import Batch
    from pygaussia.statistical.base import StatisticalMode


class Gaussia(ABC):
    """
    Abstract base class for implementing fairness metrics and analysis.

    This class provides the framework for processing datasets and computing fairness metrics.
    Subclasses should implement the batch method to define specific metric calculations.

    Attributes:
        retriever (Type[Retriever]): The retriever class to use for loading data.
        metrics (list): List to store computed metrics.
        dataset (list[Dataset]): The loaded dataset for processing.
        verbose (bool): Whether to enable verbose logging.
        logger (VerboseLogger): Logger instance for verbose logging.
    """

    def __init__(self, retriever: type["Retriever"], verbose: bool = False, **kwargs):
        """
        Initialize Gaussia with a data retriever.

        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            verbose (bool): Whether to enable verbose logging.
            **kwargs: Additional configuration parameters.
        """
        self.retriever = retriever(**kwargs)
        self.metrics = []
        self.verbose = verbose
        self.logger = VerboseLogger(verbose)

        self.dataset = self.retriever.load_dataset()
        self.level = self.retriever.iteration_level

        if isinstance(self.dataset, Iterator) and self.level.value == "full_dataset":
            raise ValueError(
                "When using a generator, you must explicitly set 'iteration_level' ('stream_sessions' or 'stream_batches') in the Retriever."
            )

        strategies = {
            "full_dataset": self._process_dataset,
            "stream_sessions": self._process_dataset,
            "stream_batches": self._process_qa,
        }

        self._iteration_processor = strategies.get(self.level.value)
        if not self._iteration_processor:
            raise ValueError(f"Unknown iteration_level: {self.level}")

    @abstractmethod
    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list["Batch"],
        language: str | None,
    ):
        """
        Process a single batch of conversation data.

        This method should be implemented by subclasses to define how each batch
        of conversation data is processed and what metrics are computed.

        Args:
            session_id (str): Unique identifier for the conversation session.
            context (str): Contextual information for the conversation.
            assistant_id (str): Identifier for the AI assistant.
            batch (list[Batch]): List of conversation batches to process.
            language (Optional[str]): Language of the conversation, if specified.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Should be implemented by each metric")

    @classmethod
    def run(cls, retriever: type["Retriever"], **kwargs) -> list:
        """
        Run the metric analysis on the entire dataset.

        This class method provides a convenient way to instantiate and run the metric
        analysis in one step.

        Args:
            retriever (Type[Retriever]): The retriever class to use for loading data.
            **kwargs: Additional configuration parameters.

        Returns:
            list: The computed metrics for the entire dataset.
        """
        return cls(retriever, **kwargs)._process()

    def _process_dataset(self, data):
        self.logger.info("Processing using dataset/session level parsing")
        for element in data:
            self.logger.info(f"Session ID: {element.session_id}, Assistant ID: {element.assistant_id}")
            self.batch(
                session_id=element.session_id,
                context=element.context,
                assistant_id=element.assistant_id,
                batch=element.conversation,
                language=element.language,
            )

    def _process_qa(self, data):
        self.logger.info("Processing using QA batch level parsing")
        for streamed in data:
            self.logger.info(
                f"Session ID: {streamed.metadata.session_id}, Assistant ID: {streamed.metadata.assistant_id}"
            )
            self.batch(
                session_id=streamed.metadata.session_id,
                context=streamed.metadata.context,
                assistant_id=streamed.metadata.assistant_id,
                batch=[streamed.batch],
                language=streamed.metadata.language,
            )

    def _resolve_weights(self, batch: list["Batch"]) -> dict[str, float]:
        n = len(batch)
        if n == 0:
            return {}

        weighted = {b.qa_id: b.weight for b in batch if b.weight is not None}
        unweighted_ids = [b.qa_id for b in batch if b.weight is None]

        if not weighted:
            return {b.qa_id: 1.0 / n for b in batch}

        if not unweighted_ids:
            total = sum(weighted.values())
            if abs(total - 1.0) > 1e-6:
                self.logger.warning(f"Provided weights sum to {total:.4f}, not 1.0. Falling back to equal weights.")
                return {b.qa_id: 1.0 / n for b in batch}
            return weighted

        assigned = sum(weighted.values())
        if assigned >= 1.0:
            self.logger.warning(
                f"Provided weights sum to {assigned:.4f} >= 1.0 with {len(unweighted_ids)} unweighted "
                "interaction(s). Falling back to equal weights."
            )
            return {b.qa_id: 1.0 / n for b in batch}
        remaining = 1.0 - assigned
        per_unweighted = remaining / len(unweighted_ids)
        return {**weighted, **{qa_id: per_unweighted for qa_id in unweighted_ids}}

    def _aggregate_scores(
        self,
        scores: list[float],
        batches: list["Batch"],
        weights: dict[str, float],
        statistical_mode: "StatisticalMode",
    ) -> tuple[float, float | None, float | None]:
        if statistical_mode.get_result_type() == "point_estimate":
            metrics_dict = {b.qa_id: score for b, score in zip(batches, scores, strict=False)}
            result = statistical_mode.aggregate_metrics(metrics_dict, weights)
            return float(result), None, None

        mc_samples = getattr(statistical_mode, "mc_samples", 5000)
        ci_level = getattr(statistical_mode, "ci_level", 0.95)
        rng = getattr(statistical_mode, "rng", np.random.default_rng())
        np_scores = np.array(scores)
        np_weights = np.array([weights[b.qa_id] for b in batches])
        total_weight = np_weights.sum()
        if total_weight <= 0.0:
            self.logger.warning(
                "Non-positive total weight encountered during aggregation. Falling back to equal weights."
            )
            np_weights = np.ones_like(np_weights, dtype=float) / len(np_weights)
        else:
            np_weights = np_weights / total_weight

        n_scores = len(scores)
        bootstrap_indices = rng.choice(n_scores, size=(mc_samples, n_scores), replace=True, p=np_weights)
        bootstrap_means = np_scores[bootstrap_indices].mean(axis=1)

        alpha = (1.0 - ci_level) / 2.0
        return (
            float(np.mean(bootstrap_means)),
            float(np.quantile(bootstrap_means, alpha)),
            float(np.quantile(bootstrap_means, 1.0 - alpha)),
        )

    def on_process_complete(self):  # noqa: B027
        """Optional hook evaluated after all dataset elements are processed. Useful for accumulator metrics."""

    def _process(self) -> list:
        """
        Process the entire dataset and compute metrics using the configured strategy.
        """
        self.logger.info("Starting to process dataset")

        self._iteration_processor(self.dataset)
        self.on_process_complete()

        self.logger.info(f"Completed processing. Total metrics collected: {len(self.metrics)}")
        return self.metrics
