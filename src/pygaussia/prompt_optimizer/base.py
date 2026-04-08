"""BaseOptimizer abstract class for prompt optimization strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygaussia.core.retriever import Retriever
    from pygaussia.prompt_optimizer.schemas import OptimizationResult


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    Follows the same pattern as Gaussia metrics:
        GEPAOptimizer.run(RetrieverClass, seed_prompt=..., model=..., ...)
    """

    def __init__(self, retriever: type["Retriever"], **kwargs):
        from pygaussia.schemas.common import Dataset

        raw = retriever(**kwargs).load_dataset()
        self.dataset: list[Dataset] = list(raw)  # type: ignore[arg-type]

        if self.dataset and not isinstance(self.dataset[0], Dataset):
            raise TypeError(
                "Prompt optimizers require a Retriever that returns list[Dataset]. "
                "StreamedBatch retrievers are not supported."
            )

    @classmethod
    def run(cls, retriever: type["Retriever"], **kwargs) -> "OptimizationResult":
        return cls(retriever, **kwargs)._optimize()

    @abstractmethod
    def _optimize(self) -> "OptimizationResult":
        raise NotImplementedError
