"""BaseOptimizer abstract class for prompt optimization strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.core.retriever import Retriever
    from gaussia.prompt_optimizer.schemas import OptimizationResult


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    Follows the same pattern as Gaussia metrics:
        GEPAOptimizer.run(RetrieverClass, seed_prompt=..., model=..., ...)
    """

    def __init__(self, retriever: type["Retriever"], **kwargs):
        from gaussia.schemas.common import Dataset

        raw = retriever(**kwargs).load_dataset()
        items = list(raw)
        if items and not isinstance(items[0], Dataset):
            raise TypeError(
                "Prompt optimizers require a Retriever that returns list[Dataset]. "
                "StreamedBatch retrievers are not supported."
            )
        self.dataset: list[Dataset] = [item for item in items if isinstance(item, Dataset)]

    @classmethod
    def run(cls, retriever: type["Retriever"], **kwargs) -> "OptimizationResult":
        return cls(retriever, **kwargs)._optimize()

    @abstractmethod
    def _optimize(self) -> "OptimizationResult":
        raise NotImplementedError
