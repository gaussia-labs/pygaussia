"""Abstract PII detector strategy for the Privacy metric.

`PIIDetector` is a Pydantic model rather than a plain ABC so that the contextual
scalars required by the paper (`domain_fit`, `regulatory_fit`) are validated at
construction: omitting them or supplying a value outside ``[0, 1]`` raises a
``ValidationError`` (FR-005). Pydantic v2's ``ModelMetaclass`` derives from
``ABCMeta``, so ``@abstractmethod`` still forbids instantiating an incomplete
subclass.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from gaussia.schemas.privacy import Span


class PIIDetector(BaseModel):
    """Black-box adapter over a concrete PII detection backend.

    Concrete subclasses translate a backend (Presidio, a HuggingFace pipeline, …)
    into the shared ``predict`` / ``supported_classes`` contract consumed by the
    ``Privacy`` and ``PrivacyRanker`` metrics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    domain_fit: float = Field(ge=0.0, le=1.0)
    regulatory_fit: float = Field(ge=0.0, le=1.0)

    @property
    @abstractmethod
    def supported_classes(self) -> frozenset[str]:
        """Domain classes the backend advertises it can detect."""

    @abstractmethod
    def predict(self, text: str) -> list[Span]:
        """Return the PII spans the backend finds in ``text``."""

    def setup(self) -> None:
        """One-time initialisation hook (e.g. model loading).

        Called by ``Privacy`` / ``PrivacyRanker`` exactly once before any
        ``predict`` call. Default is a no-op; subclasses override it when they
        need lazy heavy initialisation.
        """
