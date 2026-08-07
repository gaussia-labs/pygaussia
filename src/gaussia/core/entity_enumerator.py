"""Entity enumerator abstract base class. Gaussia specifies it and ships no implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Document


class EntityEnumerator(ABC):
    """Lists the entities of one kind that exist in the knowledge base.

    This is the strongest absence guarantee the method has: an entity absent from a
    complete enumeration is absent from the base, which is what a ``doc = 0`` label needs
    behind it. It is also irreducibly domain knowledge — enumerating "every article that
    exists" cannot be derived from a corpus by a general library — so gaussia specifies
    the interface and ships no implementation. The enumeration probe engine is opt-in for
    exactly that reason (spec D14) and refuses to run until one is injected.
    """

    @abstractmethod
    def enumerate_entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        """Return every entity of ``kind`` that exists in the knowledge base.

        Args:
            kind: The entity type, in the user's own vocabulary. The same string a
                ``StrategySpec`` names in ``entity_kind``.
            documents: The knowledge base to enumerate over.

        Returns:
            The complete set of existing entities of that kind. Completeness is the whole
            contract: an implementation that returns a sample turns every absence label
            derived from it into a guess.
        """
