"""The enumeration engine: the strongest absence guarantee, and the reason it is opt-in.

An entity absent from a complete enumeration is absent from the base, which is exactly what a
``doc = 0`` label wants behind it. But enumerating "every entity of this kind that exists" is
irreducibly domain knowledge — no general library can derive it from a corpus — so the engine
takes an ``EntityEnumerator`` the user writes, and cannot be constructed without one (spec D14).
The refusal is structural rather than a check: the collaborator is a required argument, so an
engine that could not establish a boundary never comes into existence.

``Document.structured`` has its consumer here. It is the field that says whether a document's
knowledge boundary can be enumerated at all, and a document that says no is one this engine
declines rather than guesses over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .particularisation import ParticularisingEngine

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gaussia.core.entity_enumerator import EntityEnumerator
    from gaussia.schemas.roastme import Document


class EnumerationProbeEngine(ParticularisingEngine):
    """Probes whose grounding labels rest on a complete enumeration of the user's entities.

    Args:
        enumerator: The user's enumeration of their domain's entities. Required: without it
            there is no boundary, and a boundary is the whole of what this engine contributes.
        entity_kinds: The entity kinds this engine is being trusted with.
    """

    def __init__(self, enumerator: EntityEnumerator, entity_kinds: Iterable[str] = ()) -> None:
        super().__init__(entity_kinds)
        self._enumerator = enumerator

    @property
    def name(self) -> str:
        return "enumeration"

    @property
    def decides_absence(self) -> bool:
        # Completeness is the enumerator's contract, so absence from it is absence from the base.
        return True

    def can_handle(self, document: Document) -> bool:
        return document.structured

    def _entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        return self._enumerator.enumerate_entities(kind, documents)
