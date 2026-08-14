from __future__ import annotations

from typing import TYPE_CHECKING

from gaussia.core.entity_enumerator import EntityEnumerator

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Document


class ConfigEntityEnumerator(EntityEnumerator):
    def __init__(self, entities: dict[str, frozenset[str]]) -> None:
        self._entities = entities

    def enumerate_entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        return self._entities[kind]
