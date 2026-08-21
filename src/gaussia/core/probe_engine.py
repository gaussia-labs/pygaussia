"""Probe engine abstract base class: the only component with knowledge-base access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Catalogue, Document, Probe


class ProbeEngine(ABC):
    """Turns a knowledge base plus a user catalogue into tagged adversarial probes.

    Particularisation is the only place the knowledge base is reached (FR-020): an engine
    exposes what it read solely as ``Probe`` objects, so the Profiler and the graders
    never see the corpus (paper invariant 2).

    Two obligations no signature can express, and both are load-bearing:

    - The ``doc`` label on every hook must be derived from the engine's own knowledge of
      the base's boundary, never from an enumeration supplied for scoring (FR-021).
      Deriving it from the scoring enumeration would make the downstream absence test
      circular.
    - An engine that cannot decide absence must set ``KnowledgeHook.absence_reliable`` to
      ``False`` on the probes it emits (FR-023), so an unreliable label is never
      indistinguishable from a confirmed one. Similarity search is structurally unable to
      decide absence, because it never reveals what it failed to retrieve.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identity recorded on ``Probe.engine``.

        Recorded for reading and never branched on, which is what keeps the
        absence/breadth trade-off measurable after several engines compose (FR-022).
        """

    @property
    @abstractmethod
    def entity_kinds(self) -> frozenset[str]:
        """The ``StrategySpec.entity_kind`` values this engine can extract or retrieve.

        Declared rather than inferred because ``entity_kind`` is the user's own
        vocabulary and gaussia never learns what it means. Catalogue validation rejects a
        strategy whose entity kind no configured engine declares here, so a typo fails
        loudly instead of silently yielding no probes (FR-025).
        """

    @abstractmethod
    def can_handle(self, document: Document) -> bool:
        """Whether this engine can extract from ``document``.

        Args:
            document: One unit of the knowledge base. ``structured`` says whether the
                document's knowledge boundary is enumerable, which is what decides
                whether an engine can establish absence over it.

        Returns:
            ``True`` when the document is within this engine's reach. Engine selection is
            composition rather than a cascade: every engine answering ``True``
            contributes, and their outputs are merged.
        """

    @abstractmethod
    def generate(self, documents: list[Document], catalogue: Catalogue) -> list[Probe]:
        """Produce tagged probes from the documents this engine can handle.

        Args:
            documents: The knowledge base. An empty list means there is none, and the
                engine must then return domain-agnostic probes rather than nothing
                (FR-024).
            catalogue: The already-validated plugins and strategies. Each probe records
                the strategy that produced it, and carries no plugin when that strategy
                names none — which is the only mechanism by which a control is recognised
                (FR-026).

        Returns:
            The probes, each with a hook whose ``doc`` label and ``absence_reliable`` flag
            honour the obligations above.
        """
