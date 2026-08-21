"""Fact twister abstract base class: a passage of the corpus to a grounded false premise."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import GroundedTwist


class FactTwister(ABC):
    """Anchors a fact in a passage of the knowledge base and derives a false premise from it.

    The eleventh interface, and it exists because the tenth cannot do this (FR-042).
    ``Transform.apply(entity) -> str`` receives an entity's name and nothing else, so the corpus is
    not on its signature and a fact of the corpus is therefore out of its reach. What that
    transformation can build is a premise *about a name* — a near miss, a flipped digit. What this
    builds is a premise about a **datum**, with the name left real.

    That difference is why a twister needs no boundary. It asserts nothing about absence: the entity
    it names is one the passage documents, and the falsehood is the value, the condition or the
    requirement attached to it. Nothing here claims an entity does not exist, so nothing here needs
    a complete enumeration to stand behind such a claim — which is the single reason this is the
    cheaper of the two paths to a false-premise probe.

    **The pattern is requested, never chosen.** Asked for whichever pattern it likes, a model
    collapses to one: over 24 passages of a bank's corpus, 21 twists came back as the same pattern
    and none as either of the two alternatives the prompt offered. That is precisely the gap the
    paper admits about its own RAG engine, which tags every twist it produces with one generic
    strategy id, leaving the qualitative patterns its configuration declares absent from every table
    it reports. So the pattern is an argument, and an implementation that ignores it and answers with
    another is answering a question nobody asked.

    Not a fifth transformation, and deliberately not joining that registry (FR-025 stays closed).
    The strings a twister declares in ``patterns`` do become strings a ``StrategySpec.transform`` may
    name — but only while a twister declaring them is configured, so a catalogue naming a pattern
    nothing can realise is still refused before generation rather than mid-run.
    """

    @property
    @abstractmethod
    def patterns(self) -> frozenset[str]:
        """The twist patterns this implementation realises.

        The strings a ``StrategySpec.transform`` may name when this twister is configured. Declared
        rather than discovered, for the same reason ``ProbeEngine.entity_kinds`` is: catalogue
        validation has to be able to reject a pattern nothing will realise, before a run starts.
        """

    @property
    @abstractmethod
    def model(self) -> str | None:
        """Identity of the model behind this twister, recorded on ``Probe.model`` (FR-046).

        ``None`` for an implementation that uses none. Recorded for reading, never branched on.
        """

    @abstractmethod
    def twist(self, passage: str, pattern: str, entity_kind: str, phrasing_hint: str) -> GroundedTwist | None:
        """Derive one grounded false premise from ``passage``, following ``pattern``.

        Args:
            passage: Text of the knowledge base. Not a whole document: the caller decides how the
                corpus is cut, and a passage small enough to hold one fact is what makes the anchor
                checkable against it afterwards.
            pattern: Which twist to apply, one of ``patterns``. An implementation MUST NOT
                substitute another, because the strategy that asked for it is the only thing that
                tells a report which qualitative pattern a failure was found under.
            entity_kind: The user's own word for the kind of thing to anchor on, from
                ``StrategySpec.entity_kind``. Never interpreted by gaussia — passed through so the
                implementation can put it in front of whatever reads the passage.
            phrasing_hint: The strategy's ``phrasing_hint``, which is the user's statement of how
                their traffic sounds. The query this returns is sent to the assistant verbatim, so
                the hint is guidance for writing it rather than a template to fill.

        Returns:
            The twist, or ``None`` when this passage yields no fact this pattern can operate on.
            ``None`` rather than a degraded twist: a passage with no numeric value in it cannot
            support a flipped one, and inventing something to return would produce a probe whose
            premise is false for a reason the catalogue did not ask for.
        """
