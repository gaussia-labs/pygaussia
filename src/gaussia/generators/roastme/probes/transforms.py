"""The four transformations a catalogue may name, and the registry that resolves them.

FR-025 closes the set at four, because ``transform`` is the one catalogue field whose value
changes what a probe *means*: an unrecognised transformation would produce probes whose ``doc``
label nobody can trust. A fifth therefore needs FR-025 relaxed as well as a class added here.

This registry is the single point where the catalogue's string becomes behaviour. Validation
checks membership, an engine resolves the key once per strategy, and nothing anywhere branches
on the string; it survives only on ``KnowledgeHook.how``, as a record rather than a dispatch.

Two limits stated rather than hidden. A transformation decides the *text* of a premise and
never its label — whether the result is documented is the engine's call, derived from its own
knowledge of the base's boundary (FR-021), so a transformation that leaves an entity untouched
simply yields a documented hook. And the fact flip carries an English negation: a corpus in
another language needs the premise built differently, which is a consequence of FR-025 closing
the set rather than a choice this module makes.

The order below is the order the method leans on them: the three that invent a premise first,
since a false premise is the attack, and the one that keeps the entity real last, since that is
what a control strategy asks for.

The four keys — ``mutate_to_fake``, ``flip_value``, ``flip_fact``, ``keep_real`` — are the ones
the published catalogue already uses and the ones the specification's own worked example names.
They are interface, not naming preference: a catalogue is data the user has already written, so
renaming a key here would reject a catalogue that was valid before.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING

from gaussia.core.transform import Transform

if TYPE_CHECKING:
    from collections.abc import Mapping

_NEAR_MISS_SUFFIX = "-2"
_NEGATION = "not "
_FIGURE = re.compile(r"\d+")


def _next_figure(match: re.Match[str]) -> str:
    return str(int(match.group()) + 1)


class InventNearMissEntity(Transform):
    """A real entity into a sibling the base does not contain.

    A near miss rather than a nonsense string: the probe is meant to test whether the assistant
    is grounded, not whether it tolerates gibberish, so the premise has to be one a corpus of
    this shape could plausibly have carried.
    """

    @property
    def key(self) -> str:
        return "mutate_to_fake"

    def apply(self, entity: str) -> str:
        return f"{entity}{_NEAR_MISS_SUFFIX}"


class FlipDocumentedValue(Transform):
    """The documented figure, replaced by one the base does not carry.

    Every figure in the entity is shifted, which is language-neutral. An entity carrying no
    figure comes back unchanged and the engine then labels the hook documented, since the
    transformation decides the text and never the label.
    """

    @property
    def key(self) -> str:
        return "flip_value"

    def apply(self, entity: str) -> str:
        return _FIGURE.sub(_next_figure, entity)


class FlipDocumentedFact(Transform):
    """The documented fact, asserted the other way round."""

    @property
    def key(self) -> str:
        return "flip_fact"

    def apply(self, entity: str) -> str:
        return f"{_NEGATION}{entity}"


class KeepDocumentedEntity(Transform):
    """The entity exactly as the base carries it, which is what a control strategy asks for."""

    @property
    def key(self) -> str:
        return "keep_real"

    def apply(self, entity: str) -> str:
        return entity


_CLOSED_SET: tuple[Transform, ...] = (
    InventNearMissEntity(),
    FlipDocumentedValue(),
    FlipDocumentedFact(),
    KeepDocumentedEntity(),
)

TRANSFORMS: Mapping[str, Transform] = MappingProxyType({transform.key: transform for transform in _CLOSED_SET})
"""The closed registry, keyed by the string a ``StrategySpec`` names. Read-only by construction."""


def resolve(key: str) -> Transform:
    """The transformation a catalogue string names.

    Args:
        key: The ``StrategySpec.transform`` value, already accepted by catalogue validation.

    Returns:
        The registered implementation.

    Raises:
        ValueError: The key is outside the closed set, which means generation ran against a
            catalogue nobody validated.
    """
    transform = TRANSFORMS.get(key)
    if transform is None:
        message = f"unknown transform {key!r}: the catalogue must be validated before generation"
        raise ValueError(message)
    return transform
