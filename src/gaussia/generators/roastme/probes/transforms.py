"""The four transformations gaussia ships, and the registry that resolves what a catalogue names.

``transform`` is the one catalogue field whose value changes what a probe *means*, so an
unrecognised string must never resolve to anything: a probe whose premise nobody can account for
carries a ``doc`` label nobody can trust. What FR-025 buys is that guarantee, and it is kept by
resolving against a registry rather than by the set being fixed — so the set is **the four shipped
plus whatever the caller supplies**, and a string outside that is still refused.

Supplying one matters because the four assume the shape of entity the paper's own corpus carries.
``mutate_to_fake`` suffixes ``-2``, which reads as a near miss of ``POLICY-1`` and as a typo of
``Cuenta Digital Libre``. ``flip_value`` increments every digit run, so a figure written with
thousands separators comes back mangled and an entity with no digits comes back unchanged — which
the engine then labels documented, quietly turning that strategy into a second control. And
``flip_fact`` prepends an English ``not``. None of the three is wrong for the corpus they were
written against; all three are wrong for a corpus of Spanish product names, and the answer is to
pass one that fits rather than to argue with these.

This registry is the single point where the catalogue's string becomes behaviour. Validation
checks membership, an engine resolves the key once per strategy, and nothing anywhere branches
on the string; it survives only on ``KnowledgeHook.how``, as a record rather than a dispatch.

One limit stated rather than hidden: a transformation decides the *text* of a premise and never
its label — whether the result is documented is the engine's call, derived from its own knowledge
of the base's boundary (FR-021), so a transformation that leaves an entity untouched simply
yields a documented hook.

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
    from collections.abc import Mapping, Sequence

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
"""The four shipped, keyed by the string a ``StrategySpec`` names. Read-only by construction."""


def available(extra: Sequence[Transform] = ()) -> Mapping[str, Transform]:
    """The four shipped plus the user's, which is what a catalogue may name.

    The shipped four cannot be replaced. A user key colliding with one of them is refused rather than
    preferred, because either resolution silently changes what an existing catalogue means: prefer the
    user's and a catalogue written against the shipped behaviour starts producing different premises;
    prefer the shipped and the user's implementation is ignored without a word.

    Args:
        extra: The user's transformations. An empty sequence yields the shipped registry unchanged.

    Returns:
        The merged registry, keyed by ``Transform.key``.

    Raises:
        ValueError: Two of ``extra`` share a key, or one of them collides with a shipped key.
    """
    if not extra:
        return TRANSFORMS
    merged = dict(TRANSFORMS)
    for transform in extra:
        if transform.key in merged:
            shipped = transform.key in TRANSFORMS
            source = "one of the four shipped" if shipped else "another supplied transform"
            message = f"transform key {transform.key!r} collides with {source}"
            raise ValueError(message)
        merged[transform.key] = transform
    return MappingProxyType(merged)


def resolve(key: str, extra: Sequence[Transform] = ()) -> Transform:
    """The transformation a catalogue string names.

    Args:
        key: The ``StrategySpec.transform`` value, already accepted by catalogue validation.
        extra: The user's transformations, the same ones the catalogue was validated against. Passing
            a different set here than to validation is how a catalogue that validated can still fail
            at generation.

    Returns:
        The registered implementation.

    Raises:
        ValueError: The key is in neither the shipped four nor ``extra``, which means generation ran
            against a catalogue nobody validated — or validated against a different set.
    """
    transform = available(extra).get(key)
    if transform is None:
        message = f"unknown transform {key!r}: the catalogue must be validated before generation"
        raise ValueError(message)
    return transform
