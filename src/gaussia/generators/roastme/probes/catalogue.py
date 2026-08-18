"""Catalogue validation: every rejection up front, before a single probe is generated.

Five of the six conditions are decidable from the catalogue and the contract alone. The sixth
needs the configured engines, because ``entity_kind`` is the user's own vocabulary and gaussia
never learns what it means (FR-025): without that check a plural typo validates cleanly and
yields an empty probe set with no error, which is the failure mode hardest to notice.

A strategy naming no plugin is a control, not a dangling reference — the only mechanism by which
a control is recognised (FR-026). The catalogue itself is the user's (FR-027); gaussia validates
it and owns no file format for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .transforms import available

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.fact_twister import FactTwister
    from gaussia.core.probe_engine import ProbeEngine
    from gaussia.core.transform import Transform
    from gaussia.schemas.roastme import BehavioralContract, Catalogue


def validate_catalogue(
    catalogue: Catalogue,
    contract: BehavioralContract,
    engines: Sequence[ProbeEngine],
    transforms: Sequence[Transform] = (),
    twisters: Sequence[FactTwister] = (),
) -> None:
    """Accept a catalogue, or refuse it naming what is wrong.

    Args:
        catalogue: The user's plugins and strategies.
        contract: The principles the plugins must resolve against.
        engines: The engines that will run, which is what makes ``entity_kind`` decidable.
        transforms: The user's transformations, beyond the four shipped. **Pass the same sequence the
            engines were given**: a catalogue validated against one set and generated against another
            is exactly the case where validation stops meaning anything.
        twisters: The twisters that will run, whose declared patterns are also strings a
            ``StrategySpec.transform`` may name (FR-042). Same obligation as ``transforms``: the same
            sequence the grounded engine was given. FR-025's set of *transformations* stays closed —
            a pattern is not one, and joins nothing but the set of acceptable names, and only while a
            twister declaring it is configured.

    Raises:
        ValueError: Any of the six rejections of the specification. Every one is decided here,
            so a run either generates or fails before it starts.
    """
    _reject_duplicate_identifiers(catalogue)
    _reject_dangling_principles(catalogue, contract)
    _reject_dangling_plugins(catalogue)
    _reject_unknown_transforms(catalogue, transforms, twisters)
    _reject_unhandled_entity_kinds(catalogue, engines)


def _reject_duplicate_identifiers(catalogue: Catalogue) -> None:
    families = (
        ("plugin", [plugin.id for plugin in catalogue.plugins]),
        ("strategy", [strategy.id for strategy in catalogue.strategies]),
    )
    for label, identifiers in families:
        repeated = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
        if repeated:
            message = f"duplicate {label} identifiers: {repeated}"
            raise ValueError(message)


def _reject_dangling_principles(catalogue: Catalogue, contract: BehavioralContract) -> None:
    known = {principle.id for principle in contract.principles}
    dangling = sorted({plugin.principle for plugin in catalogue.plugins if plugin.principle not in known})
    if dangling:
        message = f"plugins name principles the contract does not carry: {dangling}"
        raise ValueError(message)


def _reject_dangling_plugins(catalogue: Catalogue) -> None:
    known = {plugin.id for plugin in catalogue.plugins}
    dangling = sorted(
        {
            strategy.plugin
            # A strategy with no plugin is a control, not a dangling reference (FR-026).
            for strategy in catalogue.strategies
            if strategy.plugin is not None and strategy.plugin not in known
        }
    )
    if dangling:
        message = f"strategies name plugins the catalogue does not carry: {dangling}"
        raise ValueError(message)


def _reject_unknown_transforms(
    catalogue: Catalogue,
    transforms: Sequence[Transform],
    twisters: Sequence[FactTwister] = (),
) -> None:
    # `available` also refuses a supplied key that collides with a shipped one, so a catalogue is
    # never accepted against an ambiguous registry.
    registry = available(transforms)
    # A twist pattern is a name a strategy may carry and not a transformation (FR-042): it resolves to
    # nothing in that registry, and the grounded engine hands it to its twister instead. Accepted here
    # only because a configured twister declares it, which is what keeps the rejection meaningful —
    # a catalogue naming a pattern no configured twister realises still fails before generation.
    patterns = {pattern for twister in twisters for pattern in twister.patterns}
    known = set(registry) | patterns
    unknown = sorted({strategy.transform for strategy in catalogue.strategies if strategy.transform not in known})
    if unknown:
        message = f"strategies name transforms outside {sorted(known)}: {unknown}"
        raise ValueError(message)


def _reject_unhandled_entity_kinds(catalogue: Catalogue, engines: Sequence[ProbeEngine]) -> None:
    handled = {kind for engine in engines for kind in engine.entity_kinds}
    unhandled = sorted(
        {strategy.entity_kind for strategy in catalogue.strategies if strategy.entity_kind not in handled}
    )
    if unhandled:
        message = f"no configured engine declares it can produce the entity kinds: {unhandled}"
        raise ValueError(message)
