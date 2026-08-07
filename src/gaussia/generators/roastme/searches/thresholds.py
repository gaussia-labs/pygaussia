"""``kappa`` and ``delta``, resolved once from the component that owns their scale (FR-041).

The two thresholds gaussia compares against a number a *substitutable* component produced. Their
meaning travels with that implementation, not with the config: a filter scoring ``[0, 1]`` and one
scoring ``[0, 100]`` are both valid, and a ``kappa`` of ``0.6`` gates sensibly against the first
while admitting every query against the second. The failure is silent in the worst way — the run
completes and the report looks populated — and validating a declared *range* would not catch it,
because ``0.6`` is inside both ranges.

So the component recommends the threshold instead, and this module applies one order:

    a supplied value, else the configured component's recommendation, else refuse to construct.

Refusing is the point of the mechanism (`data-model.md`, third row): a recommendation is
meaningless outside the scale it came from, so nothing may inherit one across a substitution, and
a component that recommends nothing obliges the user to say what the number should be.

Resolution happens once, when the Exploiter is constructed, and the resolved pair is what every
downstream comparison sees. ``Threshold`` carries where its value came from so the report can say
(FR-039), on the convention of ``Probe.engine`` and ``PrincipleGrade.model``: recorded for
reading, never branched on.

Pure functions over values and one small type. It imports no model and no third-party package, so
it stays importable with the ``roastme`` extra uninstalled (FR-037).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gaussia.core.on_profile_filter import OnProfileFilter
    from gaussia.core.realism_estimator import RealismEstimator
    from gaussia.schemas.roastme import ExploiterConfig

KAPPA = "kappa"
DELTA = "delta"

SUPPLIED = "supplied"
RECOMMENDED = "recommended by {component}"


class RecommendsThreshold(Protocol):
    """What resolution needs of a component: the threshold it recommends on its own scale, or none.

    Structural rather than a base class, because the on-profile filter and the realism estimator
    are unrelated abstractions that happen to owe the same declaration. Depending on this instead
    of on both of them keeps the rule written once.
    """

    recommended_threshold: float | None


class Threshold(float):
    """The value in force, carrying where it came from.

    A ``float``, because every comparison downstream reads it as one and because keeping the
    origin in a second variable would let the two drift apart. The origin exists for the failure
    report alone (FR-041); nothing branches on it.
    """

    __slots__ = ("origin",)

    origin: str

    def __new__(cls, value: float, origin: str) -> Threshold:
        threshold = super().__new__(cls, value)
        threshold.origin = origin
        return threshold


def resolve_kappa(config: ExploiterConfig, on_profile_filter: OnProfileFilter) -> Threshold:
    """The ``kappa`` in force, on the configured filter's own scale.

    Args:
        config: The method parameters, whose ``kappa`` is the user's value or ``None``.
        on_profile_filter: The filter whose scores the gate compares against, and the only party
            that can say what a meaningful ``kappa`` is on that scale.

    Returns:
        The resolved value, recording whether it was supplied or recommended.

    Raises:
        ValueError: The filter recommends no ``kappa`` and none was supplied.
    """
    return _resolve(config.kappa, on_profile_filter, KAPPA)


def resolve_delta(config: ExploiterConfig, realism_estimator: RealismEstimator) -> Threshold:
    """The ``delta`` in force, on the configured estimator's own scale.

    Args:
        config: The method parameters, whose ``delta`` is the user's value or ``None``.
        realism_estimator: The estimator whose realism gaps the budget bounds.

    Returns:
        The resolved value, recording whether it was supplied or recommended.

    Raises:
        ValueError: The estimator recommends no ``delta`` and none was supplied.
    """
    return _resolve(config.delta, realism_estimator, DELTA)


def _resolve(supplied: float | None, component: RecommendsThreshold, parameter: str) -> Threshold:
    if supplied is not None:
        return Threshold(supplied, SUPPLIED)
    recommended = component.recommended_threshold
    if recommended is not None:
        return Threshold(recommended, RECOMMENDED.format(component=type(component).__name__))
    message = (
        f"{type(component).__name__} recommends no {parameter} and none was supplied. "
        f"A {parameter} calibrated for another component's scale would gate nothing and say nothing, "
        f"so supply {parameter} explicitly or configure a component that recommends one."
    )
    raise ValueError(message)
