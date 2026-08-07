"""On-profile filter abstract base class: the score the kappa gate compares against."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import AssistantProfile


class OnProfileFilter(ABC):
    """Scores how on-profile and indirect a single query is.

    A semantic judgement about one query, not an aggregate over values, which is why it is
    an interface rather than a function in the scoring module: that module stays pure
    arithmetic and takes this score as a number.

    The scale is the implementation's own. One filter may score in ``[0, 1]`` and another
    in ``[0, 100]``, and both are valid — so a ``kappa`` calibrated for the first admits
    every query against the second, silently, with the run completing and the report
    looking populated. An implementation therefore declares the ``kappa`` it recommends on
    its own scale (FR-041).
    """

    recommended_threshold: float | None = None
    """The ``kappa`` this implementation recommends on its own scale, or ``None``.

    ``None`` is a legitimate declaration and obliges the user to supply ``kappa``
    explicitly: a filter that recommends nothing and is used with nothing supplied fails at
    construction, naming the component and the parameter. No configured combination may
    fall back to a value calibrated for a different component's scale. Resolution happens
    once, when the Exploiter is constructed, and a user-supplied value always wins.
    """

    @abstractmethod
    def score(self, query: str, profile: AssistantProfile) -> float:
        """Score how on-profile and indirect ``query`` is.

        Args:
            query: The single query to judge.
            profile: The weakness profile the query is supposed to be on. Its weaknesses
                and retained hooks are prose, not internal identifiers (paper invariant 3).

        Returns:
            The score, on this implementation's own scale. A query below the resolved
            ``kappa`` contributes exactly ``0.0`` to its category's score (FR-030): the gate
            is what keeps a blatant ask from counting as a realistic failure.
        """
