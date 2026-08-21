"""Grader abstract base class for the Roast Me behavioral contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Principle, PrincipleGrade


class Grader(ABC):
    """Estimates ``pi_hat_j(x, r)``: whether one principle was violated by one response.

    Exactly one grader is bound per principle (spec D2): comparing graders is done by
    running the whole evaluation once per grader, never by aggregating several inside
    one principle.

    Two obligations no signature can express:

    - A grader never reaches the knowledge base (paper invariant 2). Everything it may
      check the response against arrives through the arguments below.
    - The grade it returns carries its own provenance — the verdict method, the model
      identity and the evidence — so that substituting a grader leaves every downstream
      component untouched (FR-005).
    """

    @abstractmethod
    def grade(
        self,
        query: str,
        response: str,
        principle: Principle,
        meta: dict[str, Any] | None = None,
    ) -> PrincipleGrade:
        """Estimate whether ``principle`` was violated by ``response``.

        Args:
            query: The question the assistant answered.
            response: The assistant's answer.
            principle: The principle being estimated, carrying the rubric. The rubric is
                user-supplied and must be passed to the judge unmodified — gaussia never
                substitutes or appends to it (FR-006).
            meta: The originating probe's ``meta`` — the real value, the value the probe
                asserted, the real and false chains — or ``None`` when the exchange came
                from a query the category search generated rather than from a probe.

        Returns:
            The grade, with ``score`` in ``[0, 1]``, the verdict ``method`` that produced
            it, the model identity and the evidence behind it.
        """
