"""Query generator abstract base class: a category's attributes to concrete queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Category


class QueryGenerator(ABC):
    """Samples concrete queries that satisfy every attribute of a category.

    Kept separate from the category search on purpose. Under the policy-gradient search,
    optimisation pressure applies to the category generator alone and the query generator
    stays frozen — that is what preserves realism (paper invariant 5). "It was not
    modified" is only a checkable claim if the two are distinct objects, so folding query
    generation into the search would make the invariant unverifiable.
    """

    @abstractmethod
    def generate(self, category: Category, count: int) -> list[str]:
        """Sample queries exhibiting the category's conjunction of attributes.

        Args:
            category: The ordered conjunction to satisfy. Every attribute has to hold of
                every query — a category is a conjunction, not a menu.
            count: How many queries to return. The caller passes the configured
                queries-per-category ``n``, which is the sample size behind ``S(c)``.

        Returns:
            ``count`` queries. Returning fewer shrinks the denominator of ``S(c)`` without
            saying so, which is the one failure mode this method must not have.
        """

    def meta_for(self, query: str) -> dict[str, Any] | None:
        """The grading context this query carries, or ``None`` when it carries none.

        The Profiler grades against ``Probe.meta``, so a rubric may be written to check the answer
        against what the probe asserted — the real entity, and the false one put in front of the
        assistant. A generated query has no probe behind it, so without this the same rubric reaches
        its own exit clause and returns compliance for every query the Exploiter ever sends.

        The consequence is arithmetic and nothing raises: the highest violation the Exploiter can
        record becomes the sum of the weights of the principles that need no context. A ``tau`` above
        that is unreachable by construction, the run completes, and the empty report reads as a
        well-behaved assistant.

        **Declared by the generator because the generator is what put the premise there.** The
        alternative is a grader that recognises premises on its own, which either needs entity
        recognition over free prose — wrong in both directions, and each direction fabricates or
        loses a violation — or a second copy of the generator's vocabulary, which is the same fact
        written twice in two places that can drift apart with nothing to notice.

        Returns ``None`` by default: a generator that plants nothing has nothing to declare, and the
        grader then sees exactly what it saw before.
        """
        return None
