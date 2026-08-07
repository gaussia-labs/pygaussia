"""Query generator abstract base class: a category's attributes to concrete queries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
