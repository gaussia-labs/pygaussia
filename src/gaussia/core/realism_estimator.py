"""Realism estimator abstract base class: the divergence the delta budget bounds."""

from abc import ABC, abstractmethod


class RealismEstimator(ABC):
    """Scores how far a category's sampled queries sit from the natural-query prior.

    This is the quantity ``delta`` bounds. It must be computed without querying the
    assistant (FR-031): the realism of a category is a property of the queries and the
    prior, so spending target calls on it would make the budget cost what it is meant to
    protect.

    Substitutable, since the search depends on it only through ``delta``. The scale is the
    implementation's own, so each declares the ``delta`` it recommends (FR-041) on the same
    terms as the on-profile filter.
    """

    recommended_threshold: float | None = None
    """The ``delta`` this implementation recommends on its own scale, or ``None``.

    ``None`` is a legitimate declaration and obliges the user to supply ``delta``
    explicitly: an estimator that recommends nothing and is used with nothing supplied
    fails at construction, naming the component and the parameter. Resolution happens once,
    when the Exploiter is constructed, and a user-supplied value always wins.

    **Read it on the instance.** For the shipped estimator it cannot be otherwise: its scale is a
    mean cosine distance, and what counts as far depends on the prior pool, so the recommendation
    does not exist until the pool has been encoded. The value here is what an implementation
    recommending nothing inherits — and ``None`` is not "unknown", it is the declaration that the
    user must supply ``delta``, which is why reading it off a class would answer the opposite of
    what the component actually recommends.
    """

    @abstractmethod
    def estimate(self, queries: list[str]) -> float:
        """Estimate the divergence of a category's queries from the natural-query prior.

        Args:
            queries: The queries sampled for one category.

        Returns:
            The realism gap, on this implementation's own scale. A category whose gap
            exceeds the resolved ``delta`` is discarded regardless of its score (FR-031),
            so a category that breaks the assistant only with unnatural traffic never
            reaches the report.
        """
