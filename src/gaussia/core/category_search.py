"""Category search abstract base class: a profile to scored failure categories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.core.on_profile_filter import OnProfileFilter
    from gaussia.core.query_generator import QueryGenerator
    from gaussia.core.realism_estimator import RealismEstimator
    from gaussia.core.target_assistant import TargetAssistant
    from gaussia.schemas.roastme import (
        AssistantProfile,
        BehavioralContract,
        CategoryEvaluation,
        ExploiterConfig,
    )


class CategorySearch(ABC):
    """Proposes categories from a profile and scores them.

    One interface over both procedures the paper gives, because what differs between them
    is only how categories are *proposed*: one trains a generator under policy gradient,
    the other intersects the attributes of a pool of high-scoring pairs. A category
    *generator* is deliberately not a separate interface — the training-free procedure has
    none, so requiring one would force a fake implementation.

    The decision rule is not the implementation's to choose. ``S(c)``, the ``kappa`` gate,
    the ``delta`` budget and refinement live in one scoring module both searches call, so
    switching procedure cannot change what counts as a failing category.
    """

    @abstractmethod
    def search(
        self,
        profile: AssistantProfile,
        contract: BehavioralContract,
        config: ExploiterConfig,
        target: TargetAssistant,
        query_generator: QueryGenerator,
        on_profile_filter: OnProfileFilter,
        realism_estimator: RealismEstimator,
    ) -> list[CategoryEvaluation]:
        """Search for the categories of realistic interaction that break the assistant.

        Args:
            profile: The weakness profile, and the only artifact crossing from the Profiler
                (FR-013). Categories are grounded in weaknesses at or above ``eta`` and in
                the retained hooks, and every attribute stays traceable to the entry that
                induced it.
            contract: The principles and their weights, so a response can be graded into a
                violation score.
            config: The method parameters. ``kappa`` and ``delta`` arrive already resolved
                against the configured filter and estimator (FR-041) — an implementation
                reads them and never resolves them itself, so one run has one pair of
                thresholds in force.
            target: The assistant under evaluation. The only path to it (FR-017).
            query_generator: Samples the queries a category is evaluated with. Under a
                search that optimises anything, this collaborator stays unmodified (paper
                invariant 5).
            on_profile_filter: Scores each query for the ``kappa`` gate.
            realism_estimator: Scores each category's realism gap for the ``delta`` budget.

        Returns:
            One evaluation per surviving category, auditable down to the per-query
            per-principle grades (FR-036). Ranking, and surfacing the individual queries
            that reached ``tau``, belong to the caller.
        """
