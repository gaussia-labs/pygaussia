"""Evaluating one category: sample, gate, ask, grade, score.

What every search does with a category once it has proposed one, written once so that switching
search procedure cannot change what counts as a failing category. The procedures differ only in
how categories are *proposed*; the decision rule is not theirs to choose.

The order of the steps is the requirement, not an optimisation:

* the realism gap is estimated first and a category over ``delta`` is discarded before a single
  query is sent (FR-031). Realism is a property of the queries and the prior, so a category that
  breaks the assistant only with unnatural traffic costs nothing to reject;
* a query below ``kappa`` contributes exactly ``0.0`` and is not sent either (FR-030). The gate
  decides whether the query can count at all, and asking a question whose answer cannot count
  spends a target call on nothing. The zero stays explainable through ``on_profile``;
* an exchange the target reports as failed enters no list at all. It is ungraded, so it may move
  neither the numerator nor the denominator of ``S(c)`` (FR-016) — and ``n`` travelling with the
  score is what keeps the shrunken sample visible (FR-029).

Evaluations are cached by the category's attributes, because refinement asks for the same
sub-conjunctions a search has often already evaluated, and every repeat would be paid for in
target calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from gaussia.schemas.roastme import CategoryEvaluation

from .scoring import GATED_CONTRIBUTION, category_score, is_on_profile, violation_score, within_realism_budget
from .thresholds import DELTA, KAPPA

if TYPE_CHECKING:
    from gaussia.core.on_profile_filter import OnProfileFilter
    from gaussia.core.query_generator import QueryGenerator
    from gaussia.core.realism_estimator import RealismEstimator
    from gaussia.core.target_assistant import TargetAssistant
    from gaussia.schemas.roastme import (
        AssistantProfile,
        BehavioralContract,
        Category,
        ExploiterConfig,
        PrincipleGrade,
    )

# What a category that never reached the target scores and how far it reads from the prior: a
# verdict that fails `tau` and `delta` at once, whichever way the thresholds are set.
_FAILS_EVERY_THRESHOLD = (0.0, float("inf"))

_GATED_RESPONSE = ""


class _Exchange(NamedTuple):
    """One query's contribution to its category, already gated."""

    query: str
    response: str
    violation: float
    on_profile: bool
    grades: list[PrincipleGrade]


class CategoryEvaluator:
    """Turns a proposed category into the evaluation the report is built from.

    ``verdict`` is the ``CategoryEvaluator`` callable ``scoring.refine`` takes, so refinement
    never owns evaluation and evaluation never owns refinement.

    Args:
        profile: The weakness profile the on-profile filter judges each query against.
        contract: The principles and their weights, so a response becomes a violation score.
        config: The method parameters, with ``kappa`` and ``delta`` already resolved by the
            Exploiter (FR-041). They are read, never resolved here, so one run has one pair of
            thresholds in force.
        target: The assistant under evaluation. The only path to it (FR-017).
        query_generator: Samples the queries a category is evaluated with. Never modified.
        on_profile_filter: Scores each query for the ``kappa`` gate.
        realism_estimator: Scores the category's realism gap for the ``delta`` budget.
    """

    def __init__(
        self,
        profile: AssistantProfile,
        contract: BehavioralContract,
        config: ExploiterConfig,
        target: TargetAssistant,
        query_generator: QueryGenerator,
        on_profile_filter: OnProfileFilter,
        realism_estimator: RealismEstimator,
    ) -> None:
        self._profile = profile
        self._contract = contract
        self._config = config
        self._target = target
        self._query_generator = query_generator
        self._on_profile_filter = on_profile_filter
        self._realism_estimator = realism_estimator
        self._kappa = _resolved(config.kappa, KAPPA)
        self._delta = _resolved(config.delta, DELTA)
        self._evaluated: dict[tuple[str, ...], CategoryEvaluation | None] = {}
        self._sent: set[tuple[str, ...]] = set()

    @property
    def delta(self) -> float:
        """The realism budget in force, for a caller that has to refine against the same one."""
        return self._delta

    def reached_the_target(self, category: Category) -> bool:
        """Whether any query of this category was actually sent.

        Both of ``evaluate``'s empty results look alike and are not: a category over ``delta`` was
        never asked, which is evidence about the category, while one whose every exchange the
        target dropped is evidence about the transport. A caller that learns from the result has
        to tell them apart, because FR-016 forbids a failed exchange counting as a pass.
        """
        return tuple(category.attributes) in self._sent

    def evaluate(self, category: Category) -> CategoryEvaluation | None:
        """The category's evaluation, or ``None`` when it produced no scoreable exchange.

        ``None`` covers the two cases with nothing to report: a category over ``delta``, which
        was never sent, and one whose every exchange the target reported as failed.
        """
        key = tuple(category.attributes)
        if key not in self._evaluated:
            self._evaluated[key] = self._evaluate(category)
        return self._evaluated[key]

    def verdict(self, category: Category) -> tuple[float, float]:
        """``(S(c), realism gap)``, the pair both thresholds read."""
        evaluation = self.evaluate(category)
        if evaluation is None:
            return _FAILS_EVERY_THRESHOLD
        return evaluation.score, evaluation.realism_gap

    def _evaluate(self, category: Category) -> CategoryEvaluation | None:
        queries = self._query_generator.generate(category, self._config.queries_per_category)
        realism_gap = self._realism_estimator.estimate(queries)
        if not within_realism_budget(realism_gap, self._delta):
            return None
        self._sent.add(tuple(category.attributes))
        exchanges = [exchange for query in queries if (exchange := self._exchange(query)) is not None]
        if not exchanges:
            return None
        violations = [exchange.violation for exchange in exchanges]
        return CategoryEvaluation(
            category=category,
            queries=[exchange.query for exchange in exchanges],
            responses=[exchange.response for exchange in exchanges],
            violations=violations,
            on_profile=[exchange.on_profile for exchange in exchanges],
            realism_gap=realism_gap,
            score=category_score(violations, self._config.lambda_),
            n=len(exchanges),
            rationale=[exchange.grades for exchange in exchanges],
            dropped_attributes=[],
        )

    def _exchange(self, query: str) -> _Exchange | None:
        if not is_on_profile(self._on_profile_filter.score(query, self._profile), self._kappa):
            return _Exchange(query, _GATED_RESPONSE, GATED_CONTRIBUTION, on_profile=False, grades=[])
        response = self._target.send(query)
        if response.failed:
            return None
        grades = [principle.grader.grade(query, response.content, principle) for principle in self._contract.principles]
        return _Exchange(
            query,
            response.content,
            violation_score(grades, self._contract),
            on_profile=True,
            grades=grades,
        )


def _resolved(value: float | None, parameter: str) -> float:
    if value is None:
        message = (
            f"{parameter} reaches a search already resolved against the configured component, and this one "
            f"carries none; construct the search through the Exploiter, which resolves it once"
        )
        raise ValueError(message)
    return value
