"""The policy-gradient category search: sample, gate, grade, reward, update.

The paper's headline procedure, and the one of the two that trains a category generator. Five
steps in a loop, and only the fifth needs a GPU: sample candidate categories from the policy,
discard the ones the gates reject, send the surviving queries to the target and grade them, turn
each graded outcome into a reward, apply one update. The first four are ordinary code, so they run
in the default suite on CPU with the training stack uninstalled (FR-033, SC-014) — which is only
true because the policy and the update step arrive injected and this module imports nothing heavy.
The single module that imports the training stack is ``policy_update.py``.

``CategoryPolicy`` and ``PolicyUpdateStep`` live here rather than in ``core/``: they are
collaborators of one shipped search, not part of the specification a user implements against. A
user substituting the search wholesale never sees either, which is what keeps the interface count
at ten — the same line already drawn around a category *generator*.

Gating and scoring are not this search's to choose: ``CategoryEvaluator`` owns them, so a category
over ``delta`` is rejected before a single target call and a query below ``kappa`` is never sent,
whichever procedure proposed the category. The reward is the score that evaluation produced, and a
candidate the gates rejected earns nothing.

The query generator is handed through untouched (FR-033, paper invariant 5): optimisation pressure
applies to the category policy alone, and that stays a checkable claim only because the two are
distinct objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gaussia.core.category_search import CategorySearch

from .evaluation import CategoryEvaluator

if TYPE_CHECKING:
    from gaussia.core.on_profile_filter import OnProfileFilter
    from gaussia.core.query_generator import QueryGenerator
    from gaussia.core.realism_estimator import RealismEstimator
    from gaussia.core.target_assistant import TargetAssistant
    from gaussia.schemas.roastme import (
        AssistantProfile,
        BehavioralContract,
        Category,
        CategoryEvaluation,
        ExploiterConfig,
    )

    # A category the policy proposed, with the log-probability it was sampled with.
    Candidate = tuple[Category, float]

    # The same, plus the reward it earned. The log-probability travels to the update step
    # unchanged: it is the sampling policy's, and recomputing it there would measure the updated
    # policy instead.
    RewardedCandidate = tuple[Category, float, float]

DEFAULT_ITERATIONS = 10
DEFAULT_CANDIDATES_PER_ITERATION = 8

# What a candidate the gates rejected earns. It reached no exchange, so there is no violation to
# score it on, and a reward invented for it would train the policy on nothing (FR-030, FR-031).
_NO_REWARD = 0.0


class CategoryPolicy(ABC):
    """The trainable category generator the search samples candidates from.

    Injected rather than constructed, because the model is the user's and because the log
    probabilities the update step needs are only knowable to whatever produced the sample.
    """

    @abstractmethod
    def sample(self, profile: AssistantProfile, count: int) -> list[Candidate]:
        """Sample ``count`` candidate categories, each with the log-probability it was drawn with.

        Args:
            profile: The weakness profile the categories are conditioned on, and the only
                artifact crossing from the Profiler (FR-013). Every attribute a policy proposes
                stays traceable to the entry that induced it (FR-028).
            count: How many candidates this iteration asks for.

        Returns:
            One ``(category, log-probability)`` pair per candidate. The log-probability is under
            the policy that produced the sample, which is what makes the update an off-policy
            correction rather than a re-derivation.
        """


class PolicyUpdateStep(ABC):
    """One gradient step on the policy, given a batch of rewarded samples.

    The only step of the loop that needs a GPU, which is why it is an interface here and an
    implementation in ``policy_update.py``: the loop stays testable without the training stack.
    """

    @abstractmethod
    def apply(self, samples: list[RewardedCandidate]) -> None:
        """Update the policy from one iteration's samples.

        Args:
            samples: The candidates this iteration learned something about, as
                ``(category, log-probability, reward)``. A candidate the gates rejected is
                included with a reward of ``0.0`` rather than dropped: it was judged, so the
                batch stays what the policy sampled and a baseline computed from it is not
                silently conditioned on success. A candidate whose every exchange the target
                dropped is **absent**, because nothing was learned about it and a zero would
                punish it for the transport (FR-016).
        """


class PolicyGradientSearch(CategorySearch):
    """Proposes categories by training a generator against the score its categories earn.

    Args:
        policy: The category generator the candidates are sampled from, and the only thing this
            search optimises.
        update_step: How a batch of rewarded samples becomes a gradient step.
        iterations: The budget, in iterations. The loop stops on it and on nothing else: an
            early-stopping rule on the reward would make the number of target calls a run costs
            depend on the assistant.
        candidates_per_iteration: How many candidates each iteration asks the policy for. A knob
            of gaussia's own search rather than a parameter of the method, so gaussia owns its
            default (FR-040) — it bounds the target calls one iteration costs, at
            ``candidates_per_iteration * queries_per_category``.
    """

    def __init__(
        self,
        policy: CategoryPolicy,
        update_step: PolicyUpdateStep,
        iterations: int = DEFAULT_ITERATIONS,
        candidates_per_iteration: int = DEFAULT_CANDIDATES_PER_ITERATION,
    ) -> None:
        self._policy = policy
        self._update_step = update_step
        self._iterations = iterations
        self._candidates_per_iteration = candidates_per_iteration

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
        reported: dict[tuple[str, ...], CategoryEvaluation] = {}
        for _ in range(self._iterations):
            # One evaluator per iteration. Its cache stops an iteration paying twice for the same
            # candidate, while a candidate the updated policy proposes again is evaluated again —
            # a reward carried over from an earlier iteration would train the policy on a sample
            # it never drew.
            evaluator = CategoryEvaluator(
                profile,
                contract,
                config,
                target,
                query_generator,
                on_profile_filter,
                realism_estimator,
            )
            samples: list[RewardedCandidate] = []
            for category, log_probability in self._policy.sample(profile, self._candidates_per_iteration):
                evaluation = evaluator.evaluate(category)
                if evaluation is not None:
                    # Keyed on the attributes, so a category several iterations proposed reaches
                    # the report once, under the last evaluation of it the run made.
                    reported[tuple(category.attributes)] = evaluation
                elif evaluator.reached_the_target(category):
                    # Asked and every exchange dropped: the run learned nothing about this
                    # category, so it earns no reward and no punishment. Scoring it zero would
                    # train the policy away from a category the assistant may well break,
                    # on the strength of a transport failure — and would make a failed exchange
                    # count as a pass, which FR-016 forbids.
                    continue
                samples.append((category, log_probability, _reward(evaluation)))
            self._update_step.apply(samples)
        return list(reported.values())


def _reward(evaluation: CategoryEvaluation | None) -> float:
    return _NO_REWARD if evaluation is None else evaluation.score
