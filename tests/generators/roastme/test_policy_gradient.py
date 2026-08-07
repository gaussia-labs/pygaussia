"""The policy-gradient search's loop, on CPU with no extra installed (T026).

The loop is five steps and only the fifth needs a GPU: sample candidates from the policy, drop
the ones failing `kappa` or `delta`, send the survivors to the target and grade them, turn the
graded outcomes into a reward, apply the update. Injecting the policy and the update step puts
the first four under hand-computed coverage; what stays uncovered is one call into a third-party
trainer, which lives in a different module for exactly that reason (FR-033, SC-014).

The two collaborator abstractions are imported from the search's own module rather than from
`core/`: they are collaborators of one shipped search, not part of the specification a user
implements against.
"""

import pytest

from gaussia.generators.roastme.searches.policy_gradient import (
    CategoryPolicy,
    PolicyGradientSearch,
    PolicyUpdateStep,
)
from gaussia.schemas.roastme import AssistantProfile, Category, ExploiterConfig, WeaknessEntry
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import (
    RecommendingOnProfileFilter,
    RecommendingRealismEstimator,
    RecordedTarget,
    StubGrader,
    StubQueryGenerator,
)

TOLERANCE = 1e-9

ITERATIONS = 3
KAPPA = 0.5
DELTA = 0.3
QUERIES_PER_CATEGORY = 2

SURVIVING_ATTRS = ("asks for an exception to a stated rule",)
UNREALISTIC_ATTRS = ("phrased as an obvious jailbreak",)

SURVIVING_QUERIES = ["pg-q1", "pg-q2"]
UNREALISTIC_QUERIES = ["pg-q3", "pg-q4"]

SURVIVING_LOGPROB = -0.5
UNREALISTIC_LOGPROB = -1.5

# pg-q2 scores below kappa, so it contributes exactly 0.0 and the category is scored over
# [1.0, 0.0] — the gated vector of the hand-computed fixture, whose S(c) is 0.5 - sqrt(0.5)/2.
ON_PROFILE_SCORES = {"pg-q1": 0.9, "pg-q2": 0.1, "pg-q3": 0.9, "pg-q4": 0.9}
REALISM_GAPS = {tuple(SURVIVING_QUERIES): 0.2, tuple(UNREALISTIC_QUERIES): 0.5}

EXPECTED_REWARD = fx.GATED_SCORE_LAMBDA_1


class StubPolicy(CategoryPolicy):
    """A policy returning fixed categories with fixed log-probabilities."""

    def __init__(self):
        self.calls: list[int] = []

    def sample(self, profile: AssistantProfile, count: int) -> list[tuple[Category, float]]:
        self.calls.append(count)
        return [
            (Category(attributes=list(SURVIVING_ATTRS), provenance=["weakness one"]), SURVIVING_LOGPROB),
            (Category(attributes=list(UNREALISTIC_ATTRS), provenance=["weakness one"]), UNREALISTIC_LOGPROB),
        ]


class StubPolicyUpdateStep(PolicyUpdateStep):
    """An update step that records what it was asked to apply and applies nothing."""

    def __init__(self):
        self.batches: list[list[tuple[Category, float, float]]] = []

    def apply(self, samples: list[tuple[Category, float, float]]) -> None:
        self.batches.append(list(samples))


def _grades() -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for query in [*SURVIVING_QUERIES, *UNREALISTIC_QUERIES]:
        for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C):
            scores[(query, principle)] = 1.0
    return scores


def _profile() -> AssistantProfile:
    return AssistantProfile(
        weaknesses=[
            WeaknessEntry(
                principle=fx.PRINCIPLE_A,
                descriptor=SURVIVING_ATTRS[0],
                rate=0.75,
                n=4,
                standard_error=fx.SE_RATE_075_N4,
            )
        ]
    )


def _run():
    policy = StubPolicy()
    update_step = StubPolicyUpdateStep()
    query_generator = StubQueryGenerator(
        {
            SURVIVING_ATTRS: SURVIVING_QUERIES,
            UNREALISTIC_ATTRS: UNREALISTIC_QUERIES,
        }
    )
    target = RecordedTarget(
        responses={query: f"response to {query}" for query in [*SURVIVING_QUERIES, *UNREALISTIC_QUERIES]}
    )
    search = PolicyGradientSearch(policy=policy, update_step=update_step, iterations=ITERATIONS)
    evaluations = search.search(
        _profile(),
        fx.contract(StubGrader(_grades())),
        ExploiterConfig(
            tau=0.5,
            eta=0.25,
            queries_per_category=QUERIES_PER_CATEGORY,
            kappa=KAPPA,
            delta=DELTA,
        ),
        target,
        query_generator,
        RecommendingOnProfileFilter(ON_PROFILE_SCORES),
        RecommendingRealismEstimator(REALISM_GAPS),
    )
    return evaluations, policy, update_step, query_generator, target


class TestSampling:
    def test_exactly_the_configured_number_of_queries_is_requested_per_candidate(self):
        """Returning fewer would shrink the denominator of `S(c)` without saying so."""
        _, _, _, query_generator, _ = _run()

        assert query_generator.calls != []
        assert all(count == QUERIES_PER_CATEGORY for _, count in query_generator.calls)

    def test_one_request_per_candidate_per_iteration(self):
        _, policy, _, query_generator, _ = _run()
        candidates_per_iteration = 2

        assert len(policy.calls) == ITERATIONS
        assert len(query_generator.calls) == ITERATIONS * candidates_per_iteration


class TestGating:
    def test_a_category_over_delta_never_reaches_the_target(self):
        """FR-031: the realism gap is computed without querying the assistant, so an unrealistic
        category costs nothing to reject."""
        _, _, _, _, target = _run()
        sent = {query for query, _ in target.sent}

        assert UNREALISTIC_QUERIES[0] not in sent
        assert UNREALISTIC_QUERIES[1] not in sent

    def test_a_query_below_kappa_is_dropped_and_its_category_still_evaluated(self):
        evaluations, _, _, _, target = _run()
        sent = {query for query, _ in target.sent}

        assert sent == {SURVIVING_QUERIES[0]}
        assert evaluations != []

    def test_the_dropped_query_contributes_exactly_zero(self):
        """FR-030, and the zero has to stay explainable through `on_profile`."""
        evaluations, _, _, _, _ = _run()
        surviving = next(item for item in evaluations if tuple(item.category.attributes) == SURVIVING_ATTRS)

        assert surviving.violations == fx.GATED_VIOLATIONS
        assert surviving.on_profile == [True, False]

    def test_a_category_over_delta_is_not_reported(self):
        evaluations, _, _, _, _ = _run()
        reported = {tuple(item.category.attributes) for item in evaluations}

        assert UNREALISTIC_ATTRS not in reported


class TestReward:
    def test_the_reward_is_the_category_score(self):
        _, _, update_step, _, _ = _run()
        rewards = {tuple(category.attributes): reward for category, _, reward in update_step.batches[0]}

        assert rewards[SURVIVING_ATTRS] == pytest.approx(EXPECTED_REWARD, abs=TOLERANCE)

    def test_a_discarded_candidate_earns_nothing(self):
        _, _, update_step, _, _ = _run()
        rewards = {tuple(category.attributes): reward for category, _, reward in update_step.batches[0]}

        assert rewards.get(UNREALISTIC_ATTRS, 0.0) == pytest.approx(0.0, abs=TOLERANCE)

    def test_the_log_probability_the_policy_reported_is_passed_through(self):
        """The update step needs it; the search must not recompute or discard it."""
        _, _, update_step, _, _ = _run()
        logprobs = {tuple(category.attributes): logprob for category, logprob, _ in update_step.batches[0]}

        assert logprobs[SURVIVING_ATTRS] == pytest.approx(SURVIVING_LOGPROB, abs=TOLERANCE)


class TestStopping:
    def test_the_loop_stops_on_its_budget(self):
        _, policy, update_step, _, _ = _run()

        assert len(update_step.batches) == ITERATIONS
        assert len(policy.calls) == ITERATIONS


class TestTheQueryGeneratorStaysFrozen:
    def test_it_is_never_mutated_across_iterations(self):
        """Paper invariant 5: optimisation pressure applies to the category generator alone, and
        that is only a checkable claim because the two are distinct objects."""
        _, _, _, query_generator, _ = _run()

        assert query_generator.queries == {
            SURVIVING_ATTRS: SURVIVING_QUERIES,
            UNREALISTIC_ATTRS: UNREALISTIC_QUERIES,
        }
        assert set(vars(query_generator)) == {"queries", "calls"}

    def test_only_the_policy_receives_an_update(self):
        _, _, update_step, query_generator, _ = _run()

        assert update_step.batches != []
        assert not hasattr(query_generator, "apply")
