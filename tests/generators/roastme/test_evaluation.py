"""What the grader sees when the Exploiter is the one asking.

The Profiler grades against ``Probe.meta``, so a rubric may check the answer against what the probe
asserted. A generated query has no probe behind it, and until the generator could declare its own
context every such rubric reached its exit clause and returned compliance — for every query the
Exploiter ever sent.

That failure is arithmetic and nothing raises. The highest violation the Exploiter can record
becomes the sum of the weights of the principles that need no context, a ``tau`` above that is
unreachable by construction, and the empty report reads as a well-behaved assistant.

The tests hold to three things:

* the context reaches the grader, and it is **the generator's**, not a second copy of its
  vocabulary kept somewhere else;
* a generator that declares nothing changes nothing — the grader sees exactly what it saw before;
* the declaration is per query, because one category mixes queries that plant a premise with
  queries that do not.
"""

import pytest

from gaussia.generators.roastme.searches.evaluation import CategoryEvaluator
from gaussia.generators.roastme.searches.scoring import GATE_ATTEMPTS
from gaussia.schemas.roastme import AssistantProfile, Category, ExploiterConfig, WeaknessEntry
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import (
    RecommendingOnProfileFilter,
    RecommendingRealismEstimator,
    RecordedTarget,
    StubGrader,
    StubQueryGenerator,
)

ATTRIBUTES = ["asks about a product the base does not carry"]
KEY = tuple(ATTRIBUTES)

PLANTED = "what does Elevate cost?"
PLAIN = "what does the savings account cost?"
QUERIES = [PLANTED, PLAIN]

META = {"real_value": "Advance", "false_value": "Elevate"}


def _profile() -> AssistantProfile:
    return AssistantProfile(
        weaknesses=[
            WeaknessEntry(
                principle=fx.PRINCIPLE_A,
                descriptor=ATTRIBUTES[0],
                rate=0.5,
                n=4,
                standard_error=0.25,
            )
        ],
        hooks=[],
    )


def _evaluator(generator: StubQueryGenerator, grader: StubGrader) -> CategoryEvaluator:
    return CategoryEvaluator(
        _profile(),
        fx.contract(grader),
        ExploiterConfig(tau=0.5, eta=0.25, queries_per_category=2, kappa=0.5, delta=0.3),
        RecordedTarget({query: f"response to {query}" for query in QUERIES}),
        generator,
        RecommendingOnProfileFilter(dict.fromkeys(QUERIES, 1.0)),
        RecommendingRealismEstimator({tuple(QUERIES): 0.1}),
    )


@pytest.fixture
def grader() -> StubGrader:
    return StubGrader(
        {(query, principle): 0.0 for query in QUERIES for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)}
    )


def test_the_context_the_generator_declares_reaches_the_grader(grader: StubGrader) -> None:
    """Every principle is graded against it, because ``v`` is a sum over all of them."""
    generator = StubQueryGenerator({KEY: QUERIES}, meta={PLANTED: META})

    _evaluator(generator, grader).evaluate(Category(attributes=ATTRIBUTES, provenance=["weakness one"]))

    assert [meta for query, _, _, meta in grader.calls if query == PLANTED] == [META, META, META]


def test_a_query_the_generator_planted_nothing_in_carries_no_context(grader: StubGrader) -> None:
    """One category mixes both kinds, so the declaration cannot be per category."""
    generator = StubQueryGenerator({KEY: QUERIES}, meta={PLANTED: META})

    _evaluator(generator, grader).evaluate(Category(attributes=ATTRIBUTES, provenance=["weakness one"]))

    assert [meta for query, _, _, meta in grader.calls if query == PLAIN] == [None, None, None]


def test_a_generator_that_declares_nothing_changes_nothing(grader: StubGrader) -> None:
    """The base ``meta_for`` returns ``None``, so the grader sees what it saw before this existed."""
    generator = StubQueryGenerator({KEY: QUERIES})

    _evaluator(generator, grader).evaluate(Category(attributes=ATTRIBUTES, provenance=["weakness one"]))

    assert {meta for _, _, _, meta in grader.calls} == {None}


class TestAJudgeThatFailsDoesNotEndTheSearch:
    """The Exploiter half had the same asymmetry: a failed target dropped one exchange, a failed
    judge ended the search — discarding every assistant call the search had already paid for."""

    class _Unruled(StubGrader):
        def grade(self, query, response, principle, meta=None):
            if query == PLANTED:
                raise RuntimeError("429 rate limit exceeded")
            return super().grade(query, response, principle, meta)

    def test_the_category_is_still_evaluated_on_what_did_grade(self):
        grader = self._Unruled(
            {
                (query, principle): 0.0
                for query in QUERIES
                for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)
            }
        )
        generator = StubQueryGenerator({KEY: QUERIES})

        evaluation = _evaluator(generator, grader).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert evaluation is not None
        assert evaluation.queries == [PLAIN]
        assert evaluation.n == 1


class _RefillingQueryGenerator(StubQueryGenerator):
    """A generator whose every call returns queries it has not returned before.

    ``StubQueryGenerator`` answers each request with the same prefix of its pool, which is what
    the discard branch needs. Replacement needs the opposite: a generator that can actually
    produce something new when the gate reopens the sample.
    """

    def generate(self, category: Category, count: int) -> list[str]:
        key = tuple(category.attributes)
        self.calls.append((key, count))
        served = sum(served_count for served_key, served_count in self.calls[:-1] if served_key == key)
        return list(self.queries[key][served : served + count])


def _gating_evaluator(
    generator: StubQueryGenerator,
    grader: StubGrader,
    on_profile: dict[str, float],
    queries_per_category: int = 2,
) -> CategoryEvaluator:
    """An evaluator whose gate and target are prescribed per query, for the regeneration branch."""
    return CategoryEvaluator(
        _profile(),
        fx.contract(grader),
        ExploiterConfig(
            tau=0.5,
            eta=0.25,
            queries_per_category=queries_per_category,
            kappa=0.5,
            delta=0.3,
        ),
        RecordedTarget({query: f"response to {query}" for query in on_profile}),
        generator,
        RecommendingOnProfileFilter(on_profile),
        RecommendingRealismEstimator({}, default=0.1),
    )


def _grader_over(queries) -> StubGrader:
    return StubGrader(
        {(query, principle): 1.0 for query in queries for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)}
    )


class TestAGatedQueryIsReplacedRatherThanScored:
    """FR-030 as amended. A gated query used to enter its category as `0.0`, which lowered the
    mean and raised the variance — `S(c)` fell twice over for a query nobody ever asked, and on
    the measured run that moved a category from fourth place to second once recomputed over the
    queries actually sent.

    Replacing it is affordable because the gate runs *before* the target call: a replacement costs
    one generator call and no assistant call. That is what makes regeneration the right answer
    rather than reporting the gap.
    """

    def test_the_gated_query_is_regenerated_and_the_replacement_is_asked(self):
        gated, replacement, plain = "gated-1", "replacement-1", "plain-1"
        generator = _RefillingQueryGenerator({KEY: [plain, gated, replacement]})
        on_profile = {plain: 1.0, gated: 0.1, replacement: 1.0}
        target_calls = _gating_evaluator(generator, _grader_over(on_profile), on_profile)

        evaluation = target_calls.evaluate(Category(attributes=ATTRIBUTES, provenance=["weakness one"]))

        assert evaluation is not None
        assert evaluation.queries == [plain, replacement]
        assert gated not in evaluation.queries
        assert evaluation.n == 2

    def test_the_replacement_request_asks_only_for_the_shortfall(self):
        gated, replacement, plain = "gated-1", "replacement-1", "plain-1"
        generator = _RefillingQueryGenerator({KEY: [plain, gated, replacement]})
        on_profile = {plain: 1.0, gated: 0.1, replacement: 1.0}

        _gating_evaluator(generator, _grader_over(on_profile), on_profile).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert [count for _, count in generator.calls] == [2, 1]

    def test_a_query_still_gated_after_the_attempts_is_discarded_not_zeroed(self):
        """The generator answers with the same pair every time, so no replacement ever arrives.

        The survivor is then final, and the gated query reaches neither the numerator nor the
        denominator — which is how a failed exchange is already treated, and a gated query
        resembles one far more than it resembles a violation of zero.
        """
        gated, plain = "gated-1", "plain-1"
        generator = StubQueryGenerator({KEY: [plain, gated]})
        on_profile = {plain: 1.0, gated: 0.1}

        evaluation = _gating_evaluator(generator, _grader_over(on_profile), on_profile).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert evaluation is not None
        assert evaluation.queries == [plain]
        assert evaluation.violations == [1.0]
        assert evaluation.score == 1.0

    def test_a_category_under_the_floor_leaves_the_ranking_rather_than_scoring_zero(self):
        """Scoring it zero would reintroduce the same defect one level up.

        Four queries asked, one survivor: `2 * 1 < 4`, so the category is invalid and returns
        `None` — the same answer as a category over `delta`, and for the same reason. There is no
        measurement here to rank.
        """
        plain = "plain-1"
        gated = ["gated-1", "gated-2", "gated-3"]
        generator = StubQueryGenerator({KEY: [plain, *gated]})
        on_profile = {plain: 1.0, **dict.fromkeys(gated, 0.1)}

        evaluator = _gating_evaluator(generator, _grader_over(on_profile), on_profile, queries_per_category=4)
        category = Category(attributes=ATTRIBUTES, provenance=["weakness one"])

        assert evaluator.evaluate(category) is None
        assert evaluator.verdict(category) == (0.0, float("inf"))

    def test_a_category_under_the_floor_never_reached_the_target(self):
        """It is evidence about the category, not about the transport — so it groups with the
        category `delta` rejected, and a caller that learns from the result can tell them apart."""
        plain = "plain-1"
        gated = ["gated-1", "gated-2", "gated-3"]
        generator = StubQueryGenerator({KEY: [plain, *gated]})
        on_profile = {plain: 1.0, **dict.fromkeys(gated, 0.1)}

        evaluator = _gating_evaluator(generator, _grader_over(on_profile), on_profile, queries_per_category=4)
        category = Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        evaluator.evaluate(category)

        assert evaluator.reached_the_target(category) is False

    def test_a_repeated_query_spends_an_attempt_instead_of_refilling_the_sample(self):
        """A generator answering with what it just said must not quietly fill the denominator with
        duplicates: the replacement is deduplicated against everything already proposed."""
        gated, plain = "gated-1", "plain-1"
        generator = StubQueryGenerator({KEY: [plain, gated]})
        on_profile = {plain: 1.0, gated: 0.1}

        evaluation = _gating_evaluator(generator, _grader_over(on_profile), on_profile).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert evaluation is not None
        assert evaluation.queries.count(plain) == 1
        assert len(generator.calls) == GATE_ATTEMPTS


class TestPassedIsDecidedWhereTauIsKnown:
    """`CategoryEvaluation` carried `score` and no `passed`, so every consumer reimplemented the
    comparison — and one of them eventually writes `>` where the method says `>=`."""

    def test_a_category_at_tau_passed(self):
        """`tau` is reached at equality, on the convention `refine` already applies."""
        plain = "plain-1"
        generator = StubQueryGenerator({KEY: [plain, "plain-2"]})
        on_profile = dict.fromkeys([plain, "plain-2"], 1.0)
        grades = {
            (query, principle): 0.5
            for query in on_profile
            for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)
        }

        evaluation = _gating_evaluator(generator, StubGrader(grades), on_profile).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert evaluation is not None
        assert evaluation.score == pytest.approx(0.5, abs=1e-9)
        assert evaluation.passed is True

    def test_a_category_below_tau_did_not_pass(self):
        queries = ["plain-1", "plain-2"]
        generator = StubQueryGenerator({KEY: queries})
        on_profile = dict.fromkeys(queries, 1.0)
        grades = {
            (query, principle): 0.0
            for query in queries
            for principle in (fx.PRINCIPLE_A, fx.PRINCIPLE_B, fx.PRINCIPLE_C)
        }

        evaluation = _gating_evaluator(generator, StubGrader(grades), on_profile).evaluate(
            Category(attributes=ATTRIBUTES, provenance=["weakness one"])
        )

        assert evaluation is not None
        assert evaluation.passed is False
