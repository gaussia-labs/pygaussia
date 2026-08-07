"""The shipped logprob grader (T020).

Three behaviours, each of which is silently wrong in a way no aggregate would reveal:

* the verdict is the **last** verdict-shaped token in the sequence, not the first — the first
  belongs to a reasoning model's preamble (FR-007, paper invariant 6);
* a verdict is discarded when the model's own final answer does not independently parse to one
  (FR-007), because a verdict read out of a preamble is not a verdict;
* when logprobs are unusable the grade still exists and is marked fallback-derived (FR-008,
  SC-009) — raising instead would make the violation-rate denominator depend on the provider.

The fixture is arranged so the first and the last verdict-shaped token disagree, and so the
two readings produce scores on opposite sides of 0.5. A grader reading the wrong position
cannot pass by coincidence.
"""

import pytest

from gaussia.graders.logprob import LogprobGrader
from gaussia.schemas.roastme import GraderConfig, Principle
from tests.fixtures.roastme.judge_model import StubJudgeModel, token_entry

TOLERANCE = 1e-9

POSITIVE_TOKENS = ("YES", " YES", "Yes")
NEGATIVE_TOKENS = ("NO", " NO", "No")
RUBRIC = "the rubric text the user wrote and gaussia never touches"

# A reasoning preamble that opens with a verdict-shaped token, then the real verdict.
TOKEN_SEQUENCE = [
    token_entry("NO", {"NO": 0.7, "YES": 0.2}),
    token_entry("because", {"because": 0.9, "since": 0.1}),
    token_entry("YES", {"YES": 0.6, "NO": 0.3}),
]

# Read at the last verdict-shaped token: 0.6 / (0.6 + 0.3).
LAST_TOKEN_SCORE = 0.6666666666666666
# Read at the first one, which is the bug this fixture exists to catch: 0.2 / (0.2 + 0.7).
FIRST_TOKEN_SCORE = 0.2222222222222222

# The sampling fallback over k = 4: three of four samples say the principle was violated.
FALLBACK_SAMPLES = ["YES", "NO", "YES", "YES"]
FALLBACK_SCORE = 0.75


def _config(fallback_samples: int = 4) -> GraderConfig:
    return GraderConfig(
        positive_tokens=POSITIVE_TOKENS,
        negative_tokens=NEGATIVE_TOKENS,
        reasoning_budget=64,
        fallback_samples=fallback_samples,
        top_logprobs=5,
    )


def _principle(grader: LogprobGrader) -> Principle:
    return Principle(id="no_invention", weight=1.0, rubric=RUBRIC, grader=grader)


def _grade(model: StubJudgeModel, config: GraderConfig | None = None):
    grader = LogprobGrader(model=model, config=config or _config())
    return grader.grade("the probe query", "the assistant response", _principle(grader))


class TestVerdictLocation:
    def test_the_last_verdict_shaped_token_is_the_verdict(self):
        model = StubJudgeModel(token_entries=TOKEN_SEQUENCE, final_content="YES")
        grade = _grade(model)

        assert grade.score == pytest.approx(LAST_TOKEN_SCORE, abs=TOLERANCE)

    def test_the_first_verdict_shaped_token_is_not(self):
        model = StubJudgeModel(token_entries=TOKEN_SEQUENCE, final_content="YES")
        grade = _grade(model)

        assert grade.score != pytest.approx(FIRST_TOKEN_SCORE, abs=1e-6)
        assert grade.score > 0.5

    def test_the_grade_records_its_own_provenance(self):
        """FR-005: substituting a grader must leave everything downstream untouched, which is
        only true if the grade carries the grader, the model and the verdict method itself."""
        model = StubJudgeModel(token_entries=TOKEN_SEQUENCE, final_content="YES")
        grade = _grade(model)

        assert grade.principle == "no_invention"
        assert grade.method.strip() != ""
        assert "fallback" not in grade.method
        assert grade.model is not None
        assert grade.evidence != {}

    def test_the_configured_surface_forms_and_budget_reach_the_provider(self):
        model = StubJudgeModel(token_entries=TOKEN_SEQUENCE, final_content="YES")
        config = _config()
        _grade(model, config)

        assert model.bind_calls != []
        assert model.bind_calls[0].get("logprobs") is True
        assert model.bind_calls[0].get("top_logprobs") == config.top_logprobs

    def test_the_user_s_rubric_is_passed_unmodified(self):
        """FR-006: gaussia never substitutes or appends to the rubric."""
        model = StubJudgeModel(token_entries=TOKEN_SEQUENCE, final_content="YES")
        _grade(model)

        assert RUBRIC in model.prompt_text()


class TestVerdictDiscarded:
    def test_an_unparsable_final_answer_discards_the_logprob_verdict(self):
        """FR-007: the model's own final answer has to parse to a verdict independently."""
        model = StubJudgeModel(
            token_entries=TOKEN_SEQUENCE,
            final_content="I am not certain either way",
            sample_contents=FALLBACK_SAMPLES,
        )
        grade = _grade(model)

        assert grade.score != pytest.approx(LAST_TOKEN_SCORE, abs=1e-6)
        assert "fallback" in grade.method

    def test_the_discarded_verdict_is_replaced_by_sampling_over_k(self):
        model = StubJudgeModel(
            token_entries=TOKEN_SEQUENCE,
            final_content="I am not certain either way",
            sample_contents=FALLBACK_SAMPLES,
        )
        config = _config(fallback_samples=len(FALLBACK_SAMPLES))
        grade = _grade(model, config)

        assert model.sample_invocations == config.fallback_samples
        assert grade.score == pytest.approx(FALLBACK_SCORE, abs=TOLERANCE)


class TestSamplingFallback:
    def test_a_provider_with_no_usable_logprobs_still_produces_a_grade(self):
        """SC-009: the grade exists and says how it was obtained. Raising here would make the
        violation-rate denominator depend on provider behaviour (spec D13)."""
        model = StubJudgeModel(
            token_entries=TOKEN_SEQUENCE,
            final_content="YES",
            sample_contents=FALLBACK_SAMPLES,
            logprobs_supported=False,
        )
        grade = _grade(model, _config(fallback_samples=len(FALLBACK_SAMPLES)))

        assert grade.score == pytest.approx(FALLBACK_SCORE, abs=TOLERANCE)
        assert "fallback" in grade.method
        assert model.logprob_invocations == 0

    def test_no_verdict_token_in_the_sequence_also_falls_back(self):
        """FR-008 covers both cases: logprobs unusable, or usable with no verdict token in them."""
        model = StubJudgeModel(
            token_entries=[token_entry("maybe", {"maybe": 0.8, "perhaps": 0.2})],
            final_content="maybe",
            sample_contents=FALLBACK_SAMPLES,
        )
        grade = _grade(model, _config(fallback_samples=len(FALLBACK_SAMPLES)))

        assert grade.score == pytest.approx(FALLBACK_SCORE, abs=TOLERANCE)
        assert "fallback" in grade.method

    def test_the_fallback_grade_is_still_in_range_and_auditable(self):
        model = StubJudgeModel(
            token_entries=TOKEN_SEQUENCE,
            final_content="YES",
            sample_contents=FALLBACK_SAMPLES,
            logprobs_supported=False,
        )
        grade = _grade(model, _config(fallback_samples=len(FALLBACK_SAMPLES)))

        assert 0.0 <= grade.score <= 1.0
        assert grade.evidence != {}
