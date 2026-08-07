"""The shipped query generator, and the one failure mode it must not have (T045).

A category is only as strong as the queries that stand for it, and the denominator of `S(c)` is
how many were asked for. So the behaviour under test is narrow and load-bearing: `generate` either
returns exactly `count` distinct queries or raises. Returning three where ten were asked would
divide by three and report the category as measured, which is the silent failure the module's
docstring names.

Three paths follow from that: the ordinary one, the short reply that is re-asked for the shortfall
alone, and the model that never fills the sample — a refusal answers with no questions at all, and
after the attempt budget the run fails loudly rather than shrinking.

The model is a stub, so nothing here needs a key or a network.
"""

from typing import Any

import pytest

from gaussia.generators.roastme.searches.query_generation import DEFAULT_ATTEMPTS, PromptedQueryGenerator
from gaussia.schemas.roastme import Category

ATTRIBUTE_ONE = "asks for an exception to a stated rule"
ATTRIBUTE_TWO = "quotes a figure from the price list"
PROVENANCE = ["weakness one", "hook one"]


class _BoundStubQueryModel:
    def __init__(self, model: "StubQueryModel", schema: Any):
        self._model = model
        self._schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self._model.invoked_with.append(messages)
        return self._schema(questions=self._model.reply(len(self._model.invoked_with) - 1))


class StubQueryModel:
    """A chat model whose successive replies are prescribed, answering nothing once they run out.

    Structured output is what the generator binds, so the double answers through the schema the
    generator passed rather than assuming its shape. An exhausted fixture yields an empty reply
    rather than raising, which is what a model that has stopped cooperating does.
    """

    def __init__(self, replies: list[list[str]]):
        self.replies = replies
        self.invoked_with: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _BoundStubQueryModel:
        return _BoundStubQueryModel(self, schema)

    def reply(self, index: int) -> list[str]:
        return list(self.replies[index]) if index < len(self.replies) else []

    def prompts(self) -> list[str]:
        return ["\n".join(str(message.content) for message in messages) for messages in self.invoked_with]


def _category() -> Category:
    return Category(attributes=[ATTRIBUTE_ONE, ATTRIBUTE_TWO], provenance=list(PROVENANCE))


def _generate(replies: list[list[str]], count: int, attempts: int = DEFAULT_ATTEMPTS):
    model = StubQueryModel(replies)
    queries = PromptedQueryGenerator(model=model, attempts=attempts).generate(_category(), count)
    return queries, model


class TestTheOrdinaryPath:
    def test_a_categorys_attributes_become_the_requested_number_of_queries(self):
        queries, model = _generate([["a plain question", "another one", "a third"]], count=3)

        assert queries == ["a plain question", "another one", "a third"]
        assert len(model.invoked_with) == 1

    def test_a_reply_longer_than_asked_for_is_truncated(self):
        """The denominator is `count`, so over-delivery may not inflate it either."""
        queries, _ = _generate([["one", "two", "three", "four", "five"]], count=3)

        assert queries == ["one", "two", "three"]

    def test_blank_answers_are_discarded_and_the_rest_stripped(self):
        queries, model = _generate([["  one  ", "", "   ", "two"], ["three"]], count=3)

        assert queries == ["one", "two", "three"]
        assert len(model.invoked_with) == 2

    def test_a_repeated_question_does_not_fill_the_sample_twice(self):
        """Distinct is the requirement: two identical queries measure one query twice."""
        queries, model = _generate([["one", "one", "two"], ["two", "three"], ["four"]], count=4)

        assert queries == ["one", "two", "three", "four"]
        assert len(model.invoked_with) == 3


class TestTheShortReply:
    def test_a_short_reply_is_re_asked_for_the_shortfall_alone(self):
        _, model = _generate([["one", "two"], ["three", "four"]], count=4)

        assert len(model.invoked_with) == 2
        assert "Write 4 distinct questions" in model.prompts()[0]
        assert "Write 2 distinct questions" in model.prompts()[1]

    def test_re_asking_stops_as_soon_as_the_sample_is_full(self):
        queries, model = _generate([["one", "two"], ["three", "four"], ["five", "six"]], count=4)

        assert queries == ["one", "two", "three", "four"]
        assert len(model.invoked_with) == 2

    def test_a_run_that_cannot_fill_the_sample_fails_loudly(self):
        """Rather than shrinking the denominator of S(c) without saying so."""
        with pytest.raises(ValueError, match=r"StubQueryModel produced 2 distinct queries of the 4 asked for"):
            _generate([["one", "two"], ["one"], ["two"]], count=4)

    def test_a_model_that_answers_nothing_at_all_raises(self):
        """The shape a refusal arrives in: the call succeeds and carries no question."""
        with pytest.raises(ValueError, match=r"produced 0 distinct queries of the 2 asked for"):
            _generate([[], [], []], count=2)

    def test_the_attempt_budget_bounds_the_re_asking(self):
        """A knob of gaussia's own implementation (FR-040), and the reason the loop is finite."""
        model = StubQueryModel([[], [], ["one", "two"], ["three", "four"]])
        generator = PromptedQueryGenerator(model=model, attempts=2)

        with pytest.raises(ValueError, match=r"produced 0 distinct queries"):
            generator.generate(_category(), 2)

        assert len(model.invoked_with) == 2

    def test_the_message_names_the_model_that_fell_short(self):
        """FR-039: a weak report has to be attributable to the piece that can be swapped."""
        with pytest.raises(ValueError, match=r"StubQueryModel"):
            _generate([["one"]], count=2, attempts=1)


class TestWhatReachesTheModel:
    def test_every_attribute_is_shown_and_shown_as_a_conjunction(self):
        """A menu would realise each attribute in its own query, and the category would then
        stand for something it never proposed."""
        _, model = _generate([["one", "two"]], count=2)
        prompt = model.prompts()[0]

        assert f"- {ATTRIBUTE_ONE}" in prompt
        assert f"- {ATTRIBUTE_TWO}" in prompt
        assert "ALL of them at once" in prompt

    def test_the_provenance_stays_out_of_the_prompt(self):
        """It is the report's audit trail, not something a query has to exhibit; showing it would
        put the Profiler's own bookkeeping into the traffic the target sees."""
        _, model = _generate([["one", "two"]], count=2)
        prompt = model.prompts()[0]

        for entry in PROVENANCE:
            assert entry not in prompt
