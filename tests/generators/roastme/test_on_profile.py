"""What the shipped on-profile filter shows the model (FR-013, FR-041, paper invariant 3).

This filter is the one shipped component that renders the profile into a prompt, so it is where
the user's private vocabulary leaks if it leaks anywhere. The profile's weakness map crosses
scrubbed — `test_profiler.py` scans it — but the retained hooks cross whole, carrying `kind` (the
strategy's entity kind) and `how` (the transform key). That is deliberate: FR-013 requires the
hooks themselves, and an evaluator reading the report needs to know what was done to which kind
of entity.

Invariant 3 is satisfied not by blanking those fields but by no consumer reading them, and this
module is where that stops being a claim: the hooks handed to the filter carry the identifiers,
and the prompt that reaches the model must not.

The model is a stub, so nothing here needs a key (SC-010).
"""

from typing import Any

import pytest

from gaussia.generators.roastme.searches.on_profile import RECOMMENDED_KAPPA, JudgeOnProfileFilter
from gaussia.schemas.roastme import AssistantProfile, KnowledgeHook, WeaknessEntry
from tests.fixtures.roastme import expected as fx

QUERY = "a question a real user would send"
DESCRIPTOR = "asks for an exception to a stated rule"
REFERENCE = "entity-one"


class _BoundStubScoringModel:
    def __init__(self, model: "StubScoringModel", schema: Any):
        self._model = model
        self._schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self._model.invoked_with.append(messages)
        return self._schema(score=self._model.score)


class StubScoringModel:
    """A chat model answering a prescribed score and recording what it was asked.

    Structured output is what the filter binds, so the double answers through the schema the
    filter passed rather than assuming its shape.
    """

    def __init__(self, score: float = 0.9):
        self.score = score
        self.invoked_with: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _BoundStubScoringModel:
        return _BoundStubScoringModel(self, schema)

    def prompt_text(self) -> str:
        return " ".join(str(message.content) for messages in self.invoked_with for message in messages)


def _profile() -> AssistantProfile:
    """A profile whose retained hook carries the user's vocabulary, as a real one does."""
    return AssistantProfile(
        weaknesses=[
            WeaknessEntry(
                principle=fx.PRINCIPLE_A,
                descriptor=DESCRIPTOR,
                rate=0.75,
                n=4,
                standard_error=fx.SE_RATE_075_N4,
            )
        ],
        hooks=[
            KnowledgeHook(
                kind=fx.ENTITY_KIND,
                references=REFERENCE,
                doc=0,
                how=fx.TRANSFORM_INVENT,
                principle=fx.PRINCIPLE_A,
            )
        ],
    )


def _score(model: StubScoringModel, profile: AssistantProfile | None = None) -> float:
    return JudgeOnProfileFilter(model=model).score(QUERY, profile or _profile())


class TestWhatReachesTheModel:
    def test_the_hook_is_rendered_by_its_reference_alone(self):
        """Invariant 3 enforced where the profile is read: the identifiers are on the hook the
        filter was handed, and they are not in the prompt it built."""
        model = StubScoringModel()
        _score(model)
        prompt = model.prompt_text()

        assert REFERENCE in prompt
        assert DESCRIPTOR in prompt
        for identifier in fx.OPAQUE_IDENTIFIERS:
            assert identifier not in prompt

    def test_the_query_and_the_sample_size_travel_with_it(self):
        """A descriptor resting on four probes must not be shown as if it were settled (FR-012)."""
        model = StubScoringModel()
        _score(model)
        prompt = model.prompt_text()

        assert QUERY in prompt
        assert "4" in prompt

    def test_an_empty_profile_still_renders(self):
        """A first iteration with nothing retained has to produce a prompt, not a blank section."""
        model = StubScoringModel()
        _score(model, AssistantProfile())

        assert model.prompt_text().strip() != ""


class TestTheScale:
    def test_it_declares_the_kappa_it_recommends(self):
        """FR-041: the scale is the filter's own, so the threshold travels with it."""
        assert JudgeOnProfileFilter.recommended_threshold == RECOMMENDED_KAPPA

    @pytest.mark.parametrize(("answered", "expected"), [(1.2, 1.0), (-0.5, 0.0), (0.7, 0.7)])
    def test_an_answer_off_the_scale_is_clamped_rather_than_rejected(self, answered, expected):
        """A model answering 1.2 has still judged the query on profile; letting the value out of
        range would put the gate on a scale the recommended `kappa` was never calibrated for."""
        assert _score(StubScoringModel(score=answered)) == expected
