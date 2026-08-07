"""Conformance of every double to its `core/` interface (T027).

FR-019 asks gaussia to *specify* ten interfaces. A specification a third party implements against
needs an executable definition of conformance, and this module is it: each double is checked for
being a real instance of its abstraction, for returning the declared shape, and — the part a
docstring cannot give — for the abstraction rejecting an implementation that leaves a method out.

Nothing here reaches an implementation, so these assertions hold from the moment the interfaces
exist. That is deliberate: a conformance suite that only works once the library is finished is
not a specification.
"""

import inspect

import pytest

from gaussia.core.category_search import CategorySearch
from gaussia.core.entity_enumerator import EntityEnumerator
from gaussia.core.grader import Grader
from gaussia.core.hook_verifier import HookVerifier
from gaussia.core.on_profile_filter import OnProfileFilter
from gaussia.core.probe_engine import ProbeEngine
from gaussia.core.query_generator import QueryGenerator
from gaussia.core.realism_estimator import RealismEstimator
from gaussia.core.target_assistant import TargetAssistant
from gaussia.core.transform import Transform
from gaussia.schemas.roastme import (
    AssistantProfile,
    Category,
    KnowledgeHook,
    PrincipleGrade,
    Probe,
    TargetResponse,
)
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import (
    FailingTarget,
    NoLogprobStubGrader,
    RecommendingOnProfileFilter,
    RecommendingRealismEstimator,
    SilentOnProfileFilter,
    SilentRealismEstimator,
    StubCategorySearch,
    StubEntityEnumerator,
    StubGrader,
    StubHookVerifier,
    StubProbeEngine,
    StubQueryGenerator,
    StubTransform,
)

INTERFACES = [
    Grader,
    ProbeEngine,
    EntityEnumerator,
    HookVerifier,
    Transform,
    TargetAssistant,
    QueryGenerator,
    OnProfileFilter,
    RealismEstimator,
    CategorySearch,
]


class TestTheSpecificationIsEnforced:
    def test_ten_interfaces(self):
        assert len(INTERFACES) == 10

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_an_interface_cannot_be_instantiated(self, interface):
        with pytest.raises(TypeError, match=r"(?i)abstract"):
            interface()

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_an_incomplete_implementation_is_rejected(self, interface):
        incomplete = type(f"Incomplete{interface.__name__}", (interface,), {})

        with pytest.raises(TypeError, match=r"(?i)abstract"):
            incomplete()

    @pytest.mark.parametrize("interface", [Grader, QueryGenerator, OnProfileFilter, RealismEstimator, TargetAssistant])
    def test_the_interfaces_downstream_of_the_corpus_cannot_reach_it(self, interface):
        """Interface segregation: a grader does not inherit knowledge-base access, which is what
        keeps paper invariant 2 structurally true rather than merely documented."""
        for name, member in inspect.getmembers(interface, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            annotations = [str(parameter.annotation) for parameter in inspect.signature(member).parameters.values()]
            assert not any("Document" in annotation for annotation in annotations)


class TestGraderDoubles:
    @pytest.mark.parametrize("grader_class", [StubGrader, NoLogprobStubGrader])
    def test_conforms(self, grader_class):
        grader = grader_class(fx.GRADES)
        assert isinstance(grader, Grader)

    def test_returns_a_grade_in_range_with_its_provenance(self):
        grader = fx.stub_grader()
        principle = fx.contract(grader).principles[0]
        grade = grader.grade(fx.QUERY_ONE, fx.RESPONSES[fx.QUERY_ONE], principle)

        assert isinstance(grade, PrincipleGrade)
        assert 0.0 <= grade.score <= 1.0
        assert grade.method.strip() != ""

    def test_the_no_logprob_double_marks_itself_fallback_derived(self):
        """SC-009: the grade exists, and says it came from the fallback."""
        grader = NoLogprobStubGrader(fx.GRADES)
        principle = fx.contract(grader).principles[0]
        grade = grader.grade(fx.QUERY_ONE, fx.RESPONSES[fx.QUERY_ONE], principle)

        assert "fallback" in grade.method

    def test_meta_is_optional_because_a_generated_query_has_no_probe(self):
        grader = fx.stub_grader()
        principle = fx.contract(grader).principles[0]

        assert grader.grade(fx.QUERY_ONE, "r", principle, None) is not None
        assert grader.grade(fx.QUERY_ONE, "r", principle, {"real_value": "x"}) is not None


class TestTargetDoubles:
    def test_the_recorded_target_conforms(self):
        target = fx.recorded_target()
        assert isinstance(target, TargetAssistant)

        response = target.send(fx.QUERY_ONE)
        assert isinstance(response, TargetResponse)
        assert response.failed is False
        assert response.content == fx.RESPONSES[fx.QUERY_ONE]

    def test_the_recorded_target_reports_the_failed_exchange(self):
        response = fx.recorded_target().send(fx.QUERY_FAILED)

        assert response.failed is True
        assert response.content == ""
        assert response.failure_reason is not None

    def test_the_failing_target_conforms(self):
        target = FailingTarget()
        assert isinstance(target, TargetAssistant)

        response = target.send("anything")
        assert response.failed is True

    def test_a_session_identifier_is_optional(self):
        target = fx.recorded_target()
        target.send(fx.QUERY_ONE)
        target.send(fx.QUERY_ONE, "session-1")

        assert target.sent == [(fx.QUERY_ONE, None), (fx.QUERY_ONE, "session-1")]


class TestProbeGenerationDoubles:
    def test_the_probe_engine_conforms(self):
        engine = StubProbeEngine(engine_name="e", kinds=frozenset({fx.ENTITY_KIND}), probes=fx.probes())

        assert isinstance(engine, ProbeEngine)
        assert engine.name == "e"
        assert engine.entity_kinds == frozenset({fx.ENTITY_KIND})
        assert engine.can_handle(fx.documents()[0]) is True

        probes = engine.generate(fx.documents(), fx.catalogue("tf-any"))
        assert all(isinstance(probe, Probe) for probe in probes)
        assert all(probe.engine == "e" for probe in probes)

    def test_the_enumerator_conforms(self):
        enumerator = StubEntityEnumerator({fx.ENTITY_KIND: fx.KNOWN_ENTITIES})

        assert isinstance(enumerator, EntityEnumerator)
        assert enumerator.enumerate_entities(fx.ENTITY_KIND, fx.documents()) == fx.KNOWN_ENTITIES
        assert enumerator.enumerate_entities("unknown-kind", fx.documents()) == frozenset()

    def test_the_hook_verifier_conforms(self):
        hook = KnowledgeHook(kind=fx.ENTITY_KIND, references=fx.SHARED_ENTITY, doc=1, how=fx.TRANSFORM_KEEP)
        verifier = StubHookVerifier({fx.SHARED_ENTITY: True})

        assert isinstance(verifier, HookVerifier)
        assert verifier.verify(hook, fx.documents()) is True

    def test_the_transform_conforms(self):
        transform = StubTransform("tf-stub", {"entity-alpha": "entity-alpha-prime"})

        assert isinstance(transform, Transform)
        assert transform.key == "tf-stub"
        assert transform.apply("entity-alpha") == "entity-alpha-prime"


class TestSearchDoubles:
    def test_the_query_generator_conforms_and_returns_exactly_what_was_asked(self):
        generator = StubQueryGenerator({("a",): ["q1", "q2", "q3"]})
        category = Category(attributes=["a"], provenance=["weakness one"])

        assert isinstance(generator, QueryGenerator)
        assert generator.generate(category, 2) == ["q1", "q2"]
        assert generator.calls == [(("a",), 2)]

    def test_the_filters_conform_and_differ_only_in_what_they_recommend(self):
        recommending = RecommendingOnProfileFilter({"q": 0.9})
        silent = SilentOnProfileFilter({"q": 0.9})

        assert isinstance(recommending, OnProfileFilter)
        assert isinstance(silent, OnProfileFilter)
        assert recommending.recommended_threshold is not None
        assert silent.recommended_threshold is None
        assert recommending.score("q", AssistantProfile()) == 0.9

    def test_the_estimators_conform_and_differ_only_in_what_they_recommend(self):
        recommending = RecommendingRealismEstimator({("q",): 0.2})
        silent = SilentRealismEstimator({("q",): 0.2})

        assert isinstance(recommending, RealismEstimator)
        assert isinstance(silent, RealismEstimator)
        assert recommending.recommended_threshold is not None
        assert silent.recommended_threshold is None
        assert recommending.estimate(["q"]) == 0.2

    def test_the_category_search_conforms(self):
        search = StubCategorySearch([])
        assert isinstance(search, CategorySearch)

    def test_the_search_reads_the_thresholds_and_never_resolves_them(self):
        """FR-041: `kappa` and `delta` arrive already resolved, so one run has one pair in force."""
        parameters = inspect.signature(CategorySearch.search).parameters

        assert "config" in parameters
        assert "on_profile_filter" in parameters
        assert "realism_estimator" in parameters
