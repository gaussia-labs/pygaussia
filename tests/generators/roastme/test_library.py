"""The Probe Library and its engines (T023).

Four things, each tied to an invariant rather than to an implementation:

* engines compose over one base, duplicates merge, and every surviving probe still records which
  engine produced it — otherwise the absence/breadth trade-off stops being measurable after
  composition (FR-022);
* the graph engine's absence labels are confirmed by exact enumeration and the retrieval engine's
  are not, and the retrieval engine says so on the probe (FR-023, SC-004). Similarity search is
  *structurally* unable to decide absence, because it never reveals what it failed to retrieve;
* the enumeration engine refuses to run until an enumerator is injected (spec D14);
* with no knowledge base, probes still come back — domain-agnostic, with an empty hook (FR-024).

The enumeration the absence labels are scored against is the test's, never the engine's: deriving
a label from the enumeration used for scoring would make the whole downstream test circular
(FR-021).
"""

import inspect

import numpy as np
import pytest

from gaussia.core.embedder import Embedder
from gaussia.core.probe_engine import ProbeEngine
from gaussia.generators.roastme.probes.enumeration import EnumerationProbeEngine
from gaussia.generators.roastme.probes.graph import GraphProbeEngine
from gaussia.generators.roastme.probes.library import ProbeLibrary
from gaussia.generators.roastme.probes.retrieval import RetrievalProbeEngine
from gaussia.generators.roastme.probes.transforms import TRANSFORMS
from gaussia.schemas.roastme import Document
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import StubEntityEnumerator, StubProbeEngine

TRANSFORM_KEY = next(iter(TRANSFORMS))


class _StubEmbedder(Embedder):
    """A deterministic embedder: one axis per known entity, so similarity is a lookup."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        entities = sorted(fx.KNOWN_ENTITIES)
        rows = [[1.0 if entity in sentence else 0.0 for entity in entities] + [0.5] for sentence in sentences]
        return np.array(rows, dtype=float)


def _catalogue():
    return fx.catalogue(TRANSFORM_KEY)


def _absence_probes(probes):
    return [probe for probe in probes if probe.hook is not None and probe.hook.doc == 0]


class TestComposition:
    def test_every_surviving_probe_records_its_engine(self):
        first = StubProbeEngine(engine_name="engine-one", kinds=frozenset({fx.ENTITY_KIND}), probes=fx.probes()[:2])
        second = StubProbeEngine(engine_name="engine-two", kinds=frozenset({fx.ENTITY_KIND}), probes=fx.probes()[2:4])
        library = ProbeLibrary(engines=[first, second])

        probes = library.generate(fx.documents(), _catalogue())

        assert probes != []
        assert all(probe.engine is not None for probe in probes)
        assert {probe.engine for probe in probes} == {"engine-one", "engine-two"}

    def test_duplicates_merge(self):
        shared = fx.probes()[:1]
        first = StubProbeEngine(engine_name="engine-one", kinds=frozenset({fx.ENTITY_KIND}), probes=shared)
        second = StubProbeEngine(engine_name="engine-two", kinds=frozenset({fx.ENTITY_KIND}), probes=shared)
        library = ProbeLibrary(engines=[first, second])

        probes = library.generate(fx.documents(), _catalogue())

        assert len(probes) == 1
        assert probes[0].engine in {"engine-one", "engine-two"}

    def test_only_the_engines_that_can_handle_a_document_see_it(self):
        """Engine selection is composition, not a cascade: every engine answering yes contributes."""
        narrow = StubProbeEngine(
            engine_name="engine-narrow",
            kinds=frozenset({fx.ENTITY_KIND}),
            probes=fx.probes()[:1],
            handles=frozenset({fx.DOCUMENT_STRUCTURED_ID}),
        )
        library = ProbeLibrary(engines=[narrow])

        library.generate(fx.documents(), _catalogue())

        assert narrow.generate_calls == [(1, len(_catalogue().strategies))]


class TestNoKnowledgeBase:
    def test_probes_still_come_back_with_an_empty_hook(self):
        """FR-024: a domain-agnostic probe leans on no entity, so fabricating a placeholder hook
        would put an invented entity into the retained hooks the Exploiter grounds categories on."""
        engine = StubProbeEngine(engine_name="engine-one", kinds=frozenset({fx.ENTITY_KIND}), probes=fx.probes())
        library = ProbeLibrary(engines=[engine])

        probes = library.generate([], _catalogue())

        assert probes != []
        assert all(probe.hook is None for probe in probes)

    def test_a_hookless_probe_is_still_recognisably_a_control_or_not(self):
        engine = StubProbeEngine(engine_name="engine-one", kinds=frozenset({fx.ENTITY_KIND}), probes=fx.probes())
        library = ProbeLibrary(engines=[engine])

        probes = library.generate([], _catalogue())

        assert any(probe.plugin is None for probe in probes)
        assert any(probe.plugin is not None for probe in probes)


class TestAbsenceReliability:
    def test_the_graph_engine_s_absence_labels_are_confirmed_by_exact_enumeration(self):
        """SC-004: absence from a complete graph is absence, so the label is trustworthy."""
        engine = GraphProbeEngine()
        probes = engine.generate(fx.documents(), _catalogue())
        absent = _absence_probes(probes)

        assert absent != []
        for probe in absent:
            assert probe.hook.absence_reliable is True
            assert probe.hook.references not in fx.KNOWN_ENTITIES

    def test_the_graph_engine_s_presence_labels_are_confirmed_too(self):
        engine = GraphProbeEngine()
        probes = engine.generate(fx.documents(), _catalogue())
        present = [probe for probe in probes if probe.hook is not None and probe.hook.doc == 1]

        for probe in present:
            assert probe.hook.references in fx.KNOWN_ENTITIES

    def test_the_retrieval_engine_marks_its_absence_labels_unreliable(self):
        """FR-023: an unreliable absence label must never be indistinguishable from a confirmed
        one. Similarity search never reveals what it failed to retrieve."""
        engine = RetrievalProbeEngine(embedder=_StubEmbedder())
        probes = engine.generate(fx.documents(), _catalogue())
        absent = _absence_probes(probes)

        assert absent != []
        assert all(probe.hook.absence_reliable is False for probe in absent)

    def test_composition_keeps_the_two_distinguishable(self):
        library = ProbeLibrary(engines=[GraphProbeEngine(), RetrievalProbeEngine(embedder=_StubEmbedder())])
        absent = _absence_probes(library.generate(fx.documents(), _catalogue()))

        assert {probe.hook.absence_reliable for probe in absent} == {True, False}


class TestEnumerationEngine:
    def test_it_refuses_to_run_with_no_enumerator(self):
        """Spec D14: it is the only engine that cannot work from a knowledge base alone, which is
        why it is opt-in rather than omitted."""
        with pytest.raises((TypeError, ValueError), match=r"(?i)enumerator"):
            EnumerationProbeEngine()

    def test_it_runs_once_an_enumerator_is_injected(self):
        enumerator = StubEntityEnumerator({fx.ENTITY_KIND: fx.KNOWN_ENTITIES})
        engine = EnumerationProbeEngine(enumerator=enumerator)

        probes = engine.generate(fx.documents(), _catalogue())

        assert probes != []
        assert enumerator.calls != []

    def test_it_reads_whether_a_document_s_boundary_is_enumerable(self):
        """`Document.structured` exists for this engine: it is what says whether absence can be
        established over a document at all."""
        engine = EnumerationProbeEngine(enumerator=StubEntityEnumerator({fx.ENTITY_KIND: fx.KNOWN_ENTITIES}))

        assert engine.can_handle(Document(id="d", content="c", structured=True)) is True
        assert engine.can_handle(Document(id="d", content="c", structured=False)) is False

    def test_its_absence_labels_agree_with_the_enumeration(self):
        engine = EnumerationProbeEngine(enumerator=StubEntityEnumerator({fx.ENTITY_KIND: fx.KNOWN_ENTITIES}))
        probes = engine.generate(fx.documents(), _catalogue())

        for probe in _absence_probes(probes):
            assert probe.hook.absence_reliable is True
            assert probe.hook.references not in fx.KNOWN_ENTITIES

    def test_it_takes_the_enumerator_as_an_injected_collaborator(self):
        """Domain-specific by nature, so gaussia specifies the interface and ships none."""
        parameters = inspect.signature(EnumerationProbeEngine.__init__).parameters
        assert "enumerator" in parameters


class TestFourEnginesShip:
    def test_all_four_sit_behind_the_one_interface(self):
        """FR-022: retrieval, graph, multi-hop and enumeration, composable over one base."""
        from gaussia.generators.roastme.probes.grag import MultiHopProbeEngine

        for engine_class in (RetrievalProbeEngine, GraphProbeEngine, MultiHopProbeEngine, EnumerationProbeEngine):
            assert issubclass(engine_class, ProbeEngine)

    def test_each_declares_a_name_and_the_entity_kinds_it_handles(self):
        graph = GraphProbeEngine()
        retrieval = RetrievalProbeEngine(embedder=_StubEmbedder())

        for engine in (graph, retrieval):
            assert engine.name.strip() != ""
            assert isinstance(engine.entity_kinds, frozenset)

    def test_two_engines_over_one_base_do_not_report_the_same_name(self):
        assert GraphProbeEngine().name != RetrievalProbeEngine(embedder=_StubEmbedder()).name
