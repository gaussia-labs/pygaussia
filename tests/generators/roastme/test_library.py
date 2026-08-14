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
from gaussia.core.transform import Transform
from gaussia.generators.roastme.probes.catalogue import validate_catalogue
from gaussia.generators.roastme.probes.enumeration import EnumerationProbeEngine
from gaussia.generators.roastme.probes.grag import MultiHopProbeEngine
from gaussia.generators.roastme.probes.graph import GraphProbeEngine
from gaussia.generators.roastme.probes.library import ProbeLibrary
from gaussia.generators.roastme.probes.mentions import CompoundTokenExtractor, MentionExtractor
from gaussia.generators.roastme.probes.retrieval import RetrievalProbeEngine
from gaussia.generators.roastme.probes.transforms import TRANSFORMS, available
from gaussia.schemas.roastme import Catalogue, Document, PluginSpec, StrategySpec
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


class _HeadingExtractor(MentionExtractor):
    """Reads entities out of second-level headings, which is what a product site carries."""

    def extract(self, documents):
        return frozenset(
            line.removeprefix(fx.HEADING_PREFIX).strip()
            for document in documents
            for line in document.content.splitlines()
            if line.startswith(fx.HEADING_PREFIX)
        )


class _PlausibleFake(Transform):
    """A premise that reads like a real name instead of a suffixed one."""

    @property
    def key(self) -> str:
        return "plausible_fake"

    def apply(self, entity: str) -> str:
        return entity.replace("Libre", "Popular") if "Libre" in entity else f"{entity} Plus"


class _CollidingTransform(Transform):
    @property
    def key(self) -> str:
        return TRANSFORM_KEY

    def apply(self, entity: str) -> str:
        return entity


class TestACorpusOfOrdinaryWords:
    """The shape the default extractor cannot read, which is the shape most corpora have.

    The three engines that read a corpus find mentions through an extractor, and the shipped one
    recognises compound identifiers — `POLICY-1`, `Articulo_25`. A corpus of ordinary words yields it
    nothing, and *nothing is what the engine must then produce*: a boundary it cannot see is not a
    boundary it may guess at. What makes the corpus usable is injecting a reading of it, not changing
    the engine.
    """

    def test_the_default_extractor_finds_nothing_in_it(self):
        assert CompoundTokenExtractor().extract(fx.ordinary_word_documents()) == frozenset()

    def test_so_the_engines_produce_no_probes_rather_than_probes_over_junk(self):
        catalogue = fx.catalogue(TRANSFORM_KEY)
        for engine in (
            GraphProbeEngine(entity_kinds={fx.ENTITY_KIND}),
            MultiHopProbeEngine(entity_kinds={fx.ENTITY_KIND}),
        ):
            assert engine.generate(fx.ordinary_word_documents(), catalogue) == []

    def test_an_injected_extractor_makes_the_same_engines_produce_probes(self):
        engine = GraphProbeEngine(entity_kinds={fx.ENTITY_KIND}, extractor=_HeadingExtractor())

        probes = engine.generate(fx.ordinary_word_documents(), fx.catalogue(TRANSFORM_KEY))

        assert probes != []
        premises = {probe.hook.references for probe in probes if probe.hook is not None}
        assert any(entity in premise for entity in fx.ORDINARY_WORD_ENTITIES for premise in premises)

    def test_the_engine_records_which_reading_produced_its_probes(self):
        extractor = _HeadingExtractor()
        assert GraphProbeEngine(extractor=extractor).extractor is extractor
        assert isinstance(GraphProbeEngine().extractor, CompoundTokenExtractor)


class TestATransformTheUserSupplies:
    """`Transform` is one of the ten interfaces, so a catalogue may name one gaussia did not write.

    The four shipped assume the entity shape of the paper's own corpus: a `-2` suffix reads as a near
    miss of `POLICY-1` and as a typo of `Cuenta Digital Libre`. Supplying one is how a corpus gets a
    premise its own users would recognise, and the same sequence has to reach validation and
    generation or a catalogue that validated can still fail.
    """

    def test_validation_accepts_a_strategy_naming_it(self):
        catalogue = fx.catalogue(_PlausibleFake().key, control_transform_key=TRANSFORM_KEY)

        validate_catalogue(
            catalogue,
            fx.contract(fx.stub_grader()),
            [GraphProbeEngine(entity_kinds={fx.ENTITY_KIND})],
            transforms=[_PlausibleFake()],
        )

    def test_validation_refuses_it_when_it_was_not_supplied(self):
        catalogue = fx.catalogue(_PlausibleFake().key, control_transform_key=TRANSFORM_KEY)

        with pytest.raises(ValueError, match="outside"):
            validate_catalogue(
                catalogue, fx.contract(fx.stub_grader()), [GraphProbeEngine(entity_kinds={fx.ENTITY_KIND})]
            )

    def test_generation_builds_the_premise_it_defines(self):
        engine = GraphProbeEngine(
            entity_kinds={fx.ENTITY_KIND},
            extractor=_HeadingExtractor(),
            transforms=[_PlausibleFake()],
        )

        probes = engine.generate(
            fx.ordinary_word_documents(),
            fx.catalogue(_PlausibleFake().key, control_transform_key=TRANSFORM_KEY),
        )

        supplied = {
            probe.hook.references for probe in probes if probe.hook is not None and probe.strategy == fx.STRATEGY_ONE
        }
        assert "Cuenta Digital Popular" in supplied
        # La strategy que nombra el transform del usuario no produce ninguna premisa sufijada: el
        # `-2` de las otras dos sale de `mutate_to_fake`, que sigue disponible.
        assert not any(premise.endswith("-2") for premise in supplied)

    def test_a_key_colliding_with_a_shipped_one_is_refused(self):
        """Either resolution silently changes what an existing catalogue means, so neither is taken."""
        with pytest.raises(ValueError, match="collides"):
            available([_CollidingTransform()])

    def test_two_supplied_keys_colliding_with_each_other_are_refused_too(self):
        with pytest.raises(ValueError, match="collides"):
            available([_PlausibleFake(), _PlausibleFake()])


class TestAnEngineOnlyCoversTheKindsItDeclared:
    """`entity_kinds` was read by catalogue validation and by nothing at generation, so an engine
    produced probes for kinds it had declared it did not handle.

    Of the four shipped engines only the enumeration one passes `kind` through: graph and multi-hop
    ignore the argument entirely. On a catalogue with two entity kinds they hand the same boundary
    to both, so a strategy over one kind builds premises out of the other's entities.
    """

    def _catalogue(self):
        return Catalogue(
            plugins=[PluginSpec(id="p", name="p", description="d", principle=fx.PRINCIPLE_A)],
            strategies=[
                StrategySpec(
                    id=f"s-{kind}",
                    name=kind,
                    description="a premise",
                    plugin="p",
                    entity_kind=kind,
                    transform="keep_real",
                    doc=1,
                    phrasing_hint="hint",
                )
                for kind in ("product", "figure")
            ],
        )

    def _engine(self, kinds):
        return EnumerationProbeEngine(
            StubEntityEnumerator({"product": frozenset({"one"}), "figure": frozenset({"two"})}),
            kinds,
        )

    def test_a_declared_kind_is_the_only_one_it_produces(self):
        documents = [Document(id="d", content="text", structured=True)]

        probes = ProbeLibrary([self._engine(["product"])]).generate(documents, self._catalogue())

        assert {probe.hook.kind for probe in probes} == {"product"}

    def test_declaring_nothing_keeps_the_old_reach(self):
        """Empty is the default, not a claim to handle none: a user who named no kind asked for no
        filtering, and a catalogue that worked before still works."""
        documents = [Document(id="d", content="text", structured=True)]

        probes = ProbeLibrary([self._engine([])]).generate(documents, self._catalogue())

        assert {probe.hook.kind for probe in probes} == {"product", "figure"}

    def test_the_declaration_holds_with_no_knowledge_base_too(self):
        """FR-025 was applied only where documents exist, so the same engine covered different
        strategies depending on whether a corpus happened to be there — this one yielded
        `['s-product']` with documents and `['s-figure', 's-product']` without.

        A hookless probe carries no kind to assert on, so the strategy it came from is what says
        the filter ran. An engine's declaration is a claim about what it is competent for, and
        nothing about that claim depends on a corpus being present to read.
        """
        probes = ProbeLibrary([self._engine(["product"])]).generate([], self._catalogue())

        assert all(probe.hook is None for probe in probes)
        assert sorted(probe.strategy for probe in probes) == ["s-product"]

    def test_declaring_nothing_keeps_the_old_reach_with_no_knowledge_base(self):
        probes = ProbeLibrary([self._engine([])]).generate([], self._catalogue())

        assert sorted(probe.strategy for probe in probes) == ["s-figure", "s-product"]


class TestEngineDeclaration:
    """Which engines ran, and which of them the paper put a number on.

    Both halves are facts the framework holds and the user does not — the composed set is the
    library's own execution, and what the paper measured is gaussia's own paper. Neither is a
    judgement about a domain, so unlike `tau`, the contract or the catalogue, neither is the
    user's to supply.

    It exists because the default composition produced no published number: retrieval, graph and
    multi-hop run by default, and the paper's trade-off tables cover retrieval, graph and
    enumeration. The gap cannot be closed by changing the default — enumeration needs an
    `EntityEnumerator`, which gaussia ships none of by decision D14 — so it is made visible.
    """

    def test_the_default_three_produce_a_set_the_paper_does_not_cover(self):
        library = ProbeLibrary(
            [RetrievalProbeEngine(embedder=_StubEmbedder()), GraphProbeEngine(), MultiHopProbeEngine()]
        )

        declaration = library.declaration

        assert declaration.ran == ["retrieval", "graph", "multi-hop"]
        assert declaration.in_paper_tables == ["retrieval", "graph"]
        assert declaration.outside_paper_tables == ["multi-hop"]

    def test_the_evaluated_set_is_reachable_only_with_an_enumerator(self):
        """Which is what makes it unreachable out of the box rather than merely unset."""
        enumerator = StubEntityEnumerator({"product": frozenset({"one"})})
        library = ProbeLibrary(
            [RetrievalProbeEngine(embedder=_StubEmbedder()), GraphProbeEngine(), EnumerationProbeEngine(enumerator)]
        )

        declaration = library.declaration

        assert declaration.outside_paper_tables == []
        assert declaration.in_paper_tables == declaration.ran

    def test_an_engine_that_produced_no_probe_still_says_it_ran(self):
        """The case the probe set cannot show, and the one FR-025 makes interesting: an engine
        reaching no document is invisible in the output it did not produce."""
        silent = StubProbeEngine(engine_name="retrieval", kinds=frozenset({fx.ENTITY_KIND}), probes=[])

        library = ProbeLibrary([silent])

        assert library.generate([], fx.catalogue(TRANSFORM_KEY)) == []
        assert library.declaration.ran == ["retrieval"]

    def test_the_paper_reference_travels_with_the_claim(self):
        """`in_paper_tables` is a fact about the paper, not about the run, so it can go stale if
        the paper changes — and that is exactly why the version it was read from is recorded."""
        assert ProbeLibrary([GraphProbeEngine()]).declaration.paper == "gaussia-labs/papers#20"
