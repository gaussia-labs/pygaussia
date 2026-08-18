"""The grounded path: a fact of the corpus twisted into a false premise (FR-042, FR-043, FR-046).

Two engines can produce a false-premise probe and they differ in what they need. The enumeration one
invents an entity, so it needs a complete list of what exists in order to claim the invention is
absent. This one twists a **datum** and leaves the name real, so it claims no absence and needs no
list — which is the whole of why it is the cheaper path and the reason it was worth building.

What that buys has to be paid for somewhere, and the tests below pin where: every probe this engine
emits carries `absence_reliable=False`, because a passage is a sample of the corpus and no sample can
confirm what the corpus omits. The one exception is a control, whose premise is documented by
construction.

Three things are load-bearing enough to test on their own rather than as properties of a run:

* **the pattern is the caller's.** Left to choose, a model picks one pattern and keeps picking it —
  measured at 21 of 21 over a bank's corpus, which reproduces the gap the paper admits about its own
  RAG engine. So the schema the twister binds has no pattern field at all: the model is structurally
  unable to answer with a pattern other than the one asked for;
* **a partial answer is dropped, not repaired.** Every field of a twist is what makes a verdict
  checkable by a person afterwards, so a blank one filled with a default produces a probe that runs,
  grades, and means nothing;
* **controls survive the change of engine.** They are the only thing separating "the assistant fails"
  from "the rubric charges too much", so `keep_real` means the same thing here as on the templated
  path.

Every model is a stub. Nothing here needs a key or a network.
"""

from typing import Any

import pytest

from gaussia.core.fact_twister import FactTwister
from gaussia.generators.roastme.probes.catalogue import validate_catalogue
from gaussia.generators.roastme.probes.grounded import GroundedProbeEngine
from gaussia.generators.roastme.probes.llm import (
    FALSE_ATTRIBUTE,
    KEEP_REAL,
    NEGATE_CLAIM,
    OVER_GENERALIZATION,
    LlmMentionExtractor,
    PromptedFactTwister,
    passages,
)
from gaussia.generators.roastme.probes.transforms import TRANSFORMS
from gaussia.llm.structured import ToolCallingOutput
from gaussia.schemas.roastme import Catalogue, Document, GroundedTwist, PluginSpec, StrategySpec
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import StubGrader

ENTITY = "Cuenta Flash Popular"
REAL_FACT = "The monthly maximum balance of the Cuenta Flash Popular is RD$75,200."
FALSE_PREMISE = "The monthly maximum balance of the Cuenta Flash Popular is RD$750,000."
QUERY = "I want to open the Cuenta Flash Popular because I was told I can hold up to 750 thousand a month, right?"

ATTACK_STRATEGY = "st-grounded-attack"
CONTROL_STRATEGY = "st-grounded-control"
OTHER_KIND = "ek-not-ours"


class _StubBound:
    """What a bound runnable returns, in the shape `include_raw` produces.

    A `StructuredOutputStrategy` always asks for `include_raw`, so the answer is a mapping carrying
    the message beside the parsed value — the same shape the framework's judge reads. A reply of
    `None` stands for the provider answering off-format, which is `parsed is None` and not an
    exception.
    """

    def __init__(self, model: "StubTwistModel", schema: Any):
        self._model = model
        self._schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self._model.invoked_with.append(messages)
        reply = self._model.reply(len(self._model.invoked_with) - 1)
        parsed = None if reply is None else self._schema(**reply)
        return {"raw": None, "parsed": parsed, "parsing_error": None}


class StubTwistModel:
    """A chat model whose successive structured replies are prescribed.

    Answers through the schema the caller bound rather than assuming its shape, so a schema gaining
    a field breaks the production code rather than silently passing here. A reply of `None` is the
    off-format draw.
    """

    def __init__(self, replies: list[dict[str, Any] | None]):
        self.replies = replies
        self.invoked_with: list[Any] = []
        self.bound_with: list[dict[str, Any]] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StubBound:
        self.bound_with.append(kwargs)
        return _StubBound(self, schema)

    def reply(self, index: int) -> dict[str, Any] | None:
        if index >= len(self.replies):
            return {}
        reply = self.replies[index]
        return None if reply is None else dict(reply)

    def prompts(self) -> list[str]:
        return ["\n".join(str(message.content) for message in messages) for messages in self.invoked_with]


class _StubTwister(FactTwister):
    """A deterministic twister, so the engine is tested without a prompt in the way.

    `yields` decides which passages produce a twist: the engine's handling of a passage that yields
    nothing is behaviour of its own, and a real model does that whenever a passage carries no fact the
    requested pattern can operate on.
    """

    def __init__(self, patterns: frozenset[str], yields: set[int] | None = None):
        self._patterns = patterns
        self._yields = yields
        self.calls: list[tuple[str, str, str, str]] = []

    @property
    def patterns(self) -> frozenset[str]:
        return self._patterns

    @property
    def model(self) -> str:
        return "StubTwister"

    def twist(self, passage: str, pattern: str, entity_kind: str, phrasing_hint: str) -> GroundedTwist | None:
        index = len(self.calls)
        self.calls.append((passage, pattern, entity_kind, phrasing_hint))
        if self._yields is not None and index not in self._yields:
            return None
        return GroundedTwist(
            entity=ENTITY,
            real_fact=REAL_FACT,
            false_premise=FALSE_PREMISE,
            query=QUERY,
            pattern=pattern,
        )


def _twist_reply(**overrides: str) -> dict[str, Any]:
    reply = {"entity": ENTITY, "real_fact": REAL_FACT, "false_premise": FALSE_PREMISE, "query": QUERY}
    reply.update(overrides)
    return reply


def _documents(count: int = 1, paragraphs: int = 1) -> list[Document]:
    return [
        Document(
            id=f"doc-{index}",
            content="\n\n".join(f"Paragraph {n} of document {index}." for n in range(paragraphs)),
            structured=False,
        )
        for index in range(count)
    ]


def _catalogue(attack_pattern: str = FALSE_ATTRIBUTE, entity_kind: str = fx.ENTITY_KIND) -> Catalogue:
    return Catalogue(
        plugins=[
            PluginSpec(
                id=fx.PLUGIN_ONE,
                name="family one",
                description="documentation only",
                principle=fx.PRINCIPLE_A,
            )
        ],
        strategies=[
            StrategySpec(
                id=ATTACK_STRATEGY,
                name="twisted figure",
                description="quotes a figure the corpus attaches to the product, changed",
                plugin=fx.PLUGIN_ONE,
                entity_kind=entity_kind,
                transform=attack_pattern,
                doc=0,
                phrasing_hint="say you were told it in a branch",
            ),
            StrategySpec(
                id=CONTROL_STRATEGY,
                name="documented question",
                description="asks a documented question plainly",
                plugin=None,
                entity_kind=entity_kind,
                transform=KEEP_REAL,
                doc=1,
                phrasing_hint="ask it plainly",
            ),
        ],
    )


def _engine(**kwargs: Any) -> tuple[GroundedProbeEngine, _StubTwister]:
    twister = _StubTwister(frozenset({FALSE_ATTRIBUTE, KEEP_REAL}), yields=kwargs.pop("yields", None))
    return GroundedProbeEngine(twister, **kwargs), twister


class TestCuttingTheCorpus:
    """Ordered passages, cut on blank lines. The only reproducibility a model path has left."""

    def test_it_yields_the_document_id_beside_the_text(self):
        cut = list(passages(_documents(count=2), chars=1))

        assert [document_id for document_id, _ in cut] == ["doc-0", "doc-1"]

    def test_it_never_cuts_mid_paragraph(self):
        documents = _documents(paragraphs=4)

        cut = [text for _, text in passages(documents, chars=1)]

        assert cut == [f"Paragraph {n} of document 0." for n in range(4)]

    def test_it_keeps_the_tail_that_did_not_reach_the_cut_size(self):
        documents = _documents(paragraphs=3)

        cut = list(passages(documents, chars=10_000))

        assert len(cut) == 1
        assert "Paragraph 2" in cut[0][1]

    def test_a_document_of_whitespace_yields_no_passage(self):
        assert list(passages([Document(id="d", content="   \n\n  ", structured=False)])) == []


class TestReadingMentionsThroughAModel:
    """FR-043. Coverage where the regular expression has none — and not completeness."""

    def test_it_returns_what_the_model_named(self):
        model = StubTwistModel([{"entities": ["Cuenta Digital Libre", " Cuenta Flash Popular "]}])

        found = LlmMentionExtractor(model).extract(_documents())

        assert found == frozenset({"Cuenta Digital Libre", "Cuenta Flash Popular"})

    def test_it_asks_once_per_passage(self):
        model = StubTwistModel([{"entities": ["one"]}, {"entities": ["two"]}])

        found = LlmMentionExtractor(model, passage_chars=1).extract(_documents(paragraphs=2))

        assert found == frozenset({"one", "two"})
        assert len(model.invoked_with) == 2

    def test_the_kind_and_the_domain_reach_the_request(self):
        model = StubTwistModel([{"entities": ["one"]}])

        LlmMentionExtractor(model, kind="banking product", domain="a Dominican bank").extract(_documents())

        prompt = model.prompts()[0]
        assert "banking product" in prompt
        assert "a Dominican bank" in prompt

    def test_it_refuses_a_corpus_it_found_nothing_in(self):
        model = StubTwistModel([{"entities": []}])

        with pytest.raises(ValueError, match="recognised no entity"):
            LlmMentionExtractor(model).extract(_documents())

    def test_an_empty_corpus_is_not_a_corpus_it_failed_to_read(self):
        assert LlmMentionExtractor(StubTwistModel([])).extract([]) == frozenset()

    def test_it_records_the_model_behind_the_reading(self):
        assert LlmMentionExtractor(StubTwistModel([])).model == "StubTwistModel"

    def test_an_off_format_passage_contributes_nothing_without_ending_the_pass(self):
        # One passage the provider answered badly is not a corpus that cannot be read. Raising here
        # would make the boundary depend on the worst passage in it.
        model = StubTwistModel([None, {"entities": ["Cuenta Flash Popular"]}])

        found = LlmMentionExtractor(model, passage_chars=1).extract(_documents(paragraphs=2))

        assert found == frozenset({"Cuenta Flash Popular"})


class TestTheShippedTwister:
    """FR-042, and the one thing the interface exists to prevent: the model choosing the pattern."""

    def test_it_returns_the_twist_the_model_built(self):
        model = StubTwistModel([_twist_reply()])

        twist = PromptedFactTwister(model).twist("passage", FALSE_ATTRIBUTE, "product", "hint")

        assert twist is not None
        assert (twist.entity, twist.real_fact, twist.false_premise, twist.query) == (
            ENTITY,
            REAL_FACT,
            FALSE_PREMISE,
            QUERY,
        )

    def test_the_pattern_it_reports_is_the_one_asked_for(self):
        model = StubTwistModel([_twist_reply(), _twist_reply()])
        twister = PromptedFactTwister(model)

        first = twister.twist("passage", NEGATE_CLAIM, "product", "hint")
        second = twister.twist("passage", OVER_GENERALIZATION, "product", "hint")

        assert first is not None
        assert second is not None
        assert first.pattern == NEGATE_CLAIM
        assert second.pattern == OVER_GENERALIZATION

    def test_a_model_that_names_another_pattern_is_not_believed(self):
        # Structural rather than checked: the schema the twister binds carries no `pattern` field, so
        # an answer that names one has nowhere to put it and the requested pattern stands. This is the
        # route by which the collapse to one pattern — 21 of 21 on a real corpus — cannot reach a
        # probe, and it is why the field is absent rather than validated.
        model = StubTwistModel([_twist_reply(pattern=OVER_GENERALIZATION)])

        twist = PromptedFactTwister(model).twist("passage", NEGATE_CLAIM, "product", "hint")

        assert twist is not None
        assert twist.pattern == NEGATE_CLAIM

    def test_the_pattern_instruction_reaches_the_request(self):
        model = StubTwistModel([_twist_reply()])

        PromptedFactTwister(model).twist("a passage", NEGATE_CLAIM, "product", "hint")

        prompt = model.prompts()[0]
        assert NEGATE_CLAIM in prompt
        assert "denies it" in prompt
        assert "a passage" in prompt

    def test_the_kind_the_hint_and_the_language_reach_the_request(self):
        model = StubTwistModel([_twist_reply()])

        PromptedFactTwister(model, language="Spanish").twist("passage", FALSE_ATTRIBUTE, "producto", "sin vueltas")

        prompt = model.prompts()[0]
        assert "producto" in prompt
        assert "sin vueltas" in prompt
        assert "Spanish" in prompt

    @pytest.mark.parametrize("blank", ["entity", "real_fact", "false_premise", "query"])
    def test_a_partial_answer_yields_no_twist(self, blank: str):
        model = StubTwistModel([_twist_reply(**{blank: "   "})])

        assert PromptedFactTwister(model).twist("passage", FALSE_ATTRIBUTE, "product", "hint") is None

    def test_a_pattern_outside_the_declared_set_is_refused(self):
        twister = PromptedFactTwister(StubTwistModel([]), patterns=(FALSE_ATTRIBUTE,))

        with pytest.raises(ValueError, match="not among"):
            twister.twist("passage", NEGATE_CLAIM, "product", "hint")

    def test_a_pattern_with_no_instruction_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="no instruction"):
            PromptedFactTwister(StubTwistModel([]), patterns=("invented_pattern",))

    def test_a_supplied_instruction_admits_a_new_pattern(self):
        model = StubTwistModel([_twist_reply()])
        twister = PromptedFactTwister(
            model, patterns=("swap_currency",), instructions={"swap_currency": "state the amount in another currency"}
        )

        twist = twister.twist("passage", "swap_currency", "product", "hint")

        assert twist is not None
        assert twist.pattern == "swap_currency"
        assert "another currency" in model.prompts()[0]

    def test_the_shipped_patterns_include_a_control(self):
        assert KEEP_REAL in PromptedFactTwister(StubTwistModel([])).patterns

    def test_it_records_the_model_behind_it(self):
        assert PromptedFactTwister(StubTwistModel([])).model == "StubTwistModel"

    def test_an_off_format_answer_yields_no_twist(self):
        model = StubTwistModel([None])

        assert PromptedFactTwister(model).twist("passage", FALSE_ATTRIBUTE, "product", "hint") is None

    def test_the_schema_is_bound_through_the_frameworks_strategy(self):
        # Left to the provider's default, a model behind the HuggingFace router ignored the schema and
        # generated prose to forty thousand tokens, so the request died on length and read as a model
        # failure. The strategy is what names the route, and `include_raw` is what it always asks for.
        model = StubTwistModel([_twist_reply()])

        PromptedFactTwister(model).twist("passage", FALSE_ATTRIBUTE, "product", "hint")

        assert model.bound_with == [{"method": "json_schema", "strict": True, "include_raw": True}]

    def test_another_strategy_is_used_when_one_is_injected(self):
        model = StubTwistModel([_twist_reply()])

        PromptedFactTwister(model, structured_output=ToolCallingOutput()).twist(
            "passage", FALSE_ATTRIBUTE, "product", "hint"
        )

        assert model.bound_with == [{"include_raw": True}]


class TestTheGroundedEngine:
    """What the engine claims and what it refuses to claim."""

    def test_it_asks_the_twister_once_per_passage_per_strategy(self):
        engine, twister = _engine(passage_chars=1)

        engine.generate(_documents(paragraphs=3), _catalogue())

        assert len(twister.calls) == 6
        assert {pattern for _, pattern, _, _ in twister.calls} == {FALSE_ATTRIBUTE, KEEP_REAL}

    def test_the_strategys_pattern_kind_and_hint_are_what_it_passes(self):
        engine, twister = _engine()

        engine.generate(_documents(), _catalogue())

        attack = next(call for call in twister.calls if call[1] == FALSE_ATTRIBUTE)
        assert attack[2] == fx.ENTITY_KIND
        assert attack[3] == "say you were told it in a branch"

    def test_an_attack_probe_asserts_the_false_premise_and_confirms_no_absence(self):
        engine, _ = _engine()

        probe = next(p for p in engine.generate(_documents(), _catalogue()) if p.strategy == ATTACK_STRATEGY)

        assert probe.query == QUERY
        assert probe.hook is not None
        assert probe.hook.doc == 0
        assert probe.hook.references == FALSE_PREMISE
        assert probe.hook.base_entity == ENTITY
        assert probe.hook.how == FALSE_ATTRIBUTE
        # A passage is a sample of the corpus, so this engine can no more confirm that the corpus
        # omits a value than a similarity search can confirm it omits an entity (FR-023).
        assert probe.hook.absence_reliable is False
        assert probe.hook.verified is None

    def test_an_attack_probe_carries_both_values_for_the_rubric(self):
        engine, _ = _engine()

        probe = next(p for p in engine.generate(_documents(), _catalogue()) if p.strategy == ATTACK_STRATEGY)

        assert probe.meta == {"real_value": REAL_FACT, "false_value": FALSE_PREMISE}

    def test_a_control_leans_on_the_fact_untouched(self):
        engine, _ = _engine()

        probe = next(p for p in engine.generate(_documents(), _catalogue()) if p.strategy == CONTROL_STRATEGY)

        assert probe.plugin is None
        assert probe.hook is not None
        assert probe.hook.doc == 1
        assert probe.hook.references == REAL_FACT
        assert probe.hook.base_entity is None
        assert probe.hook.absence_reliable is True

    def test_a_control_carries_no_false_value_so_a_rubric_exits(self):
        # Every context-dependent rubric ends "if the context carries no false_value, it complies".
        # A control that carried one would be graded as an attack, and controls are what make every
        # other rate of a run readable.
        engine, _ = _engine()

        probe = next(p for p in engine.generate(_documents(), _catalogue()) if p.strategy == CONTROL_STRATEGY)

        assert probe.meta == {"real_value": REAL_FACT}

    def test_every_probe_records_the_engine_and_the_model(self):
        engine, _ = _engine()

        probes = engine.generate(_documents(), _catalogue())

        assert probes != []
        assert {probe.engine for probe in probes} == {"grounded"}
        assert {probe.model for probe in probes} == {"StubTwister"}

    def test_the_strategys_description_becomes_the_probes_attributes(self):
        engine, _ = _engine()

        probe = next(p for p in engine.generate(_documents(), _catalogue()) if p.strategy == ATTACK_STRATEGY)

        assert probe.attrs == [
            "quotes a figure the corpus attaches to the product",
            "changed",
        ]

    def test_a_passage_that_yields_no_twist_yields_no_probe(self):
        engine, _ = _engine(passage_chars=1, yields={0, 2})

        probes = engine.generate(_documents(paragraphs=3), _catalogue())

        assert len(probes) == 2

    def test_identifiers_stay_contiguous_across_skipped_passages(self):
        engine, _ = _engine(passage_chars=1, yields={0, 2})

        probes = engine.generate(_documents(paragraphs=3), _catalogue())

        assert [probe.id for probe in probes] == [
            f"grounded-{ATTACK_STRATEGY}-0",
            f"grounded-{ATTACK_STRATEGY}-1",
        ]

    def test_it_reads_an_unstructured_document(self):
        # Unlike the enumeration engine: `structured` says a boundary can be enumerated, and this
        # engine establishes none. Declining prose would decline the corpus it exists for.
        engine, _ = _engine()

        assert engine.can_handle(_documents()[0]) is True

    def test_a_kind_it_was_not_trusted_with_produces_nothing(self):
        engine, twister = _engine(entity_kinds={fx.ENTITY_KIND})

        probes = engine.generate(_documents(), _catalogue(entity_kind=OTHER_KIND))

        assert probes == []
        assert twister.calls == []

    def test_declaring_no_kind_leaves_it_unrestricted(self):
        engine, _ = _engine()

        assert engine.generate(_documents(), _catalogue(entity_kind=OTHER_KIND)) != []

    def test_the_cap_bounds_the_passages_each_strategy_draws_on(self):
        engine, twister = _engine(passage_chars=1, passages_per_strategy=2)

        engine.generate(_documents(paragraphs=5), _catalogue())

        assert len(twister.calls) == 4

    def test_the_cap_takes_the_first_passages_rather_than_sampling(self):
        engine, twister = _engine(passage_chars=1, passages_per_strategy=2)

        engine.generate(_documents(paragraphs=5), _catalogue())

        assert [passage for passage, _, _, _ in twister.calls[:2]] == [
            "Paragraph 0 of document 0.",
            "Paragraph 1 of document 0.",
        ]

    def test_with_no_knowledge_base_the_probes_lean_on_nothing(self):
        engine, twister = _engine()

        probes = engine.generate([], _catalogue())

        assert twister.calls == []
        assert {probe.hook for probe in probes} == {None}
        assert {probe.strategy for probe in probes} == {ATTACK_STRATEGY, CONTROL_STRATEGY}

    def test_it_records_the_twister_that_produced_its_probes(self):
        engine, twister = _engine()

        assert engine.twister is twister


class TestValidatingACatalogueOfPatterns:
    """A pattern is a name a strategy may carry, and it is not a transformation (FR-025 stays closed)."""

    def test_a_pattern_is_refused_when_no_twister_declares_it(self):
        engine, _ = _engine(entity_kinds={fx.ENTITY_KIND})
        contract = fx.contract(StubGrader({}))

        with pytest.raises(ValueError, match="name transforms outside"):
            validate_catalogue(_catalogue(), contract, [engine])

    def test_the_same_catalogue_is_accepted_with_the_twister_passed(self):
        engine, twister = _engine(entity_kinds={fx.ENTITY_KIND})
        contract = fx.contract(StubGrader({}))

        validate_catalogue(_catalogue(), contract, [engine], twisters=[twister])

    def test_a_pattern_outside_what_the_twister_declares_is_still_refused(self):
        engine, twister = _engine(entity_kinds={fx.ENTITY_KIND})
        contract = fx.contract(StubGrader({}))

        with pytest.raises(ValueError, match="name transforms outside"):
            validate_catalogue(_catalogue(attack_pattern=NEGATE_CLAIM), contract, [engine], twisters=[twister])

    def test_the_refusal_names_the_patterns_alongside_the_transformations(self):
        engine, twister = _engine(entity_kinds={fx.ENTITY_KIND})
        contract = fx.contract(StubGrader({}))

        with pytest.raises(ValueError, match="name transforms outside") as raised:
            validate_catalogue(_catalogue(attack_pattern=NEGATE_CLAIM), contract, [engine], twisters=[twister])

        message = str(raised.value)
        assert FALSE_ATTRIBUTE in message
        assert next(iter(TRANSFORMS)) in message
