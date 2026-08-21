"""Catalogue validation: every rejection path, up front (T022).

SC-005 lists six rejections and `data-model.md` gives the condition for each. Five are decidable
from the catalogue and the contract alone; the sixth needs the configured engines, because
`entity_kind` is the user's own vocabulary and gaussia never learns what it means. Without that
last check a plural typo validates cleanly and yields an empty probe set with no error, which is
the failure mode hardest to notice.

FR-026 is the other half: a strategy with no plugin is a control, not a dangling reference.
"""

import pytest
from pydantic import ValidationError

from gaussia.generators.roastme.probes.catalogue import validate_catalogue
from gaussia.generators.roastme.probes.transforms import TRANSFORMS
from gaussia.schemas.roastme import Catalogue, PluginSpec, StrategySpec
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import StubProbeEngine

TRANSFORM_KEY = next(iter(TRANSFORMS))


def _engines(kinds: frozenset[str] = frozenset({fx.ENTITY_KIND})) -> list[StubProbeEngine]:
    return [StubProbeEngine(engine_name="stub-engine", kinds=kinds, probes=fx.probes())]


def _contract():
    return fx.contract(fx.stub_grader())


def _strategy(**overrides) -> StrategySpec:
    fields = {
        "id": fx.STRATEGY_ONE,
        "name": "pattern one",
        "description": "documentation only",
        "plugin": fx.PLUGIN_ONE,
        "entity_kind": fx.ENTITY_KIND,
        "transform": TRANSFORM_KEY,
        "doc": 0,
        "phrasing_hint": "hint",
    }
    fields.update(overrides)
    return StrategySpec(**fields)


def _plugin(**overrides) -> PluginSpec:
    fields = {
        "id": fx.PLUGIN_ONE,
        "name": "family one",
        "description": "documentation only",
        "principle": fx.PRINCIPLE_A,
    }
    fields.update(overrides)
    return PluginSpec(**fields)


class TestTheTransformSetIsClosed:
    def test_exactly_four_transformations_are_registered(self):
        """FR-025 closes the set at four: keep the entity real, mutate it into a fake one, flip a
        documented value, flip a documented fact. A fifth needs the requirement relaxed too."""
        assert len(TRANSFORMS) == 4

    def test_each_registered_transform_answers_to_its_own_key(self):
        for key, transform in TRANSFORMS.items():
            assert transform.key == key


class TestAValidCatalogue:
    def test_is_accepted(self):
        validate_catalogue(fx.catalogue(TRANSFORM_KEY), _contract(), _engines())

    def test_a_strategy_with_no_plugin_is_a_control_not_a_dangling_reference(self):
        """FR-026: this is the only mechanism by which a control is recognised."""
        catalogue = Catalogue(plugins=[_plugin()], strategies=[_strategy(plugin=None)])
        validate_catalogue(catalogue, _contract(), _engines())


class TestRejections:
    def test_a_dangling_principle(self):
        catalogue = Catalogue(plugins=[_plugin(principle="absent_from_the_contract")], strategies=[_strategy()])

        with pytest.raises(ValueError, match=r"(?i)principle"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_a_dangling_plugin(self):
        catalogue = Catalogue(plugins=[_plugin()], strategies=[_strategy(plugin="no-such-plugin")])

        with pytest.raises(ValueError, match=r"(?i)plugin"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_an_unknown_transform(self):
        catalogue = Catalogue(plugins=[_plugin()], strategies=[_strategy(transform="not-a-registered-transform")])

        with pytest.raises(ValueError, match=r"(?i)transform"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_a_duplicate_plugin_identifier(self):
        catalogue = Catalogue(plugins=[_plugin(), _plugin()], strategies=[_strategy()])

        with pytest.raises(ValueError, match=r"(?i)duplicate|unique"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_a_duplicate_strategy_identifier(self):
        catalogue = Catalogue(plugins=[_plugin()], strategies=[_strategy(), _strategy()])

        with pytest.raises(ValueError, match=r"(?i)duplicate|unique"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_an_entity_kind_no_configured_engine_handles(self):
        """The one rejection whose condition depends on the engines rather than the catalogue."""
        catalogue = Catalogue(plugins=[_plugin()], strategies=[_strategy(entity_kind="ek-nobody-handles")])

        with pytest.raises(ValueError, match=r"(?i)entity"):
            validate_catalogue(catalogue, _contract(), _engines())

    def test_a_grounding_label_outside_the_two_permitted_values(self):
        """`doc` is constrained on the field, so the rejection is structural and comes even
        earlier than catalogue validation — the label can never reach a catalogue at all."""
        with pytest.raises(ValidationError, match=r"(?i)doc|less than or equal"):
            _strategy(doc=2)

        with pytest.raises(ValidationError, match=r"(?i)doc|greater than or equal"):
            _strategy(doc=-1)


class TestNothingDependsOnAnIdentifier:
    def test_validation_holds_when_every_identifier_is_renamed(self):
        """SC-003: the catalogue's identifiers are entirely the user's own, so no library
        behaviour may depend on any of them."""
        renamed_plugin = "pl-renamed-e2c8"
        catalogue = Catalogue(
            plugins=[_plugin(id=renamed_plugin)],
            strategies=[_strategy(id="st-renamed-19b4", plugin=renamed_plugin)],
        )
        validate_catalogue(catalogue, _contract(), _engines())
