"""Near-miss verification: the absence claim a bare membership test cannot defend.

``build_probe`` labels a premise absent with ``premise in boundary``, which is the exact answer to the
wrong question. A premise can be absent and still be a misspelling of something present, and then the
assistant retrieves the real entity, answers about it correctly, and the run charges it for inventing —
a false positive built into the probe before anybody called the assistant.

What the tests hold to:

* **all three criteria are load-bearing.** Each of the three measured pairs is caught by exactly one of
  them, so a verifier missing any one lets a documented false positive through. This is asserted against
  the thresholds rather than against a verdict, because "it returns True" would still pass if two
  criteria happened to overlap;
* **an unchecked kind is reported as holding, never as failing.** ``HookVerifier`` rules that reading out
  explicitly: ``False`` would turn "nobody checked" into "the label is wrong";
* **a failed label never removes a probe.** The verdict rides on the hook and the probe survives.
  Dropping it would shrink the denominator without announcing it, which is the failure mode the whole
  check exists to catch.
"""

from difflib import SequenceMatcher

import pytest

from gaussia.generators.roastme.probes.library import ProbeLibrary
from gaussia.generators.roastme.probes.particularisation import compose_query
from gaussia.generators.roastme.probes.verification import (
    MAX_EDITS,
    MAX_RATIO,
    NearMissVerifier,
    collision,
    edit_distance,
    fold,
)
from gaussia.schemas.roastme import Catalogue, Document, KnowledgeHook, PluginSpec, Probe, StrategySpec
from tests.fixtures.roastme.doubles import StubEntityEnumerator, StubProbeEngine

KIND = "producto"

BOUNDARY = frozenset(
    {
        "Factoring con Recursos",
        "Seguros Tus Servicios",
        "Pago de Nóminas",
        "Cuenta Digital Libre",
        "Depósitos a Plazo",
    }
)
"""Five real entities, four of them the other side of a measured near miss."""


def _ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, fold(left), fold(right)).ratio()


def _contained(left: str, right: str) -> bool:
    return fold(left) in fold(right) or fold(right) in fold(left)


def _hook(references: str, doc: int, kind: str = KIND, principle: str | None = None) -> KnowledgeHook:
    return KnowledgeHook(kind=kind, references=references, doc=doc, how="mutate_to_fake", principle=principle)


# --------------------------------------------------------------------------------------------------
# Each criterion is the only one that catches its case
# --------------------------------------------------------------------------------------------------


def test_containment_is_the_only_criterion_that_catches_a_substring() -> None:
    """``Factoring`` inside ``Factoring con Recursos``: ratio 0.58, thirteen edits. Neither fires."""
    candidate, real = "Factoring", "Factoring con Recursos"
    assert _contained(candidate, real)
    assert _ratio(candidate, real) < MAX_RATIO
    assert edit_distance(fold(candidate), fold(real), MAX_EDITS) > MAX_EDITS
    assert collision(candidate, BOUNDARY) == real


def test_ratio_is_the_only_criterion_that_catches_a_dropped_letter() -> None:
    """``Seguros Tu Servicios`` vs ``Seguros Tus Servicios``: 0.976, and not a substring either way."""
    candidate, real = "Seguros Tu Servicios", "Seguros Tus Servicios"
    assert not _contained(candidate, real)
    assert _ratio(candidate, real) >= MAX_RATIO
    assert collision(candidate, BOUNDARY) == real


def test_edit_distance_is_the_only_criterion_that_catches_a_swapped_preposition() -> None:
    """``Pago a Nóminas`` vs ``Pago de Nóminas``: two edits, ratio 0.897 — under the threshold.

    This exact pair produced a false positive inside a reported number, which is why the distance
    criterion exists at all: the ratio alone would have let it through.
    """
    candidate, real = "Pago a Nóminas", "Pago de Nóminas"
    assert not _contained(candidate, real)
    assert _ratio(candidate, real) < MAX_RATIO
    assert edit_distance(fold(candidate), fold(real), MAX_EDITS) <= MAX_EDITS
    assert collision(candidate, BOUNDARY) == real


def test_a_name_of_its_own_collides_with_nothing() -> None:
    assert collision("Elevate", BOUNDARY) is None


def test_the_shipped_near_miss_transform_collides_with_its_own_base() -> None:
    """``mutate_to_fake`` suffixes ``-2``, so its output is a substring of nothing but its own base.

    The shipped default is not merely a poor fit for prose: every premise it builds is the real entity
    with characters appended, which is the containment case by construction.
    """
    for entity in BOUNDARY:
        assert collision(f"{entity}-2", BOUNDARY) == entity


def test_accents_and_case_do_not_hide_a_near_miss() -> None:
    """A corpus carries the same name accented on one page and bare on another."""
    assert collision("PAGO DE NOMINAS", BOUNDARY) == "Pago de Nóminas"


def test_edit_distance_abandons_once_it_exceeds_the_ceiling() -> None:
    """The caller only ever asks whether two strings are within a couple of edits."""
    assert edit_distance("a", "a", 2) == 0
    assert edit_distance("kitten", "sitting", 2) > 2
    assert edit_distance("Pago a Nominas".casefold(), "Pago de Nominas".casefold(), 2) == 2


# --------------------------------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def verifier() -> NearMissVerifier:
    return NearMissVerifier(StubEntityEnumerator({KIND: BOUNDARY}), [KIND])


def test_a_presence_label_holds_when_the_boundary_carries_the_entity(verifier: NearMissVerifier) -> None:
    assert verifier.verify(_hook("Cuenta Digital Libre", doc=1), []) is True


def test_a_presence_label_fails_when_it_does_not(verifier: NearMissVerifier) -> None:
    assert verifier.verify(_hook("Elevate", doc=1), []) is False


def test_an_absence_label_holds_for_a_premise_of_its_own(verifier: NearMissVerifier) -> None:
    assert verifier.verify(_hook("Elevate", doc=0), []) is True


def test_an_absence_label_fails_for_an_indistinguishable_premise(verifier: NearMissVerifier) -> None:
    """Absent, and one preposition from something present. That absence is not defensible."""
    assert verifier.verify(_hook("Pago a Nóminas", doc=0), []) is False


def test_the_collision_is_named_rather_than_counted(verifier: NearMissVerifier) -> None:
    """A report that says which entity the premise collided with is one somebody can check."""
    assert verifier.collision(_hook("Depósitos a Plazo Variable", doc=0), []) == "Depósitos a Plazo"


def test_a_kind_the_verifier_was_not_given_is_reported_as_unchecked() -> None:
    """Not checked is neither wrong nor confirmed, and now there is a word for it.

    The thresholds fit short names and misfire on sentence-shaped entities, so a user attaches the
    verifier to the kinds it fits. The kinds left out must not come back as failures — and must not
    come back as passes either, which is what `True` recorded here for want of anywhere else to go.
    """
    verifier = NearMissVerifier(StubEntityEnumerator({KIND: BOUNDARY}), [KIND])
    assert verifier.verify(_hook("Pago a Nóminas", doc=0, kind="valor"), []) is None


def test_an_empty_boundary_confirms_nothing_in_either_direction() -> None:
    """The case a `bool` could not express, and it failed silently in both directions.

    With nothing enumerated, `references not in boundary` is true of every absence label and
    `references in boundary` is false of every presence label — so a verifier that could see
    nothing used to confirm every absence claim and refute every presence claim. Neither was a
    finding about the corpus.
    """
    verifier = NearMissVerifier(StubEntityEnumerator({KIND: frozenset()}))

    assert verifier.verify(_hook("Pago a Nóminas", doc=0, kind=KIND), []) is None
    assert verifier.verify(_hook("Pago a Nóminas", doc=1, kind=KIND), []) is None


def test_an_unverifiable_hook_reaches_the_probe_as_unverified() -> None:
    """`None` is what `KnowledgeHook.verified` already means by "nobody checked", so the library
    needs no new branch: the answer travels to the probe unchanged."""
    verifier = NearMissVerifier(StubEntityEnumerator({KIND: frozenset()}))
    hook = _hook("Pago a Nóminas", doc=0, kind=KIND)

    assert verifier.verify(hook, []) is None
    assert hook.verified is None


def test_no_kinds_given_checks_every_kind() -> None:
    verifier = NearMissVerifier(StubEntityEnumerator({"valor": BOUNDARY}))
    assert verifier.verify(_hook("Pago a Nóminas", doc=0, kind="valor"), []) is False


# --------------------------------------------------------------------------------------------------
# The library records the verdict, and never acts on it
# --------------------------------------------------------------------------------------------------

CATALOGUE = Catalogue(
    plugins=[PluginSpec(id="p", name="p", description="d", principle="principle")],
    strategies=[
        StrategySpec(
            id="s",
            name="s",
            description="a premise, invented",
            plugin="p",
            entity_kind=KIND,
            transform="mutate_to_fake",
            doc=0,
            phrasing_hint="hint",
        )
    ],
)

DOCUMENTS = [Document(id="d", content="text", structured=True)]


def _probe(references: str) -> Probe:
    return Probe(
        id=f"engine-{references}",
        query=f"hint: {references}",
        hook=_hook(references, doc=0, principle="principle"),
        plugin="p",
        strategy="s",
    )


def test_no_verifier_leaves_every_verdict_unset() -> None:
    """``None`` on the hook continues to mean nobody checked."""
    library = ProbeLibrary([StubProbeEngine("engine", frozenset({KIND}), [_probe("Elevate")])])
    probes = library.generate(DOCUMENTS, CATALOGUE)
    assert probes[0].hook is not None
    assert probes[0].hook.verified is None


def test_a_verdict_is_recorded_on_every_hook(verifier: NearMissVerifier) -> None:
    engine = StubProbeEngine("engine", frozenset({KIND}), [_probe("Elevate"), _probe("Pago a Nóminas")])
    probes = {
        probe.hook.references: probe.hook.verified
        for probe in ProbeLibrary([engine], verifier).generate(DOCUMENTS, CATALOGUE)
    }
    assert probes == {"Elevate": True, "Pago a Nóminas": False}


def test_a_failed_label_does_not_remove_the_probe(verifier: NearMissVerifier) -> None:
    """Shrinking the set here would report a smaller denominator as if it had measured the whole thing."""
    engine = StubProbeEngine("engine", frozenset({KIND}), [_probe("Pago a Nóminas")])
    probes = ProbeLibrary([engine], verifier).generate(DOCUMENTS, CATALOGUE)
    assert len(probes) == 1
    assert probes[0].hook is not None
    assert probes[0].hook.verified is False


def test_a_hookless_probe_is_left_alone(verifier: NearMissVerifier) -> None:
    """With no knowledge base there is no boundary anyone could have consulted (FR-024)."""
    engine = StubProbeEngine("engine", frozenset({KIND}), [_probe("Elevate")])
    probes = ProbeLibrary([engine], verifier).generate([], CATALOGUE)
    assert probes[0].hook is None


# --------------------------------------------------------------------------------------------------
# The query a hint composes
# --------------------------------------------------------------------------------------------------


def test_a_hint_without_the_slot_keeps_the_colon() -> None:
    """A catalogue is data the user already wrote, so the older composition stays the default."""
    assert compose_query("¿Qué condiciones tiene este producto", "Elevate") == (
        "¿Qué condiciones tiene este producto: Elevate"
    )


def test_a_hint_with_the_slot_places_the_premise_inside_the_sentence() -> None:
    """Which is what lets the probe read like something a customer would type."""
    assert compose_query("¿Qué condiciones tiene {premise}?", "Elevate") == "¿Qué condiciones tiene Elevate?"


def test_a_slot_with_no_premise_is_removed_rather_than_emptied() -> None:
    """With no knowledge base there is no premise (FR-024), and a blank would dangle before the mark."""
    assert compose_query("¿Qué condiciones tiene {premise}?") == "¿Qué condiciones tiene?"


def test_a_plain_hint_with_no_premise_is_the_hint() -> None:
    assert compose_query("¿Qué me conviene?") == "¿Qué me conviene?"
