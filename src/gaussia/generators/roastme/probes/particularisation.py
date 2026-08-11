"""Particularisation: what every shipped probe engine does the same way.

FR-020 makes this stage the only one with knowledge-base access, and FR-021 makes the ``doc``
label the generating engine's own call. The surest way to keep that second promise identical
across four engines is for exactly one piece of code to make it, so the flow lives here and only
the step that actually differs is left to the engine: **which entities it can see for an entity
kind**, and **whether absence from that view is absence from the base**. Everything else — the
premise, the hook, the control marker, the identifiers — follows from those two.

Membership is what decides the label: a premise the engine can see is documented, a premise it
cannot is not. That is also why the same mechanism carries FR-023 for free — the retrieval
engine's view is a retrieved subset, so absence from it is absence from a *sample*, and the
engine says so through ``decides_absence`` rather than through a second code path.

How a corpus is read for mentions, and how a premise is built from an entity, are both the user's to
replace: an engine takes a ``MentionExtractor`` and a set of extra ``Transform`` implementations, and
defaults to the shipped ones. Without that, the shared flow would only fit a corpus of compound
identifiers and the four closed transforms — which is what it fit before, silently.

Nothing here imports an embedder or a graph library, so the shared flow costs no extra (FR-037).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from gaussia.core.probe_engine import ProbeEngine
from gaussia.schemas.roastme import KnowledgeHook, Probe

from .mentions import CompoundTokenExtractor
from .transforms import resolve

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from gaussia.core.transform import Transform
    from gaussia.schemas.roastme import Catalogue, Document, StrategySpec

    from .mentions import MentionExtractor

_QUERY_TEMPLATE = "{hint}: {premise}"
_ATTRIBUTE_SEPARATOR = ","


def extract_mentions(documents: Sequence[Document]) -> frozenset[str]:
    """The compound identifiers a corpus carries. Kept as the module-level shorthand it always was.

    Prefer ``CompoundTokenExtractor`` where an extractor is what is wanted, since that is the object
    an engine can be handed a replacement for.
    """
    return CompoundTokenExtractor().extract(documents)


def strategy_attributes(strategy: StrategySpec) -> list[str]:
    """The prose a probe exhibits, taken from the strategy's own description.

    The identifiers are the user's private vocabulary and may never cross to the Exploiter
    (FR-013), so what a probe carries is the description's clauses. A comma-separated description
    yields one attribute per clause, which is what lets a category be grounded in part of a
    pattern rather than in all of it.
    """
    return [clause.strip() for clause in strategy.description.split(_ATTRIBUTE_SEPARATOR) if clause.strip()]


def principle_by_plugin(catalogue: Catalogue) -> dict[str, str]:
    """Each risk family to the principle it attacks. The catalogue is expected validated (FR-025)."""
    return {plugin.id: plugin.principle for plugin in catalogue.plugins}


def build_probe(
    *,
    engine: str,
    strategy: StrategySpec,
    principle: str | None,
    entity: str,
    premise: str,
    boundary: frozenset[str],
    decides_absence: bool,
    index: int,
) -> Probe:
    """One tagged probe, with the ``doc`` label the engine's own view of the boundary supports.

    Args:
        engine: The producing engine's name, recorded so the absence/breadth trade-off stays
            measurable after composition (FR-022).
        strategy: The interaction pattern that asked for this probe.
        principle: The principle under test, or ``None`` for a control (FR-026).
        entity: The real entity the premise was derived from.
        premise: What the probe leans on, real or invented.
        boundary: Everything the engine can see for this entity kind. Membership *is* the label.
        decides_absence: Whether absence from ``boundary`` is absence from the base (FR-023).
        index: Position within this engine's run of the strategy, so identifiers are stable.
    """
    documented = premise in boundary
    hook = KnowledgeHook(
        kind=strategy.entity_kind,
        references=premise,
        doc=int(documented),
        how=strategy.transform,
        base_entity=entity if premise != entity else None,
        principle=principle,
        # A presence label is reliable whoever produced it — the engine saw the entity. Only
        # absence depends on whether the engine can see the whole boundary (FR-023).
        absence_reliable=documented or decides_absence,
    )
    return Probe(
        id=f"{engine}-{strategy.id}-{index}",
        query=_QUERY_TEMPLATE.format(hint=strategy.phrasing_hint, premise=premise),
        hook=hook,
        plugin=strategy.plugin,
        strategy=strategy.id,
        attrs=strategy_attributes(strategy),
        engine=engine,
        meta=_meta(entity, premise, documented=documented),
    )


def domain_agnostic_probes(catalogue: Catalogue, engine: str) -> list[Probe]:
    """FR-024: with no knowledge base, probes still come back — leaning on nothing.

    The hook stays empty rather than being filled with a placeholder: a fabricated hook would put
    an invented entity into the retained hooks ``H``, which is exactly what the Exploiter grounds
    its categories on. The probe is still recognisably a control or not, because ``plugin`` alone
    carries that (FR-026).
    """
    return [
        Probe(
            id=f"{engine}-{strategy.id}",
            query=strategy.phrasing_hint,
            hook=None,
            plugin=strategy.plugin,
            strategy=strategy.id,
            attrs=strategy_attributes(strategy),
            engine=engine,
        )
        for strategy in catalogue.strategies
    ]


class ParticularisingEngine(ProbeEngine, ABC):
    """The flow the shipped engines share, with the boundary left to the subclass.

    Template Method, because the four engines are one algorithm with one varying step. What a
    subclass supplies is ``_entities`` — the entities of a kind it can see — and
    ``decides_absence``. It never decides a ``doc`` label itself, which is how FR-021 stays one
    promise made in one place instead of four promises that can drift apart.

    Args:
        entity_kinds: The ``StrategySpec.entity_kind`` values this engine is being trusted with.
            Configuration rather than a discovery: extraction is kind-agnostic, so the engine
            cannot infer the user's vocabulary, and catalogue validation needs someone to have
            declared it before a typo can be caught (FR-025).
        extractor: How this engine reads mentions out of a corpus. Defaults to the compound-identifier
            reading, which is the shape of a corpus of numbered clauses and the wrong shape for
            prose — see ``mentions.py``. The enumeration engine ignores it: its boundary comes from
            the user's enumerator instead.
        transforms: Transform implementations beyond the four shipped, so a strategy can name a
            premise construction that suits the corpus. The four remain available; a key colliding
            with one of them is refused rather than silently preferred.
    """

    def __init__(
        self,
        entity_kinds: Iterable[str] = (),
        extractor: MentionExtractor | None = None,
        transforms: Sequence[Transform] = (),
    ) -> None:
        self._entity_kinds = frozenset(entity_kinds)
        self._extractor: MentionExtractor = extractor or CompoundTokenExtractor()
        self._transforms = tuple(transforms)

    @property
    def entity_kinds(self) -> frozenset[str]:
        return self._entity_kinds

    @property
    def extractor(self) -> MentionExtractor:
        """Recorded for reading, so a report can say which reading of the corpus produced a probe."""
        return self._extractor

    @property
    @abstractmethod
    def decides_absence(self) -> bool:
        """Whether absence from this engine's view is absence from the knowledge base (FR-023)."""

    @abstractmethod
    def _entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        """The entities of ``kind`` this engine can see in these documents.

        Same shape as ``EntityEnumerator.enumerate_entities`` on purpose: the enumeration engine
        is the case where this step *is* the user's enumerator, and the others are weaker views
        of the same thing.
        """

    def can_handle(self, document: Document) -> bool:
        """Any document is text an extractor can read.

        The enumeration engine overrides this: a boundary can only be established over a document
        that says it is enumerable.
        """
        return True

    def generate(self, documents: list[Document], catalogue: Catalogue) -> list[Probe]:
        if not documents:
            return domain_agnostic_probes(catalogue, self.name)
        boundaries = {kind: self._entities(kind, documents) for kind in _entity_kinds_of(catalogue)}
        principles = principle_by_plugin(catalogue)
        return [
            probe
            for strategy in catalogue.strategies
            for probe in self._probes_for(strategy, boundaries[strategy.entity_kind], principles)
        ]

    def _probes_for(
        self,
        strategy: StrategySpec,
        boundary: frozenset[str],
        principles: dict[str, str],
    ) -> list[Probe]:
        transform = resolve(strategy.transform, self._transforms)
        principle = principles[strategy.plugin] if strategy.plugin is not None else None
        return [
            build_probe(
                engine=self.name,
                strategy=strategy,
                principle=principle,
                entity=entity,
                premise=transform.apply(entity),
                boundary=boundary,
                decides_absence=self.decides_absence,
                index=index,
            )
            # Sorted so a run is reproducible: the boundary is a set, and an identifier derived
            # from iteration order would otherwise differ between runs over the same corpus.
            for index, entity in enumerate(sorted(boundary))
        ]


def _entity_kinds_of(catalogue: Catalogue) -> list[str]:
    return sorted({strategy.entity_kind for strategy in catalogue.strategies})


def _meta(entity: str, premise: str, *, documented: bool) -> dict[str, Any]:
    """What a grader needs to judge the answer: the real value, and the false one asserted."""
    meta: dict[str, Any] = {"real_value": entity}
    if not documented:
        meta["false_value"] = premise
    return meta
