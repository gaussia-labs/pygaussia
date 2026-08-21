"""The Probe Library: several engines over one knowledge base, merged into one probe set.

Composition rather than a cascade (FR-022): every engine that can handle a document sees it, and
every engine's output contributes. That is what makes the absence/breadth trade-off a thing a
run can measure rather than a thing a user has to choose up front — the graph engine's confirmed
labels and the retrieval engine's reach arrive in the same set, each probe still saying which
engine produced it.

Two consequences of merging worth stating, because both are easy to get silently wrong:

* identity is ``Probe.id``. The shipped engines scope theirs by engine name, so probes two
  engines derived *independently* stay distinct — otherwise a confirmed absence label and an
  unreliable one would collapse into one probe and FR-023 would stop holding after composition.
  What merges is the same probe surfaced twice, and the survivor records every engine that
  produced it;
* with no knowledge base there is no boundary anyone could have consulted, so no hook may claim
  one. The probes still come back, domain-agnostic and hookless (FR-024): a fabricated hook
  would put an invented entity into the retained hooks the Exploiter grounds categories on.

This module imports no engine, and must not: it is re-exported from the subsystem facade, so an
engine import here would make ``from gaussia.generators.roastme import ProbeLibrary`` require an
embedder and a graph library (FR-037). The default three are composed by the caller.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from gaussia.schemas.roastme import EngineDeclaration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.hook_verifier import HookVerifier
    from gaussia.core.probe_engine import ProbeEngine
    from gaussia.schemas.roastme import Catalogue, Document, Probe

MERGED_ENGINES = "merged_engines"
"""Where a merged probe records every engine that produced it. Read, never branched on."""

PAPER = "gaussia-labs/papers#20"
"""The version of the paper ``IN_PAPER_TABLES`` was read from, merged to ``main``."""

IN_PAPER_TABLES = ("retrieval", "graph", "enumeration")
"""The engines the paper's absence/breadth trade-off tables characterise.

Multi-hop is deliberately absent: the paper characterises it by example and puts no number on it,
so a run composing it produced a figure the tables do not cover. A fact about the paper rather than
about any run, which is why it is a constant here and pinned to ``PAPER``.
"""


class ProbeLibrary:
    """Composes probe engines over one knowledge base and one validated catalogue.

    Args:
        engines: The engines to run. Retrieval, graph and multi-hop are the default three; the
            enumeration engine joins them only when the user has an enumerator to inject
            (spec D14). Passed in rather than discovered, so composing them never costs an
            extra nobody installed.
        verifier: Confirms each hook's ``doc`` label against the corpus, independently of the
            engine that produced it. Here rather than in an engine because that is what makes
            two engines' absence accuracy comparable — an engine checking its own labels would
            measure its confidence instead. ``None`` leaves every ``verified`` at ``None``,
            which continues to mean nobody checked.
    """

    def __init__(self, engines: Sequence[ProbeEngine], verifier: HookVerifier | None = None) -> None:
        self._engines = list(engines)
        self._verifier = verifier

    @property
    def declaration(self) -> EngineDeclaration:
        """Which engines this library composes, and which of them the paper put a number on.

        Here because this is the only object that knows the composed set: an engine that ran and
        produced no probe is invisible in the probe set, and that is exactly the case FR-025 makes
        interesting. It stays on this side of the Profiler/Exploiter boundary — the weakness
        profile is the only artifact that crosses it (FR-013), and an engine name is precisely the
        kind of identifier that may not.
        """
        ran = [engine.name for engine in self._engines]
        return EngineDeclaration(
            ran=ran,
            in_paper_tables=[name for name in ran if name in IN_PAPER_TABLES],
            outside_paper_tables=[name for name in ran if name not in IN_PAPER_TABLES],
            paper=PAPER,
        )

    def generate(self, documents: Sequence[Document], catalogue: Catalogue) -> list[Probe]:
        """The merged probe set.

        Args:
            documents: The knowledge base, or an empty list for a black-box run.
            catalogue: The plugins and strategies, already validated against the contract and
                against these engines (FR-025).

        Returns:
            One probe per surviving identity, each recording its originating engine.
        """
        produced = [
            (engine.name, probe)
            for engine in self._engines
            for probe in engine.generate(self._reachable(engine, documents), catalogue)
        ]
        merged: dict[str, Probe] = {}
        contributors: defaultdict[str, list[str]] = defaultdict(list)
        for name, probe in produced:
            merged.setdefault(probe.id, probe.model_copy(update={"engine": probe.engine or name}))
            contributors[probe.id].append(probe.engine or name)

        probes = [_with_provenance(probe, contributors[probe.id]) for probe in merged.values()]
        if not documents:
            return [probe.model_copy(update={"hook": None}) for probe in probes]
        return [self._verified(probe, list(documents)) for probe in probes]

    def _verified(self, probe: Probe, documents: list[Document]) -> Probe:
        """The probe with its label's verdict recorded, and nothing else changed.

        The probe survives whatever the verdict is. A failed label means this probe cannot support the
        claim it was built to make, which is something the human reviewing the probe set has to see —
        dropping it here would shrink the denominator without saying so, and a smaller denominator that
        announces nothing is the failure mode this whole check exists to catch.
        """
        if self._verifier is None or probe.hook is None:
            return probe
        held = self._verifier.verify(probe.hook, documents)
        return probe.model_copy(update={"hook": probe.hook.model_copy(update={"verified": held})})

    @staticmethod
    def _reachable(engine: ProbeEngine, documents: Sequence[Document]) -> list[Document]:
        return [document for document in documents if engine.can_handle(document)]


def _with_provenance(probe: Probe, contributors: list[str]) -> Probe:
    """A probe two engines produced still says so, so composition stays measurable (FR-022)."""
    if len(contributors) == 1:
        return probe
    return probe.model_copy(update={"meta": {**probe.meta, MERGED_ENGINES: contributors}})
