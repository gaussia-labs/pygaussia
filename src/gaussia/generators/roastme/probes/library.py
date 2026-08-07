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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.probe_engine import ProbeEngine
    from gaussia.schemas.roastme import Catalogue, Document, Probe

MERGED_ENGINES = "merged_engines"
"""Where a merged probe records every engine that produced it. Read, never branched on."""


class ProbeLibrary:
    """Composes probe engines over one knowledge base and one validated catalogue.

    Args:
        engines: The engines to run. Retrieval, graph and multi-hop are the default three; the
            enumeration engine joins them only when the user has an enumerator to inject
            (spec D14). Passed in rather than discovered, so composing them never costs an
            extra nobody installed.
    """

    def __init__(self, engines: Sequence[ProbeEngine]) -> None:
        self._engines = list(engines)

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
        if documents:
            return probes
        return [probe.model_copy(update={"hook": None}) for probe in probes]

    @staticmethod
    def _reachable(engine: ProbeEngine, documents: Sequence[Document]) -> list[Document]:
        return [document for document in documents if engine.can_handle(document)]


def _with_provenance(probe: Probe, contributors: list[str]) -> Probe:
    """A probe two engines produced still says so, so composition stays measurable (FR-022)."""
    if len(contributors) == 1:
        return probe
    return probe.model_copy(update={"meta": {**probe.meta, MERGED_ENGINES: contributors}})
