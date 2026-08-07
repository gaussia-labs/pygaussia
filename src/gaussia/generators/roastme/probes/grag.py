"""The multi-hop engine: false premises spanning several entities.

Where the graph engine leans on one entity, this one leans on a chain of them — two hops of the
co-occurrence graph, written out as the premise. A chain is documented only when the corpus
carries it whole, so a transformation applied anywhere along it makes the premise false while
every individual entity in it stays real. That is the attack this engine adds: an assistant that
refuses an invented entity may still accept an invented *relation* between real ones.

The boundary is the set of documented chains, so the same membership test that labels a single
entity labels a chain, and absence is confirmed the same way the graph engine confirms it.

Composes the graph engine's graph rather than building a second one, and therefore needs the
``roastme`` extra too (FR-037).
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from .graph import entity_graph
from .particularisation import ParticularisingEngine

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import networkx as nx

    from gaussia.schemas.roastme import Document

CHAIN_SEPARATOR = " -> "


class MultiHopProbeEngine(ParticularisingEngine):
    """Probes leaning on a two-hop chain of real entities.

    Args:
        entity_kinds: The entity kinds this engine is being trusted with.
        max_chains: How many chains to keep per entity kind. Chain enumeration is combinatorial
            in the graph's degree, so a corpus with a dense entity graph would otherwise produce
            probes without bound. A knob of gaussia's own engine, not a parameter of the method.
    """

    def __init__(self, entity_kinds: Iterable[str] = (), max_chains: int = 100) -> None:
        super().__init__(entity_kinds)
        self._max_chains = max_chains

    @property
    def name(self) -> str:
        return "multi-hop"

    @property
    def decides_absence(self) -> bool:
        # A chain the complete graph does not carry is a chain the corpus does not carry.
        return True

    def _entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        return frozenset(islice(_chains(entity_graph(documents)), self._max_chains))


def _chains(graph: nx.Graph) -> Iterator[str]:
    """Every two-hop walk that does not double back, in a fixed order so runs are reproducible."""
    for start in sorted(graph.nodes):
        for middle in sorted(graph.neighbors(start)):
            for end in sorted(graph.neighbors(middle)):
                if end != start:
                    yield CHAIN_SEPARATOR.join((start, middle, end))
