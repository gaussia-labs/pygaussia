"""The graph engine: absence confirmed against the complete entity graph.

Its view of the corpus is every entity mention the graph has a node for, so a premise absent
from that view is absent from everything the corpus says — which is what makes a ``doc = 0``
label from this engine trustworthy (FR-021, SC-004). The trade-off against the retrieval engine
is exactly this: narrower reach, labels that hold.

The graph is co-occurrence: two entities are linked when a document mentions both. The
multi-hop engine composes the same graph, which is why building it is a module function rather
than a private method.

Needs the ``roastme`` extra for ``networkx``, so it is imported directly from this module and
never re-exported from the package (FR-037).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx

from .mentions import CompoundTokenExtractor
from .particularisation import ParticularisingEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.schemas.roastme import Document

    from .mentions import MentionExtractor


def entity_graph(documents: Sequence[Document], extractor: MentionExtractor | None = None) -> nx.Graph:
    """The co-occurrence graph over the corpus's entity mentions.

    Args:
        documents: The knowledge base.
        extractor: How mentions are read. Defaults to the compound-identifier reading, which is the
            wrong reading for a corpus of ordinary words — see ``mentions.py``.

    Returns:
        A graph whose nodes are the mentions and whose edges are same-document co-occurrence.
        The node set is the corpus boundary; the edges are what the multi-hop engine walks.
    """
    reader = extractor or CompoundTokenExtractor()
    graph = nx.Graph()
    for document in documents:
        mentions = sorted(reader.extract([document]))
        graph.add_nodes_from(mentions)
        graph.add_edges_from(combinations(mentions, 2))
    return graph


class GraphProbeEngine(ParticularisingEngine):
    """Probes whose grounding labels are settled by the graph's own completeness."""

    @property
    def name(self) -> str:
        return "graph"

    @property
    def decides_absence(self) -> bool:
        # The graph carries every mention the corpus makes, so absence from it is absence.
        return True

    def _entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        return frozenset(entity_graph(documents, self._extractor).nodes)
