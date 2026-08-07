"""The retrieval engine: breadth of false premises, and absence it cannot confirm.

Its view of the corpus is whatever similarity search returned for an entity kind, which is a
*sample* of the base and never the whole of it. Absence from a sample is not absence, so every
hook this engine labels ``doc = 0`` records ``absence_reliable = False`` (FR-023): similarity
search is structurally unable to decide absence, because it never reveals what it failed to
retrieve. That is the price of its reach, and recording it is what keeps an unreliable label
distinguishable from one the graph engine confirmed (SC-004).

The embedder is injected: the framework already specifies one, and encoding vectors is not this
engine's job. The retrieval query is the strategy's ``entity_kind`` — the only statement of what
to look for that a strategy carries in the user's own words. Gaussia never interprets it; it
only embeds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .particularisation import ParticularisingEngine, extract_mentions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gaussia.core.embedder import Embedder
    from gaussia.schemas.roastme import Document

# Keeps a zero-norm embedding from turning a similarity into a warning and then into a NaN.
_EPSILON = 1e-12


class RetrievalProbeEngine(ParticularisingEngine):
    """Probes over the entities similarity search surfaces, with their absence labels flagged.

    Args:
        embedder: The framework's embedding interface. Composed rather than built, so the same
            engine runs against any model the user already has.
        entity_kinds: The entity kinds this engine is being trusted with.
        top_documents: How many retrieved documents form the view. A knob of gaussia's own
            engine: raising it widens the sample and lowers, without ever removing, the chance
            that an absence label is wrong.
    """

    def __init__(self, embedder: Embedder, entity_kinds: Iterable[str] = (), top_documents: int = 3) -> None:
        super().__init__(entity_kinds)
        self._embedder = embedder
        self._top_documents = top_documents

    @property
    def name(self) -> str:
        return "retrieval"

    @property
    def decides_absence(self) -> bool:
        # What was never retrieved was never seen, so this engine cannot rule an entity out.
        return False

    def _entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        return extract_mentions(self._retrieve(kind, documents))

    def _retrieve(self, query: str, documents: list[Document]) -> list[Document]:
        corpus = self._embedder.encode([document.content for document in documents])
        wanted = self._embedder.encode_query([query])[0]
        norms = np.linalg.norm(corpus, axis=1) * float(np.linalg.norm(wanted)) + _EPSILON
        similarity = corpus @ wanted / norms
        ranked = sorted(range(len(documents)), key=lambda position: -float(similarity[position]))
        return [documents[position] for position in ranked[: self._top_documents]]
