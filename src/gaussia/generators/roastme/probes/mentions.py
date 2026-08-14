"""How an engine finds the entities a corpus mentions, and how to say it differently.

The three engines that read a corpus rather than being handed a boundary — retrieval, graph and
multi-hop — need one thing from the text: which entities it mentions. That step used to be a single
regex compiled into the shared flow, and the regex recognises **compound identifiers**: alphanumerics
joined by ``-`` or ``_``. It reads ``POLICY-1`` and ``Articulo_25``, which is the shape of a corpus of
numbered clauses, and it is language-neutral because it recognises punctuation rather than words.

What it cannot read is a corpus whose entities are ordinary words. Worse than finding nothing, on
scraped prose it finds the *wrong* things — phone numbers, document filenames, URL slugs — and the
engine then generates a probe per false positive with no error, because an empty or junk boundary is
indistinguishable from a legitimate one at that layer.

So the step is injectable, the same way ``EnumerationProbeEngine`` already takes an
``EntityEnumerator``. The default stays the compound-identifier reading, so a corpus that worked before
still works; a corpus of prose supplies its own.

``MentionExtractor`` lives here rather than in ``core/`` on purpose. It is a collaborator of three
shipped engines, not part of the specification a user implements against — the same call already made
for ``CategoryPolicy`` and ``PolicyUpdateStep``, which sit beside the search that samples through them.
The ten interfaces of ``core/`` stay ten.

Neither this module nor its default imports anything beyond the standard library (FR-037).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.schemas.roastme import Document

_COMPOUND_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


class MentionExtractor(ABC):
    """Reads the entities a corpus mentions on its surface.

    An implementation decides what counts as an entity in *this* corpus. Gaussia never learns what an
    entity kind means (FR-025), so the extractor is not told the kind: it returns everything it
    recognises and the engine intersects that with what a strategy asks for.

    One obligation the signature cannot express: **return what the corpus actually carries, not what
    it might carry.** The boundary this produces is what decides every ``doc`` label the engine emits,
    so a false positive becomes a probe about an entity nobody documented, and a false negative makes
    a documented entity look invented.
    """

    @abstractmethod
    def extract(self, documents: Sequence[Document]) -> frozenset[str]:
        """The entity mentions these documents carry.

        Args:
            documents: The documents this engine was given. Already filtered by ``can_handle``.

        Returns:
            Every mention recognised, deduplicated. An empty set is a legitimate answer and means
            the engine will produce no probes — which is the honest outcome when the corpus carries
            nothing this extractor can read.
        """


class CompoundTokenExtractor(MentionExtractor):
    """The default: alphanumerics joined by ``-`` or ``_``.

    Recognises ``POLICY-1``, ``FORM-7``, ``Articulo_25``. Does **not** recognise ``Cuenta Digital
    Libre`` or ``insulin``, and on scraped prose it recognises the wrong things: ``809-544-5555``,
    ``BP_TARIFAS_2026``, ``9-footer``.

    **Measured, so the warning is not abstract.** Over a corpus of Spanish banking product pages it
    returns **208 mentions that are not entities** — phone numbers, PDF filenames, footer anchors.
    None of them fails: each becomes the boundary for a strategy, each yields a probe, the assistant
    is asked about a filename, and the run completes with a violation rate over questions nobody
    would ask. The engine cannot tell a junk boundary from a legitimate one, so nothing raises.

    Check what it returns against your own corpus before trusting a run built on it::

        print(sorted(CompoundTokenExtractor().extract(documents))[:30])

    ``EnumerationProbeEngine`` is the way out for a corpus of ordinary words: it takes the user's own
    enumerator instead of reading mentions off the surface, and it is the only engine that can claim
    absence with any ground under it.
    """

    def extract(self, documents: Sequence[Document]) -> frozenset[str]:
        return frozenset(
            mention for document in documents for mention in _COMPOUND_TOKEN.findall(document.content)
        )
