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
The ten interfaces of ``core/`` stayed ten for this one. They are eleven since FR-042, and
that addition is the contrast worth keeping in view: ``FactTwister`` reads the corpus and
returns a claim about it, which is a contract a user implements against, while this reads the
corpus and returns what it says on its surface, for three shipped engines to intersect.

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
            Every mention recognised, deduplicated.

        Raises:
            ValueError: Nothing was recognised in a non-empty corpus (FR-044). An empty boundary
                used to be a legitimate answer here, on the argument that producing no probes is
                honest. It is not honest, because it is silent: the engine goes on to generate an
                empty probe set, the Profiler reports a violation rate over nothing, and the run
                completes. "This extractor cannot read this corpus" is a fact worth a failure, and
                the caller is the one who can act on it by supplying another.
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
        mentions = frozenset(
            mention for document in documents for mention in _COMPOUND_TOKEN.findall(document.content)
        )
        # FR-044. This catches only half of the failure this class is known for — the half where the
        # corpus is prose of ordinary words and no identifier appears in it. The other half, where
        # the corpus is scraped prose and the matches are phone numbers, cannot be caught here: a
        # false positive is well-formed, and deciding it is not an entity needs the domain. So the
        # loud refusal is the reachable part, and the docstring's advice above remains the rest.
        if not mentions and any(document.content.strip() for document in documents):
            message = (
                f"{type(self).__name__} recognised no mention in {len(documents)} document(s): it reads "
                "compound identifiers such as POLICY-1, and a corpus whose entities are ordinary words "
                "carries none. Supply a MentionExtractor for this corpus, or use EnumerationProbeEngine "
                "with your own enumerator"
            )
            raise ValueError(message)
        return mentions
