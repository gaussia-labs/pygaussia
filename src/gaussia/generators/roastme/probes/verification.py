"""Confirming a hook's ``doc`` label, and the one way a false premise fails that nothing else catches.

An absence label is the load-bearing claim of the whole method: a probe leans on a premise the base does
not carry, and the assistant is charged for treating it as real. ``build_probe`` decides that label with
``premise in boundary`` (``particularisation.py``), which answers the question exactly — and the exact
answer is not the useful one.

**What the exact test misses.** A premise absent from the boundary can still be a *misspelling of something
present*. ``Pago a Nóminas`` is not in the corpus; ``Pago de Nóminas`` is, one preposition away. The
assistant retrieves the real product and answers about it correctly, and the run charges it for inventing.
That is not a graded judgement gone wrong — it is a false positive built into the probe before anybody
called the assistant, and it was found inside a reported number.

**Three criteria, and all three are load-bearing.** Measured against a corpus of 136 Spanish product names:

===================  ============================================================  ==============
criterion            the case only it catches                                      the numbers
===================  ============================================================  ==============
containment          ``Factoring`` inside ``Factoring con Recursos``                ratio 0.58, 13 edits
ratio                ``Seguros Tu Servicios`` vs ``Seguros Tus Servicios``          ratio 0.976
edit distance        ``Pago a Nóminas`` vs ``Pago de Nóminas``                      2 edits, ratio 0.897
===================  ============================================================  ==============

Drop containment and the first survives; drop the ratio and the second does; drop the distance and the
third does. Two of the three are not enough.

**Why this is opt-in rather than the default.** The thresholds are calibrated on short name-like entities.
An entity kind whose members are whole sentences — ``Balance mínimo de Cuenta Digital Libre es RD$0.00``,
which is what a value enumerator yields — sits above 0.97 against its own one-digit flip, so a verifier
applied to that kind would reject every legitimate value falsification. The check is right for names and
wrong for statements, and nothing in a hook says which kind it is looking at. So the user attaches it to
the kinds it fits, and ``ProbeLibrary`` records the verdict without acting on it.

**Nobody is dropped.** The probe is still produced; ``KnowledgeHook.verified`` says whether its label held.
Silently shrinking a probe set is how a run reports a smaller denominator as if it had measured the whole
thing, and this module exists because of a false positive that nothing announced — it is not going to
answer it with a silent deletion.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from gaussia.core.hook_verifier import HookVerifier

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gaussia.core.entity_enumerator import EntityEnumerator
    from gaussia.schemas.roastme import Document, KnowledgeHook

MAX_RATIO = 0.93
"""Above this, two names are the same name written twice. ``Seguros Tu Servicios`` scores 0.976."""

MAX_EDITS = 2
"""At or below this, a candidate is a real entity misspelled. ``Pago a Nóminas`` is 2 away from a real one."""


def fold(text: str) -> str:
    """Case and accents removed, so ``Nóminas`` and ``nominas`` compare as the same word.

    A corpus in Spanish carries the same product name with and without its accents depending on which page
    it came from, and a comparison that treats those as different lets the near miss through on a detail
    of transcription.
    """
    stripped = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in stripped if not unicodedata.combining(character))


def edit_distance(left: str, right: str, ceiling: int) -> int:
    """Levenshtein distance, abandoned once it is known to exceed ``ceiling``.

    The ceiling is not an optimisation detail: the caller only ever asks "is this within two edits", and a
    full distance between two long sentences is work whose answer is discarded.
    """
    if abs(len(left) - len(right)) > ceiling:
        return ceiling + 1
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, start=1):
        current = [i]
        for j, right_character in enumerate(right, start=1):
            current.append(
                previous[j - 1]
                if left_character == right_character
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        if min(current) > ceiling:
            return ceiling + 1
        previous = current
    return previous[-1]


def collision(
    premise: str, boundary: Iterable[str], *, max_ratio: float = MAX_RATIO, max_edits: int = MAX_EDITS
) -> str | None:
    """The entity this premise is indistinguishable from, or ``None`` when it stands on its own.

    Args:
        premise: The text a transformation produced and a hook claims is absent.
        boundary: Everything the engine can see for the premise's entity kind.
        max_ratio: Similarity above which two names are the same name. See ``MAX_RATIO``.
        max_edits: Edit distance at or below which a candidate is a misspelling. See ``MAX_EDITS``.

    Returns:
        The colliding entity as the boundary carries it — returned rather than a bool so a report can name
        what the premise collided with, which is the whole difference between "7 probes were rejected" and
        a list somebody can check.
    """
    folded = fold(premise)
    if not folded:
        return None
    for entity in sorted(boundary):
        other = fold(entity)
        if not other:
            continue
        if folded in other or other in folded:
            return entity
        if SequenceMatcher(None, folded, other).ratio() >= max_ratio:
            return entity
        if edit_distance(folded, other, max_edits) <= max_edits:
            return entity
    return None


class NearMissVerifier(HookVerifier):
    """Confirms a hook's ``doc`` label, treating an indistinguishable premise as a failed absence claim.

    A ``doc = 1`` hook claims the premise is in the base, and that is confirmed by membership: the engine
    saw the entity, and presence needs no interpretation.

    A ``doc = 0`` hook claims the premise is *not* in the base, and that is the claim this class exists to
    doubt. Absent and indistinguishable from something present is not a defensible absence: the assistant
    will retrieve the real entity, answer about it, and be charged for fabricating.

    Args:
        enumerator: How the boundary is read for a hook's entity kind. The same object the enumeration
            engine takes, so the verifier checks against exactly the view the label was derived from
            rather than a second reading that could disagree with it.
        kinds: The entity kinds to check, or empty for all of them. Naming them is the normal case: the
            thresholds fit short names and misfire on sentence-shaped entities — see the module docstring.
        max_ratio: Overrides ``MAX_RATIO``.
        max_edits: Overrides ``MAX_EDITS``.
    """

    def __init__(
        self,
        enumerator: EntityEnumerator,
        kinds: Iterable[str] = (),
        *,
        max_ratio: float = MAX_RATIO,
        max_edits: int = MAX_EDITS,
    ) -> None:
        self._enumerator = enumerator
        self._kinds = frozenset(kinds)
        self._max_ratio = max_ratio
        self._max_edits = max_edits

    def verify(self, hook: KnowledgeHook, documents: list[Document]) -> bool | None:
        """Whether the hook's ``doc`` label holds, or ``None`` when it could not be checked.

        Two cases answer ``None``, and they used to answer ``True`` for want of anywhere else to
        go — which recorded an unperformed check as a passed one:

        * a kind this verifier was not given. It was not checked, and reporting an unchecked label
          as wrong is the one reading the ABC rules out — but reporting it as confirmed is the
          other, and the ABC now has a word for neither;
        * an empty boundary. Nothing was enumerated, so there is nothing to check against. Read as
          a verdict it was worse than useless: ``references not in boundary`` is true of every
          absence label over an empty set and ``references in boundary`` is false of every presence
          label, so a verifier that could see nothing confirmed every absence claim and refuted
          every presence claim. Silent, and in the direction the method's own claims rest on.
        """
        if self._kinds and hook.kind not in self._kinds:
            return None
        boundary = self._enumerator.enumerate_entities(hook.kind, documents)
        if not boundary:
            return None
        if hook.doc == 1:
            return hook.references in boundary
        return hook.references not in boundary and self.collision(hook, documents) is None

    def collision(self, hook: KnowledgeHook, documents: list[Document]) -> str | None:
        """What the hook's premise collided with, for a report that names it instead of counting it."""
        boundary = self._enumerator.enumerate_entities(hook.kind, documents)
        return collision(
            hook.references,
            boundary - {hook.references},
            max_ratio=self._max_ratio,
            max_edits=self._max_edits,
        )
