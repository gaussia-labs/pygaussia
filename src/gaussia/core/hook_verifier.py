"""Hook verifier abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import Document, KnowledgeHook


class HookVerifier(ABC):
    """Confirms a hook's ``doc`` label against the corpus, independently of the engine.

    Shared across engines on purpose: one verifier applied to every engine's output is
    what makes their absence accuracy comparable. An engine grading its own labels would
    measure its confidence rather than its correctness.
    """

    @abstractmethod
    def verify(self, hook: KnowledgeHook, documents: list[Document]) -> bool | None:
        """Confirm the hook's ``doc`` label against the knowledge base.

        Args:
            hook: The hook to check. ``doc = 1`` claims the referenced entity exists in
                the base, ``doc = 0`` claims it does not.
            documents: The corpus to check against.

        Returns:
            ``True`` when the label holds, ``False`` when it does not, and ``None`` when
            this verifier could not establish a boundary to check it against.

            The third answer is the one a ``bool`` could not give, and its absence was
            silent in the direction the method's claims rest on: with no boundary, every
            absence label read as confirmed and every presence label as refuted, because
            "not in an empty set" is true and "in an empty set" is false. Neither was a
            finding about the corpus.

            The caller records this on ``KnowledgeHook.verified``, where ``None`` already
            means unverified — never ``False``, which would turn "nobody could check" into
            "the label is wrong".
        """
