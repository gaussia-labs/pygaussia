"""Target assistant abstract base class: the only path to the assistant under evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussia.schemas.roastme import TargetResponse


class TargetAssistant(ABC):
    """Adapter over the assistant under evaluation, implemented by the user for their transport.

    The assistant is sampled, never inspected (paper invariant 1): no parameters, no system
    prompt, no internal tools. This interface is the only path by which Roast Me contacts
    it, and both the Profiler and the Exploiter receive the same instance, so one adapter
    serves the whole subsystem (FR-017, spec D16).

    No transport adapter ships in gaussia (FR-018): a runtime-specific client belongs with
    that runtime. A hosted API, a local model, a browser page and a set of recorded
    responses are all implementations of this one interface — which is what makes the
    credential-free path of FR-014 the same code as a live run rather than a separate mode.

    Recognising a transport-level failure is the implementation's obligation, because the
    adapter is the only party that can: an error status, an empty body, or a payload shaped
    like an error. It must be reported by returning a response with ``failed`` set, not by
    raising, since the Profiler has to record that probe as ungraded (FR-016).
    """

    @abstractmethod
    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        """Send one query to the assistant and return what came back.

        Args:
            query: The question to ask.
            session_id: The persistent conversation to continue, or ``None`` for a
                single-turn exchange. An adapter that maintains sessions returns the
                identifier on the response so a multi-turn probe can continue one.

        Returns:
            The response. A refusal to answer is a legitimate response and must be
            returned as content — whether it violates a principle is the rubric's call.
            Only a transport failure sets ``failed``, and ``content`` may be empty only
            then.
        """
