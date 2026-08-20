"""Which model answered, as a string a report can attribute a result to."""

from __future__ import annotations

from typing import Any


def model_identity(model: Any) -> str:
    """The model's own identifier, falling back to the adapter class only when it has none.

    Recorded on ``PrincipleGrade.model`` and ``Probe.model``, and read rather than branched on. The
    point of recording it is attribution: two runs that disagree have to be separable into "the
    assistant changed" and "the judge changed", and a weak result has to be attributable to the
    substitutable piece that produced it.

    Which is why the adapter's class name is the fallback and never the answer. ``ChatOpenAI`` is
    what every OpenAI-compatible provider is reached through — a local server, a router, a hosted
    API, four different models behind one class — so a run recording it has recorded nothing that
    separates one from another, while looking exactly like a run that recorded something.

    ``model_name`` is LangChain's field for it and covers the chat models; ``model`` covers the
    adapters that name the field after the argument instead. Both are read defensively because this
    runs on the user's model and the interface gaussia asks for is ``BaseChatModel``, which
    guarantees neither.
    """
    for attribute in ("model_name", "model"):
        # Annotated `object` rather than left to inference: `getattr` returns `Any`, and returning
        # it from a function declared `-> str` is exactly the hole `no-any-return` exists to catch.
        name: object = getattr(model, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(model).__name__
