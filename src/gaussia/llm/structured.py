"""How a chat model is asked to answer in a schema.

Judging needs no tools, so the strategy that carries no tool declaration is the default:
an OpenAI-compatible server is free to reject a request that declares an empty ``tools``
array (vLLM 0.23 answers 400), and constraining generation through ``response_format``
asks for the same guarantee without one. Providers that expose structured output only
through tool calling are served by ``ToolCallingOutput``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable


_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


class StructuredOutputStrategy(ABC):
    """Binds a schema to a model, yielding a runnable that returns raw and parsed output.

    Implementations must request ``include_raw`` so the caller keeps the message itself:
    a reasoning model carries its trace beside the parsed answer, and a parse failure has
    to be visible rather than raised out of the provider's own parser.
    """

    @abstractmethod
    def bind(self, model: "BaseChatModel", schema: type[BaseModel]) -> "Runnable":
        raise NotImplementedError("Subclass must implement this method")


class ResponseFormatOutput(StructuredOutputStrategy):
    """Constrains generation with a JSON schema and declares no tools.

    Args:
        strict: whether the server must enforce the schema exactly. Passed to the
            provider, which decides what enforcement means.
    """

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict

    def bind(self, model: "BaseChatModel", schema: type[BaseModel]) -> "Runnable":
        return model.with_structured_output(schema, method="json_schema", strict=self._strict, include_raw=True)


class ToolCallingOutput(StructuredOutputStrategy):
    """Asks for the schema as a tool call, for providers that offer no other route."""

    def bind(self, model: "BaseChatModel", schema: type[BaseModel]) -> "Runnable":
        return model.with_structured_output(schema, include_raw=True)


# A `TypeVar` rather than PEP 695 syntax, which is a *parse* error before 3.12 and this package
# floors at 3.11. `ruff` asked for the newer form while it was configured a target above that floor;
# it is aligned to `requires-python` now, so nothing asks again.
def parsed(answer: object, schema: type[_SchemaT]) -> "_SchemaT | None":
    """The parsed model out of what a bound runnable returned, or ``None`` when there is none.

    Lives beside ``bind`` because it is the other half of the same contract: every strategy above
    requests ``include_raw``, so what comes back is a mapping carrying the message itself next to the
    parsed value, and unwrapping it is not each caller's own business to reinvent.

    ``parsed`` is ``None`` exactly when the provider answered off-format. That happens, and what it
    costs is the caller's decision rather than this function's — a generator can re-ask, a reading of
    one passage can contribute nothing, and a gate that has to return a number has neither option.

    Args:
        answer: Whatever the bound runnable returned. Tolerates a bare model as well as the mapping,
            so a strategy that does not request ``include_raw`` is read rather than mistaken for a
            failure.
        schema: The model class that was bound.

    Returns:
        The instance, or ``None``.
    """
    if isinstance(answer, schema):
        return answer
    if isinstance(answer, dict):
        value = answer.get("parsed")
        return value if isinstance(value, schema) else None
    return None
