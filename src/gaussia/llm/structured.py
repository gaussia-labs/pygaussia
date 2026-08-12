"""How a chat model is asked to answer in a schema.

Judging needs no tools, so the strategy that carries no tool declaration is the default:
an OpenAI-compatible server is free to reject a request that declares an empty ``tools``
array (vLLM 0.23 answers 400), and constraining generation through ``response_format``
asks for the same guarantee without one. Providers that expose structured output only
through tool calling are served by ``ToolCallingOutput``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable


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
