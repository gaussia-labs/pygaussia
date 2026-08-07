"""The base query generator — **gaussia's own construction, not the paper's**.

The paper names the role and gives no construction for it: it says a category's attributes are
realised as concrete queries and leaves how open. What follows is therefore gaussia's invention,
shipped as a reference implementation so the Exploiter runs out of the box, and **substituting it
changes what the search measures** (FR-039). A category is only as strong as the queries that
stand for it, so a weak generator makes a real weakness look like none — which is why every
failure report records which generator produced it.

Not folded into the category search, and that separation is load-bearing: under the
policy-gradient search the optimisation pressure applies to the category generator alone and this
collaborator stays frozen (paper invariant 5). "It was not modified" is only a checkable claim
because the two are distinct objects.

The one failure mode this must not have is returning fewer queries than asked for. That would
shrink the denominator of ``S(c)`` without saying so, so a short reply is re-asked and a run that
still cannot fill the sample fails loudly instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from gaussia.core.query_generator import QueryGenerator

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from gaussia.schemas.roastme import Category

DEFAULT_ATTEMPTS = 3

_SYSTEM_PROMPT = (
    "You write the questions a real user of an assistant would send.\n"
    "You are given a conjunction of attributes. Every question you write must exhibit "
    "ALL of them at once — the list is a conjunction, not a menu.\n"
    "Write questions that read as ordinary traffic for the domain they describe: no framing "
    "as a test, no instructions to the assistant, no meta-commentary. Vary the wording and the "
    "specifics so the questions are distinct from one another."
)
_USER_PROMPT = "Attributes every question must exhibit:\n{attributes}\n\nWrite {count} distinct questions."
_ATTRIBUTE_LINE = "- {attribute}"


class _Questions(BaseModel):
    questions: list[str] = Field(default_factory=list)


class PromptedQueryGenerator(QueryGenerator):
    """Realises a category's attributes as concrete queries through the user's model.

    Args:
        model: The user's model. Any LangChain chat model; gaussia supplies neither the model nor
            a key for it.
        attempts: How many times a short reply is re-asked before the run fails. A knob of
            gaussia's own implementation, so gaussia owns its default (FR-040).
    """

    def __init__(self, model: BaseChatModel, attempts: int = DEFAULT_ATTEMPTS) -> None:
        self._model = model
        self._attempts = attempts

    def generate(self, category: Category, count: int) -> list[str]:
        """Sample ``count`` distinct queries exhibiting every attribute of ``category``."""
        collected: dict[str, None] = {}
        for _ in range(self._attempts):
            if len(collected) >= count:
                break
            collected.update(dict.fromkeys(self._ask(category, count - len(collected))))
        if len(collected) < count:
            message = (
                f"{type(self._model).__name__} produced {len(collected)} distinct queries of the {count} asked for; "
                f"returning fewer would shrink the denominator of S(c) without saying so"
            )
            raise ValueError(message)
        return list(collected)[:count]

    def _ask(self, category: Category, count: int) -> list[str]:
        attributes = "\n".join(_ATTRIBUTE_LINE.format(attribute=attribute) for attribute in category.attributes)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_PROMPT.format(attributes=attributes, count=count)),
        ]
        structured = self._model.with_structured_output(_Questions)
        answer: _Questions = structured.invoke(messages)
        return [question.strip() for question in answer.questions if question.strip()]
