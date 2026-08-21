"""A deterministic stand-in for the user's judge model, for the shipped logprob grader (T016).

The response shape mirrors the one the framework already parses in `llm/judge.py`:
`response_metadata["logprobs"]["content"]` is a list of per-token entries, each carrying its
own `token` and its own `top_logprobs` alternatives. The difference the grader under test has
to exhibit is *which* entry it reads — the last verdict-shaped one rather than the first, since
the first belongs to a reasoning model's preamble (FR-007).

`logprobs_supported=False` makes the bound call raise `LogprobsNotSupportedError`, which is the
framework's own exception for a provider that returns no usable logprobs; the grader is
expected to fall back to sampling over `k` and mark the grades accordingly (FR-008, SC-009).
"""

import math
from typing import Any

from gaussia.core.exceptions import LogprobsNotSupportedError


def token_entry(token: str, alternatives: dict[str, float]) -> dict[str, Any]:
    """One per-token logprob entry, with its alternatives given as plain probabilities.

    Probabilities are converted to logprobs here so a fixture reads as probabilities and the
    expected verdict score stays hand-computable from them.
    """
    return {
        "token": token,
        "logprob": math.log(max(alternatives.values())),
        "top_logprobs": [{"token": name, "logprob": math.log(p)} for name, p in alternatives.items()],
    }


def raw_token_entry(token: str, logprobs: dict[str, float]) -> dict[str, Any]:
    """One per-token entry whose alternatives are given as logprobs rather than probabilities.

    The sibling of `token_entry`, for the values a probability cannot express: a provider filling
    in a sentinel such as `-9999.0` for a surface form it did not rank is reporting a probability
    of roughly `1e-4343`, which no float can round-trip.
    """
    return {
        "token": token,
        "logprob": max(logprobs.values()),
        "top_logprobs": [{"token": name, "logprob": logprob} for name, logprob in logprobs.items()],
    }


class StubResponse:
    """What a LangChain chat model returns, reduced to what a logprob grader reads."""

    def __init__(self, content: str, token_entries: list[dict[str, Any]] | None = None):
        self.content = content
        self.response_metadata: dict[str, Any] = (
            {"logprobs": {"content": token_entries}} if token_entries is not None else {}
        )
        self.additional_kwargs: dict[str, Any] = {}


class StubJudgeModel:
    """A chat model whose logprob sequence, final answer and sampled answers are prescribed.

    One `invoke` entry point serves both paths, dispatching on whether logprobs were requested
    in `bind`, so the double does not assume the grader reaches the sampling fallback through
    any particular call shape.
    """

    def __init__(
        self,
        token_entries: list[dict[str, Any]] | None = None,
        final_content: str = "",
        sample_contents: list[str] | None = None,
        logprobs_supported: bool = True,
    ):
        self.token_entries = token_entries
        self.final_content = final_content
        self.sample_contents = sample_contents or []
        self.logprobs_supported = logprobs_supported
        self.bind_calls: list[dict[str, Any]] = []
        self.invoked_with: list[Any] = []
        self.logprob_invocations = 0
        self.sample_invocations = 0

    def bind(self, **kwargs: Any) -> "_BoundStubJudgeModel":
        self.bind_calls.append(kwargs)
        return _BoundStubJudgeModel(self, logprobs_requested=bool(kwargs.get("logprobs")))

    def invoke(self, messages: Any, **kwargs: Any) -> StubResponse:
        self.invoked_with.append(messages)
        return self.sample()

    def prompt_text(self) -> str:
        """Everything the grader sent, flattened, so a rubric can be checked for surviving intact."""
        return " ".join(str(messages) for messages in self.invoked_with)

    def logprob_response(self) -> StubResponse:
        if not self.logprobs_supported:
            raise LogprobsNotSupportedError("stub provider exposes no logprobs")
        self.logprob_invocations += 1
        return StubResponse(self.final_content, self.token_entries)

    def sample(self) -> StubResponse:
        index = self.sample_invocations
        self.sample_invocations += 1
        return StubResponse(self.sample_contents[index])


class _BoundStubJudgeModel:
    def __init__(self, model: StubJudgeModel, logprobs_requested: bool):
        self._model = model
        self._logprobs_requested = logprobs_requested

    def invoke(self, messages: Any, **kwargs: Any) -> StubResponse:
        self._model.invoked_with.append(messages)
        if self._logprobs_requested:
            return self._model.logprob_response()
        return self._model.sample()
