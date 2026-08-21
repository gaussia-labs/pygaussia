"""Judge class for LLM-based evaluation."""

import json
import logging
import math
import re
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from gaussia.core.exceptions import LogprobsExtractionError, LogprobsNotSupportedError
from gaussia.llm.structured import ResponseFormatOutput, StructuredOutputStrategy
from gaussia.utils.logging import VerboseLogger

T = TypeVar("T", bound=BaseModel)

_DEFAULT_POSITIVE_TOKENS: tuple[str, ...] = ("YES", "Yes", "yes", " YES", " Yes", " yes")
_DEFAULT_NEGATIVE_TOKENS: tuple[str, ...] = ("NO", "No", "no", " NO", " No", " no")
_DEFAULT_SCAN_TOKENS = 32
_DEFAULT_EXTRACTION_RETRIES = 2
_DEFAULT_STRUCTURED_RETRIES = 4


class Judge:
    """LLM-based judge for evaluating AI responses.

    Supports two modes:
    - Structured output: binds the schema to the model through a StructuredOutputStrategy
    - Regex extraction: Parses JSON from model response using regex patterns

    Reasoning content is automatically extracted from LangChain's
    additional_kwargs.reasoning_content when available.

    Args:
        model: LangChain BaseChatModel instance
        use_structured_output: If True, bind the output schema to the model
        strict: Whether the provider must enforce the schema exactly. Ignored when
            structured_output is supplied — the strategy carries its own setting.
        bos_json_clause: Opening marker for JSON block (default: ```json)
        eos_json_clause: Closing marker for JSON block (default: ```)
        structured_output: How the schema is bound. Defaults to ResponseFormatOutput,
            which declares no tools and so works against servers that reject an empty
            tools array.
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        verbose: bool = False,
        structured_output: StructuredOutputStrategy | None = None,
    ):
        self.model = model
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.verbose = verbose
        self.structured_output = structured_output or ResponseFormatOutput(strict=strict)
        self.chat_history: list[tuple[str, str]] = []
        self.logger = VerboseLogger(verbose=verbose)

    def check(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[T] | None = None,
    ) -> tuple[str, T | dict | None]:
        """Evaluate using the model.

        If use_structured_output=True and output_schema provided:
            - Renders the system prompt template with data variables
            - Binds the schema to the model through the structured output strategy
        Else:
            - Includes full JSON schema in prompt
            - Uses regex extraction from response

        Args:
            system_prompt: System prompt template for the evaluation
            query: User query to evaluate
            data: Template variables for the system prompt
            output_schema: Pydantic model for structured output validation

        Returns:
            Tuple of (reasoning_content, result) where result is either a Pydantic
            model instance (structured mode) or dict (regex mode). reasoning_content
            is extracted from LangChain's additional_kwargs when available.
        """
        if self.use_structured_output and output_schema:
            return self._check_structured(system_prompt, query, data, output_schema)
        return self._check_regex(system_prompt, query, data, output_schema)

    def _render_system_prompt(self, system_prompt: str, data: dict) -> str:
        return system_prompt.format_map(data)

    def _conversation(self, rendered_system: str, query: str) -> list[tuple[str, str]]:
        # Messages are handed to the model already rendered, never as a template: a
        # query or a JSON schema carrying a literal brace is content, and a prompt
        # template would read it as a variable and raise before any request is sent.
        self.chat_history.append(("human", query))
        return [("system", rendered_system), *self.chat_history]

    def _get_json_schema_for_prompt(self, schema: type[BaseModel]) -> str:
        schema_json = schema.model_json_schema()
        props = schema_json.get("properties", {})
        field_lines = [
            f'    "{name}": <{prop.get("type", "any")}> // {prop.get("description", "")}'
            for name, prop in props.items()
        ]
        fields_str = "\n".join(field_lines)
        return f"""
After your reasoning, provide ONLY the final answer in the following JSON format:
```json
{{
{fields_str}
}}
```

Do not include any additional text after the JSON.
"""

    def _check_structured(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[T],
    ) -> tuple[str, T | None]:
        rendered_system = self._render_system_prompt(system_prompt, data)
        messages = self._conversation(rendered_system, query)
        bound = self.structured_output.bind(self.model, output_schema)

        attempts = _DEFAULT_STRUCTURED_RETRIES + 1
        reasoning = ""
        for attempt in range(attempts):
            # An answer that misses the schema is re-asked, since re-asking is a fresh
            # draw. A provider that refuses the request is raised through: every attempt
            # would send the identical request and be refused identically.
            try:
                answer = bound.invoke(messages)
            except (OutputParserException, ValidationError) as error:
                self.logger.warning(f"Retry {attempt + 1}/{attempts} - answer did not satisfy the schema: {error}")
                continue
            reasoning = _reasoning_of(answer.get("raw"))
            if answer.get("parsed") is not None:
                return reasoning, answer["parsed"]
            self.logger.warning(f"Retry {attempt + 1}/{attempts} - model returned no schema-valid answer")

        return reasoning, None

    def _check_regex(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict | None]:
        rendered_system_prompt = self._render_system_prompt(system_prompt, data)
        if output_schema:
            rendered_system_prompt += self._get_json_schema_for_prompt(output_schema)

        response = self.model.invoke(self._conversation(rendered_system_prompt, query))
        return _reasoning_of(response), self._extract_json(str(response.content))

    def _extract_json(self, text: str) -> dict | None:
        pattern = rf"{re.escape(self.bos_json_clause)}\s*(\{{.*?\}})\s*{re.escape(self.eos_json_clause)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                assert isinstance(result, dict)
                return result
            except json.JSONDecodeError:
                logging.exception("[FAIR FORGE/JUDGE] JSON decode error")
                return None
        decoder = json.JSONDecoder()
        for start in (match.start() for match in re.finditer(r"\{", text)):
            try:
                result, _end = decoder.raw_decode(text[start:])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
        logging.error(f"[FAIR FORGE/JUDGE] No JSON found between {self.bos_json_clause} and {self.eos_json_clause}")
        return None

    def check_logprob_binary(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        positive_tokens: tuple[str, ...] = _DEFAULT_POSITIVE_TOKENS,
        negative_tokens: tuple[str, ...] = _DEFAULT_NEGATIVE_TOKENS,
        top_logprobs: int = 10,
        temperature: float | None = 1.0,
        scan_tokens: int = _DEFAULT_SCAN_TOKENS,
        extraction_retries: int = _DEFAULT_EXTRACTION_RETRIES,
    ) -> tuple[float, dict]:
        """Score a binary YES/NO judgment from the token where the model commits.

        Returns P(positive) / (P(positive) + P(negative)) aggregated across
        surface-form variants with log-sum-exp.

        Args:
            temperature: passed to model.bind() to control the answer-token
                distribution. Default 1.0 follows the paper — at lower values
                the distribution sharpens and the score loses calibration
                (only the YES/NO ranking survives). Pass None to inherit
                whatever temperature the underlying model was configured with.
            scan_tokens: how many generated positions to search for the answer.
                Compliance with "answer with one token" degrades as a prompt
                grows, so a long rubric can produce a short preamble on some
                calls and not others; scanning recovers those. Kept small so a
                reasoning trace fails rather than being searched for a stray
                token. Pass 1 to score the first generated token only.
            extraction_retries: how many times to re-ask when the answer holds no
                positive or negative token. A judge sampling at temperature 1.0
                occasionally answers off-format, and re-asking is a fresh draw, so
                one bad draw need not end a run. Only re-asked when the model did
                emit tokens: an answer carrying no logprobs at all is a capability
                limit that every attempt would hit identically. Pass 0 to disable.
                Note this relies on sampling — at temperature 0 the retry returns
                much the same answer.

        Requires a sufficiently capable model — see paper for AUC by model size.

        Raises:
            LogprobsNotSupportedError: provider returned an error when logprobs
                were requested.
            LogprobsExtractionError: no position within scan_tokens emitted a
                positive or negative token, or the response carried no logprobs.
        """
        rendered_system = self._render_system_prompt(system_prompt, data)
        bind_kwargs: dict[str, Any] = {"logprobs": True, "top_logprobs": top_logprobs}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        emitted: list = []
        attempts = 0
        for _ in range(extraction_retries + 1):
            attempts += 1
            content_entries = self._request_logprobs(rendered_system, query, bind_kwargs)
            top_lp = self._answer_logprobs(content_entries, positive_tokens + negative_tokens, scan_tokens)

            log_p_pos = self._aggregate_logprobs(top_lp, positive_tokens)
            log_p_neg = self._aggregate_logprobs(top_lp, negative_tokens)
            if log_p_pos > -math.inf or log_p_neg > -math.inf:
                score = 1.0 / (1.0 + math.exp(log_p_neg - log_p_pos))
                return score, {"top_logprobs": top_lp}

            emitted = [entry.get("token") for entry in content_entries[:scan_tokens]]
            if not emitted:
                break

        raise LogprobsExtractionError(
            f"Neither positive {positive_tokens} nor negative {negative_tokens} "
            f"tokens were emitted within the first {scan_tokens} generated tokens, "
            f"over {attempts} attempt(s). Emitted: {emitted}"
        )

    def _request_logprobs(self, rendered_system: str, query: str, bind_kwargs: dict) -> list:
        try:
            response = self.model.bind(**bind_kwargs).invoke(
                [
                    ("system", rendered_system),
                    ("human", query),
                ]
            )
        except Exception as e:
            raise LogprobsNotSupportedError(
                f"Provider {type(self.model).__name__} returned an error when logprobs were requested. "
                f"Use a provider that exposes logprobs or call Judge.check() instead."
            ) from e

        metadata: dict = getattr(response, "response_metadata", {}) or {}
        # A model that accepts the parameter and ignores it answers with
        # "logprobs": null, so the key is present and None — a two-argument get
        # would return None rather than its default and chaining would raise.
        return (metadata.get("logprobs") or {}).get("content") or []

    @staticmethod
    def _answer_logprobs(
        content_entries: list[dict[str, Any]],
        target_tokens: tuple[str, ...],
        scan_tokens: int,
    ) -> list[dict[str, Any]]:
        # Selected on the token actually sampled, not on a target appearing in
        # the distribution: a bare "Yes" sits in the top-N of almost any position
        # of prose, so the looser test would score off a word of preamble.
        for entry in content_entries[:scan_tokens]:
            if entry.get("token") in target_tokens:
                return list(entry.get("top_logprobs") or [])
        return []

    @staticmethod
    def _aggregate_logprobs(top_logprobs: list[dict[str, Any]], target_tokens: tuple[str, ...]) -> float:
        matches: list[float] = [
            float(entry["logprob"]) for entry in top_logprobs if entry.get("token") in target_tokens
        ]
        if not matches:
            return -math.inf
        max_lp = max(matches)
        return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in matches))


def _reasoning_of(message: Any) -> str:
    return str(getattr(message, "additional_kwargs", {}).get("reasoning_content", "") or "")
