"""Judge class for LLM-based evaluation."""

import json
import logging
import math
import re
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.factory import ProviderStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from gaussia.core.exceptions import LogprobsExtractionError, LogprobsNotSupportedError
from gaussia.utils.logging import VerboseLogger

T = TypeVar("T", bound=BaseModel)

_DEFAULT_POSITIVE_TOKENS: tuple[str, ...] = ("YES", "Yes", "yes", " YES", " Yes", " yes")
_DEFAULT_NEGATIVE_TOKENS: tuple[str, ...] = ("NO", "No", "no", " NO", " No", " no")


class Judge:
    """LLM-based judge for evaluating AI responses.

    Supports two modes:
    - Structured output: Uses create_agent with response_format for schema validation
    - Regex extraction: Parses JSON from model response using regex patterns

    Reasoning content is automatically extracted from LangChain's
    additional_kwargs.reasoning_content when available.

    Args:
        model: LangChain BaseChatModel instance
        use_structured_output: If True, use create_agent with response_format
        bos_json_clause: Opening marker for JSON block (default: ```json)
        eos_json_clause: Closing marker for JSON block (default: ```)
    """

    def __init__(
        self,
        model: BaseChatModel,
        use_structured_output: bool = False,
        strict: bool = True,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        verbose: bool = False,
    ):
        self.model = model
        self.use_structured_output = use_structured_output
        self.strict = strict
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.verbose = verbose
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
            - Uses create_agent with response_format for structured output
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

    def _escape_prompt_template(self, prompt: str) -> str:
        return prompt.replace("{", "{{").replace("}", "}}")

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
        agent = create_agent(
            model=self.model,
            response_format=ProviderStrategy(output_schema, strict=self.strict),
            system_prompt=rendered_system,
        )

        messages = [*self.chat_history, ("human", query)]
        self.chat_history.append(("human", query))

        max_retries = 5
        result = None
        for attempt in range(max_retries):
            try:
                result = agent.invoke({"messages": messages})
                parsed = result.get("structured_response")

                if parsed is None and attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} - model returned invalid JSON")
                    continue

                break
            except Exception as e:
                if "400" in str(e) and attempt < max_retries - 1:
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} after 400 error")
                    continue
                raise

        reasoning = ""
        if result:
            for msg in reversed(result.get("messages", [])):
                reasoning_content = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                if reasoning_content:
                    reasoning = reasoning_content
                    break

        return reasoning, result.get("structured_response") if result else None

    def _check_regex(
        self,
        system_prompt: str,
        query: str,
        data: dict,
        output_schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict | None]:
        rendered_system_prompt = self._render_system_prompt(system_prompt, data)
        if output_schema:
            schema_instruction = self._get_json_schema_for_prompt(output_schema)
            enhanced_prompt = rendered_system_prompt + schema_instruction
        else:
            enhanced_prompt = rendered_system_prompt

        escaped_prompt = self._escape_prompt_template(enhanced_prompt)

        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages([("system", escaped_prompt), *self.chat_history])
        chain = prompt | self.model
        response = chain.invoke({})
        content = str(response.content)
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        json_data = self._extract_json(content)
        return reasoning, json_data

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
    ) -> tuple[float, dict]:
        """Score a binary YES/NO judgment via first-token logprobs.

        Returns P(positive) / (P(positive) + P(negative)) aggregated across
        surface-form variants with log-sum-exp.

        Args:
            temperature: passed to model.bind() to control the first-token
                distribution. Default 1.0 follows the paper — at lower values
                the distribution sharpens and the score loses calibration
                (only the YES/NO ranking survives). Pass None to inherit
                whatever temperature the underlying model was configured with.

        Requires a sufficiently capable model — see paper for AUC by model size.

        Raises:
            LogprobsNotSupportedError: provider returned an error when logprobs
                were requested.
            LogprobsExtractionError: neither positive nor negative tokens appear
                in the top_logprobs of the first generated token.
        """
        rendered_system = self._render_system_prompt(system_prompt, data)
        bind_kwargs: dict[str, Any] = {"logprobs": True, "top_logprobs": top_logprobs}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        try:
            bound = self.model.bind(**bind_kwargs)
            response = bound.invoke(
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
        content_entries: list = metadata.get("logprobs", {}).get("content", []) or []
        top_lp: list[dict[str, Any]] = content_entries[0].get("top_logprobs", []) if content_entries else []

        log_p_pos = self._aggregate_logprobs(top_lp, positive_tokens)
        log_p_neg = self._aggregate_logprobs(top_lp, negative_tokens)

        if log_p_pos == -math.inf and log_p_neg == -math.inf:
            raise LogprobsExtractionError(
                f"Neither positive {positive_tokens} nor negative {negative_tokens} "
                f"tokens appeared in top_{top_logprobs}. Observed tokens: "
                f"{[entry.get('token') for entry in top_lp]}"
            )

        score = 1.0 / (1.0 + math.exp(log_p_neg - log_p_pos))
        return score, {"top_logprobs": top_lp}

    @staticmethod
    def _aggregate_logprobs(top_logprobs: list[dict[str, Any]], target_tokens: tuple[str, ...]) -> float:
        matches: list[float] = [
            float(entry["logprob"]) for entry in top_logprobs if entry.get("token") in target_tokens
        ]
        if not matches:
            return -math.inf
        max_lp = max(matches)
        return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in matches))
