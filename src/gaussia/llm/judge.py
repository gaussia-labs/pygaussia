"""Judge class for LLM-based evaluation."""

import json
import logging
import re
from typing import TypeVar

from langchain.agents import create_agent
from langchain.agents.factory import ProviderStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from gaussia.utils.logging import VerboseLogger

T = TypeVar("T", bound=BaseModel)


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
        if output_schema:
            schema_instruction = self._get_json_schema_for_prompt(output_schema)
            # Escape JSON braces so LangChain doesn't treat them as template variables
            schema_instruction_escaped = schema_instruction.replace("{", "{{").replace("}", "}}")
            enhanced_prompt = system_prompt + schema_instruction_escaped
        else:
            enhanced_prompt = system_prompt

        self.chat_history.append(("human", query))
        prompt = ChatPromptTemplate.from_messages([("system", enhanced_prompt), *self.chat_history])
        chain = prompt | self.model
        response = chain.invoke(data)
        content = str(response.content)
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        json_data = self._extract_json(content)
        return reasoning, json_data

    def _extract_json(self, text: str) -> dict | None:
        # Primary: JSON within configured code fences
        pattern = rf"{re.escape(self.bos_json_clause)}\s*(\{{.*?\}})\s*{re.escape(self.eos_json_clause)}"
        match = re.search(pattern, text, re.DOTALL)
        candidate = match.group(1).strip() if match else None

        # Fallback: any JSON object in the response (models that skip code fences)
        if candidate is None:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            candidate = match.group(0).strip() if match else None

        if candidate is None:
            logging.error(f"[FAIR FORGE/JUDGE] No JSON found in response. Raw text: {text[:300]!r}")
            return None

        # Repair missing commas between fields (some models omit them)
        candidate = re.sub(r'(["\d\]truefals])\s*\n(\s*")', r'\1,\n\2', candidate)

        try:
            result = json.loads(candidate)
            assert isinstance(result, dict)
            return result
        except json.JSONDecodeError:
            logging.error(f"[FAIR FORGE/JUDGE] JSON decode error. Candidate: {candidate[:300]!r}")
            return None
