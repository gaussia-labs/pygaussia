"""Tests for Judge module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from gaussia.llm.judge import Judge
from gaussia.llm.schemas import ContextJudgeOutput


class MockResponseSchema(BaseModel):
    """Mock schema for testing structured output."""

    score: float = Field(ge=0, le=1)
    message: str


class TestJudge:
    """Test suite for Judge class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock BaseChatModel."""
        return MagicMock()

    def test_initialization_default(self, mock_model):
        """Test Judge initialization with defaults."""
        judge = Judge(model=mock_model)
        assert judge.model == mock_model
        assert judge.use_structured_output is False
        assert judge.bos_json_clause == "```json"
        assert judge.eos_json_clause == "```"
        assert judge.chat_history == []

    def test_initialization_with_structured_output(self, mock_model):
        """Test Judge initialization with structured output enabled."""
        judge = Judge(model=mock_model, use_structured_output=True)
        assert judge.use_structured_output is True

    def test_initialization_custom_json_clauses(self, mock_model):
        """Test Judge initialization with custom JSON clauses."""
        judge = Judge(model=mock_model, bos_json_clause="<json>", eos_json_clause="</json>")
        assert judge.bos_json_clause == "<json>"
        assert judge.eos_json_clause == "</json>"

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_regex_mode_valid_json(self, mock_template, mock_model):
        """Test check method in regex mode with valid JSON."""
        mock_response = MagicMock()
        mock_response.content = 'Here is the result:\n```json\n{"score": 0.85, "valid": true}\n```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System prompt", "Query", {"key": "value"})

        assert thought == ""
        assert json_data == {"score": 0.85, "valid": True}

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_regex_mode_no_json_found(self, mock_template, mock_model):
        """Test check method in regex mode when no JSON found."""
        mock_response = MagicMock()
        mock_response.content = "Response without JSON"
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_regex_mode_invalid_json(self, mock_template, mock_model):
        """Test check method in regex mode with invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "```json\n{invalid json}\n```"
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_regex_mode_custom_json_clauses(self, mock_template, mock_model):
        """Test check method with custom JSON clauses."""
        mock_response = MagicMock()
        mock_response.content = 'Result: <json>{"value": 42}</json>'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model, bos_json_clause="<json>", eos_json_clause="</json>")
        _thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"value": 42}

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_regex_mode_with_langchain_reasoning(self, mock_template, mock_model):
        """Test check method extracts reasoning from LangChain's additional_kwargs."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"result": "done"}\n```'
        mock_response.additional_kwargs = {"reasoning_content": "Let me analyze this"}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == "Let me analyze this"
        assert json_data == {"result": "done"}

    @patch("gaussia.llm.judge.create_agent")
    def test_check_structured_mode(self, mock_create_agent, mock_model):
        """Test check method in structured output mode."""
        expected_result = ContextJudgeOutput(score=0.9, insight="Good context")

        mock_msg = MagicMock()
        mock_msg.additional_kwargs = {}

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "structured_response": expected_result,
            "messages": [mock_msg],
        }
        mock_create_agent.return_value = mock_agent

        judge = Judge(model=mock_model, use_structured_output=True)
        thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert thought == ""
        assert result == expected_result

    @patch("gaussia.llm.judge.create_agent")
    def test_check_structured_mode_with_reasoning(self, mock_create_agent, mock_model):
        """Test check method extracts reasoning from additional_kwargs in structured mode."""
        expected_result = ContextJudgeOutput(score=0.9, insight="Good context")

        mock_msg = MagicMock()
        mock_msg.additional_kwargs = {"reasoning_content": "First I analyze. Then I evaluate."}

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "structured_response": expected_result,
            "messages": [mock_msg],
        }
        mock_create_agent.return_value = mock_agent

        judge = Judge(model=mock_model, use_structured_output=True)
        thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert thought == "First I analyze. Then I evaluate."
        assert result == expected_result

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_structured_mode_fallback_to_regex(self, mock_template, mock_model):
        """Test check falls back to regex when no schema provided in structured mode."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 0.5}\n```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model, use_structured_output=True)
        _thought, result = judge.check("System", "Query", {}, output_schema=None)

        assert result == {"score": 0.5}

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_chat_history_accumulates(self, mock_template, mock_model):
        """Test that chat history accumulates across calls."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"result": 1}\n```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        assert len(judge.chat_history) == 0

        judge.check("System", "Query 1", {})
        assert len(judge.chat_history) == 1
        assert judge.chat_history[0] == ("human", "Query 1")

        judge.check("System", "Query 2", {})
        assert len(judge.chat_history) == 2
        assert judge.chat_history[1] == ("human", "Query 2")

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_json_with_whitespace(self, mock_template, mock_model):
        """Test check handles JSON with extra whitespace."""
        mock_response = MagicMock()
        mock_response.content = '```json   \n  {"key": "value"}  \n  ```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        _thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"key": "value"}

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_nested_json(self, mock_template, mock_model):
        """Test check handles nested JSON."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model)
        _thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"outer": {"inner": [1, 2, 3]}}

    def test_get_json_schema_for_prompt(self, mock_model):
        """Test JSON schema generation for prompt."""
        judge = Judge(model=mock_model)
        schema_str = judge._get_json_schema_for_prompt(ContextJudgeOutput)

        assert "score" in schema_str
        assert "insight" in schema_str
        assert "```json" in schema_str

    @patch("gaussia.llm.judge.ChatPromptTemplate")
    def test_check_with_schema_in_regex_mode(self, mock_template, mock_model):
        """Test check appends schema to prompt in regex mode."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 0.7, "insight": "test"}\n```'
        mock_response.additional_kwargs = {}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_template.from_messages.return_value = mock_prompt

        judge = Judge(model=mock_model, use_structured_output=False)
        _thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert result == {"score": 0.7, "insight": "test"}

    def test_extract_json_basic(self, mock_model):
        """Test _extract_json with basic JSON."""
        judge = Judge(model=mock_model)
        result = judge._extract_json('some text ```json\n{"key": "value"}\n``` more text')
        assert result == {"key": "value"}

    def test_extract_json_not_found(self, mock_model):
        """Test _extract_json when no JSON found."""
        judge = Judge(model=mock_model)
        result = judge._extract_json("no json here")
        assert result is None

    def test_extract_json_invalid(self, mock_model):
        """Test _extract_json with invalid JSON."""
        judge = Judge(model=mock_model)
        result = judge._extract_json("```json\n{invalid}\n```")
        assert result is None
