"""Tests for Judge module."""

import math
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
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

    def test_check_with_schema_in_regex_mode_escapes_schema_braces(self):
        """Test schema JSON braces are not treated as prompt variables."""
        model = FakeListChatModel(responses=['```json\n{"score": 0.7, "insight": "test"}\n```'])
        judge = Judge(model=model, use_structured_output=False)

        _thought, result = judge.check(
            "Context: {context}\nAssistant answer: {assistant_answer}",
            "Query",
            {"context": "retrieved context", "assistant_answer": "assistant response"},
            output_schema=ContextJudgeOutput,
        )

        assert result == {"score": 0.7, "insight": "test"}

    def test_extract_json_basic(self, mock_model):
        """Test _extract_json with basic JSON."""
        judge = Judge(model=mock_model)
        result = judge._extract_json('some text ```json\n{"key": "value"}\n``` more text')
        assert result == {"key": "value"}

    def test_extract_json_raw_object(self, mock_model):
        """Test _extract_json accepts raw JSON without fences."""
        judge = Judge(model=mock_model)
        result = judge._extract_json('{"score": 0.97, "insight": "ok"}')
        assert result == {"score": 0.97, "insight": "ok"}

    def test_extract_json_object_with_prefix(self, mock_model):
        """Test _extract_json accepts a JSON object embedded in prose."""
        judge = Judge(model=mock_model)
        result = judge._extract_json('Result: {"score": 0.97, "insight": "ok"}')
        assert result == {"score": 0.97, "insight": "ok"}

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


def _make_fake_model(provider_name: str) -> MagicMock:
    """Create a MagicMock whose class name matches a real provider class name."""
    fake_cls = type(provider_name, (MagicMock,), {})
    return fake_cls()


class TestJudgeLogprob:
    """Test suite for Judge.check_logprob_binary and its helpers."""

    def test_supports_logprobs_openai(self):
        model = _make_fake_model("ChatOpenAI")
        assert Judge._supports_logprobs(model) is True

    def test_supports_logprobs_anthropic_returns_false(self):
        model = _make_fake_model("ChatAnthropic")
        assert Judge._supports_logprobs(model) is False

    def test_supports_logprobs_unknown_returns_false(self):
        model = _make_fake_model("ChatSomethingNew")
        assert Judge._supports_logprobs(model) is False

    def test_check_logprob_binary_raises_when_unsupported(self):
        from gaussia.core.exceptions import LogprobsNotSupportedError

        model = _make_fake_model("ChatAnthropic")
        judge = Judge(model=model)

        with pytest.raises(LogprobsNotSupportedError):
            judge.check_logprob_binary("p", "q", {})

        model.bind.assert_not_called()
        model.invoke.assert_not_called()

    @staticmethod
    def _bind_response(model: MagicMock, top_logprobs_list: list[dict]) -> MagicMock:
        response = MagicMock()
        response.response_metadata = {"logprobs": {"content": [{"top_logprobs": top_logprobs_list}]}}
        bound = MagicMock()
        bound.invoke.return_value = response
        model.bind.return_value = bound
        return bound

    def test_check_logprob_binary_extracts_yes_score(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "Yes", "logprob": -0.1},
                {"token": "No", "logprob": -2.3},
                {"token": "Maybe", "logprob": -5.0},
            ],
        )
        judge = Judge(model=model)
        score, raw = judge.check_logprob_binary("p", "q", {})

        expected = 1.0 / (1.0 + math.exp(-2.3 - (-0.1)))
        assert abs(score - expected) < 1e-9
        assert abs(score - 0.9002) < 1e-3
        assert raw["top_logprobs"][0]["token"] == "Yes"

    def test_check_logprob_binary_aggregates_variants(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "YES", "logprob": -1.0},
                {"token": "Yes", "logprob": -1.0},
                {"token": "No", "logprob": -1.0},
            ],
        )
        judge = Judge(model=model)
        score, _ = judge.check_logprob_binary("p", "q", {})

        log_p_pos = -1.0 + math.log(2)
        log_p_neg = -1.0
        expected = 1.0 / (1.0 + math.exp(log_p_neg - log_p_pos))
        assert abs(score - expected) < 1e-9
        assert score > 0.5

    def test_check_logprob_binary_no_tokens_present_raises_extraction_error(self):
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "Okay", "logprob": -0.01},
                {"token": "Sure", "logprob": -3.0},
            ],
        )
        judge = Judge(model=model)
        with pytest.raises(LogprobsExtractionError):
            judge.check_logprob_binary("p", "q", {})

    def test_check_logprob_binary_one_side_missing(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "No", "logprob": -0.5},
                {"token": "Maybe", "logprob": -3.0},
            ],
        )
        judge = Judge(model=model)
        score, _ = judge.check_logprob_binary("p", "q", {})
        assert score == 0.0

    def test_check_logprob_binary_binds_correct_params(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "Yes", "logprob": -0.1},
                {"token": "No", "logprob": -2.0},
            ],
        )
        judge = Judge(model=model)
        judge.check_logprob_binary("p", "q", {})

        model.bind.assert_called_once_with(logprobs=True, top_logprobs=10, temperature=1.0)

    def test_check_logprob_binary_custom_temperature(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "Yes", "logprob": -0.1},
                {"token": "No", "logprob": -2.0},
            ],
        )
        judge = Judge(model=model)
        judge.check_logprob_binary("p", "q", {}, temperature=0.7)

        model.bind.assert_called_once_with(logprobs=True, top_logprobs=10, temperature=0.7)

    def test_check_logprob_binary_temperature_none_inherits_model_config(self):
        model = _make_fake_model("ChatOpenAI")
        self._bind_response(
            model,
            [
                {"token": "Yes", "logprob": -0.1},
                {"token": "No", "logprob": -2.0},
            ],
        )
        judge = Judge(model=model)
        judge.check_logprob_binary("p", "q", {}, temperature=None)

        model.bind.assert_called_once_with(logprobs=True, top_logprobs=10)

    def test_aggregate_logprobs_empty_returns_neg_inf(self):
        assert Judge._aggregate_logprobs([], ("YES",)) == -math.inf

    def test_aggregate_logprobs_logsumexp_correctness(self):
        result = Judge._aggregate_logprobs(
            [
                {"token": "A", "logprob": -1.0},
                {"token": "A", "logprob": -2.0},
            ],
            ("A",),
        )
        expected = math.log(math.exp(-1.0) + math.exp(-2.0))
        assert abs(result - expected) < 1e-9
        assert abs(result - (-0.6867)) < 1e-3
