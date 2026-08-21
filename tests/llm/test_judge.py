"""Tests for Judge module."""

import math
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError

from gaussia.llm.judge import Judge
from gaussia.llm.schemas import ContextJudgeOutput
from gaussia.llm.structured import StructuredOutputStrategy


class MockResponseSchema(BaseModel):
    """Mock schema for testing structured output."""

    score: float = Field(ge=0, le=1)
    message: str


class _StubStructuredOutput(StructuredOutputStrategy):
    """Answers with a scripted sequence, so a retry can be observed without a provider.

    An entry that is an exception is raised, ``None`` stands for an answer that carried
    no parsed result, and any other value is returned as the parsed answer.
    """

    def __init__(self, answers: list, reasoning: str = ""):
        self._answers = list(answers)
        self._reasoning = reasoning

    def bind(self, model, schema):
        runnable = MagicMock()
        runnable.invoke.side_effect = lambda _messages: self._next()
        return runnable

    def _next(self) -> dict:
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        kwargs = {"reasoning_content": self._reasoning} if self._reasoning else {}
        return {"raw": AIMessage(content="", additional_kwargs=kwargs), "parsed": answer, "parsing_error": None}


def _answering(parsed, reasoning: str = "") -> _StubStructuredOutput:
    return _StubStructuredOutput([parsed], reasoning=reasoning)


def _answering_in_turn(answers: list) -> _StubStructuredOutput:
    return _StubStructuredOutput(answers)


def _schema_error() -> ValidationError:
    try:
        ContextJudgeOutput(score="not a number", insight="x")
    except ValidationError as error:
        return error
    raise AssertionError("expected the schema to reject the value")


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

    def test_check_regex_mode_valid_json(self, mock_model):
        """Test check method in regex mode with valid JSON."""
        mock_response = MagicMock()
        mock_response.content = 'Here is the result:\n```json\n{"score": 0.85, "valid": true}\n```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System prompt", "Query", {"key": "value"})

        assert thought == ""
        assert json_data == {"score": 0.85, "valid": True}

    def test_check_regex_mode_no_json_found(self, mock_model):
        """Test check method in regex mode when no JSON found."""
        mock_response = MagicMock()
        mock_response.content = "Response without JSON"
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    def test_check_regex_mode_invalid_json(self, mock_model):
        """Test check method in regex mode with invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "```json\n{invalid json}\n```"
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    def test_check_regex_mode_custom_json_clauses(self, mock_model):
        """Test check method with custom JSON clauses."""
        mock_response = MagicMock()
        mock_response.content = 'Result: <json>{"value": 42}</json>'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model, bos_json_clause="<json>", eos_json_clause="</json>")
        _thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"value": 42}

    def test_check_regex_mode_with_langchain_reasoning(self, mock_model):
        """Test check method extracts reasoning from LangChain's additional_kwargs."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"result": "done"}\n```'
        mock_response.additional_kwargs = {"reasoning_content": "Let me analyze this"}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        thought, json_data = judge.check("System", "Query", {})

        assert thought == "Let me analyze this"
        assert json_data == {"result": "done"}

    def test_check_structured_mode(self, mock_model):
        """Test check method in structured output mode."""
        expected_result = ContextJudgeOutput(score=0.9, insight="Good context")
        judge = Judge(model=mock_model, use_structured_output=True, structured_output=_answering(expected_result))

        thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert thought == ""
        assert result == expected_result

    def test_check_structured_mode_with_reasoning(self, mock_model):
        """Test check method extracts reasoning from additional_kwargs in structured mode."""
        expected_result = ContextJudgeOutput(score=0.9, insight="Good context")
        strategy = _answering(expected_result, reasoning="First I analyze. Then I evaluate.")
        judge = Judge(model=mock_model, use_structured_output=True, structured_output=strategy)

        thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert thought == "First I analyze. Then I evaluate."
        assert result == expected_result

    def test_check_structured_mode_binds_no_tools(self, mock_model):
        """The default strategy constrains generation instead of declaring an empty tool list."""
        judge = Judge(model=mock_model, use_structured_output=True)
        judge.structured_output.bind(mock_model, ContextJudgeOutput)

        mock_model.bind_tools.assert_not_called()
        _args, kwargs = mock_model.with_structured_output.call_args
        assert kwargs["method"] == "json_schema"
        assert kwargs["include_raw"] is True

    def test_check_structured_mode_retries_an_off_schema_answer(self, mock_model):
        """An answer that misses the schema is re-asked rather than returned as None."""
        expected_result = ContextJudgeOutput(score=0.4, insight="second attempt")
        strategy = _answering_in_turn([_schema_error(), expected_result])
        judge = Judge(model=mock_model, use_structured_output=True, structured_output=strategy)

        _thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert result == expected_result

    def test_check_structured_mode_raises_a_provider_refusal(self, mock_model):
        """A refused request is not re-sent: every attempt would be refused identically."""
        strategy = _answering_in_turn([RuntimeError("Error code: 400 - tools must not be empty")])
        judge = Judge(model=mock_model, use_structured_output=True, structured_output=strategy)

        with pytest.raises(RuntimeError, match="400"):
            judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

    def test_check_structured_mode_gives_up_after_the_retries(self, mock_model):
        """Exhausted retries report no result, which the caller surfaces as a failed judgment."""
        strategy = _answering_in_turn([None] * 5)
        judge = Judge(model=mock_model, use_structured_output=True, structured_output=strategy)

        _thought, result = judge.check("System", "Query", {}, output_schema=ContextJudgeOutput)

        assert result is None

    def test_check_structured_mode_fallback_to_regex(self, mock_model):
        """Test check falls back to regex when no schema provided in structured mode."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 0.5}\n```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model, use_structured_output=True)
        _thought, result = judge.check("System", "Query", {}, output_schema=None)

        assert result == {"score": 0.5}

    def test_chat_history_accumulates(self, mock_model):
        """Test that chat history accumulates across calls."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"result": 1}\n```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        assert len(judge.chat_history) == 0

        judge.check("System", "Query 1", {})
        assert len(judge.chat_history) == 1
        assert judge.chat_history[0] == ("human", "Query 1")

        judge.check("System", "Query 2", {})
        assert len(judge.chat_history) == 2
        assert judge.chat_history[1] == ("human", "Query 2")

    def test_check_json_with_whitespace(self, mock_model):
        """Test check handles JSON with extra whitespace."""
        mock_response = MagicMock()
        mock_response.content = '```json   \n  {"key": "value"}  \n  ```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        _thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"key": "value"}

    def test_check_nested_json(self, mock_model):
        """Test check handles nested JSON."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

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

    def test_check_with_schema_in_regex_mode(self, mock_model):
        """Test check appends schema to prompt in regex mode."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 0.7, "insight": "test"}\n```'
        mock_response.additional_kwargs = {}

        mock_model.invoke.return_value = mock_response

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

    def test_check_accepts_a_query_containing_braces(self):
        """A brace in the data under evaluation is content, not a prompt variable."""
        model = FakeListChatModel(responses=['```json\n{"score": 0.2, "insight": "test"}\n```'])
        judge = Judge(model=model, use_structured_output=False)

        _thought, result = judge.check(
            "Context: {context}",
            'Reply with {"time": "9am"} — when does it open?',
            {"context": "retrieved context"},
            output_schema=ContextJudgeOutput,
        )

        assert result == {"score": 0.2, "insight": "test"}

    def test_check_sends_the_rendered_system_prompt_and_the_query(self, mock_model):
        """The model receives rendered messages, in conversation order."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"score": 1.0}\n```'
        mock_response.additional_kwargs = {}
        mock_model.invoke.return_value = mock_response

        judge = Judge(model=mock_model)
        judge.check("Context: {context}", "Query", {"context": "retrieved context"})

        (messages,), _kwargs = mock_model.invoke.call_args
        assert messages == [("system", "Context: retrieved context"), ("human", "Query")]

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

    def test_check_logprob_binary_raises_when_provider_errors(self):
        from gaussia.core.exceptions import LogprobsNotSupportedError

        model = _make_fake_model("ChatAnthropic")
        bound = MagicMock()
        bound.invoke.side_effect = ValueError("logprobs not supported by this provider")
        model.bind.return_value = bound
        judge = Judge(model=model)

        with pytest.raises(LogprobsNotSupportedError):
            judge.check_logprob_binary("p", "q", {})

    @staticmethod
    def _bind_response(model: MagicMock, top_logprobs_list: list[dict]) -> MagicMock:
        sampled = max(top_logprobs_list, key=lambda entry: entry["logprob"])["token"]
        return TestJudgeLogprob._bind_positions(model, [(sampled, top_logprobs_list)])

    @staticmethod
    def _bind_positions(model: MagicMock, positions: list[tuple[str, list[dict]]]) -> MagicMock:
        """Bind a response carrying one entry per generated position.

        Each entry holds the token actually sampled there plus the distribution
        at that point, which is the shape providers return.
        """
        response = MagicMock()
        response.response_metadata = {
            "logprobs": {
                "content": [{"token": token, "top_logprobs": distribution} for token, distribution in positions]
            }
        }
        bound = MagicMock()
        bound.invoke.return_value = response
        model.bind.return_value = bound
        return bound

    @staticmethod
    def _bind_sequence(model: MagicMock, responses: list[dict]) -> MagicMock:
        """Bind one response per attempt, so a retry observes something different."""
        built = []
        for metadata in responses:
            response = MagicMock()
            response.response_metadata = metadata
            built.append(response)
        bound = MagicMock()
        bound.invoke.side_effect = built
        model.bind.return_value = bound
        return bound

    @staticmethod
    def _positions(*positions: tuple[str, list[dict]]) -> dict:
        return {
            "logprobs": {
                "content": [{"token": token, "top_logprobs": distribution} for token, distribution in positions]
            }
        }

    @staticmethod
    def _bind_metadata(model: MagicMock, metadata: dict) -> MagicMock:
        response = MagicMock()
        response.response_metadata = metadata
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

    def test_check_logprob_binary_scans_past_a_preamble(self):
        """The answer token is not always the first one generated.

        Asked with a long prompt, a model that had been answering with a bare
        token started prefixing a short preamble on some calls and not others.
        """
        model = _make_fake_model("ChatOpenAI")
        answer = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.3}]
        self._bind_positions(
            model,
            [
                ("Answer", [{"token": "Answer", "logprob": -0.2}]),
                (":", [{"token": ":", "logprob": -0.1}]),
                (" ", [{"token": " ", "logprob": -0.1}]),
                ("Yes", answer),
            ],
        )
        judge = Judge(model=model)

        score, raw = judge.check_logprob_binary("p", "q", {})

        assert score > 0.8
        assert raw["top_logprobs"] == answer

    def test_check_logprob_binary_ignores_a_candidate_it_did_not_sample(self):
        """A bare "Yes" sits in the top-N of almost any position of prose, so the
        position is chosen by what was sampled, not by what was merely available."""
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        self._bind_positions(
            model,
            [
                ("Certainly", [{"token": "Certainly", "logprob": -0.1}, {"token": "Yes", "logprob": -4.0}]),
                ("!", [{"token": "!", "logprob": -0.1}]),
            ],
        )
        judge = Judge(model=model)

        with pytest.raises(LogprobsExtractionError):
            judge.check_logprob_binary("p", "q", {})

    def test_check_logprob_binary_bounds_the_scan(self):
        """A reasoning trace must fail loudly rather than be searched to the end."""
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        preamble = [(f"word{i}", [{"token": f"word{i}", "logprob": -0.1}]) for i in range(10)]
        answer = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.0}]
        self._bind_positions(model, [*preamble, ("Yes", answer)])
        judge = Judge(model=model)

        with pytest.raises(LogprobsExtractionError):
            judge.check_logprob_binary("p", "q", {}, scan_tokens=4)

    def test_check_logprob_binary_retries_a_non_compliant_answer(self):
        """A judge sampling at temperature 1.0 occasionally answers off-format.

        Observed live: asked a yes/no question, a judge began enumerating the
        criteria instead of answering, so no yes/no token was emitted at all.
        Re-asking is a fresh draw, and one bad draw should not end a run.
        """
        model = _make_fake_model("ChatOpenAI")
        answer = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.3}]
        bound = self._bind_sequence(
            model,
            [
                self._positions(
                    (":", [{"token": ":", "logprob": -0.1}]), (" extra", [{"token": " extra", "logprob": -0.2}])
                ),
                self._positions(("Yes", answer)),
            ],
        )
        judge = Judge(model=model)

        score, _ = judge.check_logprob_binary("p", "q", {})

        assert score > 0.8
        assert bound.invoke.call_count == 2

    def test_check_logprob_binary_does_not_retry_when_no_logprobs_came_back(self):
        """An answer carrying no logprobs is a capability limit, not a bad draw.

        Retrying it would burn every attempt on every judgement of a whole run
        before failing with the same error.
        """
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        bound = self._bind_metadata(model, {"logprobs": None})
        judge = Judge(model=model)

        with pytest.raises(LogprobsExtractionError):
            judge.check_logprob_binary("p", "q", {})

        assert bound.invoke.call_count == 1

    def test_check_logprob_binary_gives_up_after_the_retries(self):
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        off_format = self._positions((":", [{"token": ":", "logprob": -0.1}]))
        bound = self._bind_sequence(model, [off_format, off_format, off_format])
        judge = Judge(model=model)

        with pytest.raises(LogprobsExtractionError, match="3 attempt"):
            judge.check_logprob_binary("p", "q", {})

        assert bound.invoke.call_count == 3

    def test_check_logprob_binary_retries_can_be_disabled(self):
        """extraction_retries=0 preserves the previous behaviour exactly."""
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        bound = self._bind_sequence(model, [self._positions((":", [{"token": ":", "logprob": -0.1}]))])
        judge = Judge(model=model)

        with pytest.raises(LogprobsExtractionError):
            judge.check_logprob_binary("p", "q", {}, extraction_retries=0)

        assert bound.invoke.call_count == 1

    def test_check_logprob_binary_handles_null_logprobs(self):
        """A model that accepts the parameter and ignores it answers with
        `"logprobs": null`, so the key is present and None."""
        from gaussia.core.exceptions import LogprobsExtractionError

        model = _make_fake_model("ChatOpenAI")
        self._bind_metadata(model, {"logprobs": None})
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
