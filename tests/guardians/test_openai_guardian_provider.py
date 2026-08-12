"""Tests for OpenAIGuardianProvider."""

import math

import pytest

from gaussia.guardians.llms.providers import OpenAIGuardianProvider
from gaussia.schemas.bias import LOGPROB_VERDICT_METHOD, SAMPLED_ANSWER_METHOD


def _chat_logprobs(*tokens: tuple[str, float]) -> dict:
    return {"content": [{"token": token, "logprob": logprob} for token, logprob in tokens]}


class TestOpenAIGuardianProviderParseResponse:
    """Tests for _parse_guardian_response error handling."""

    def _make_provider(self, unsafe_token: str = "BIASED", safe_token: str = "SAFE") -> OpenAIGuardianProvider:
        return OpenAIGuardianProvider(
            model="test-model",
            tokenizer=None,
            api_key="test",
            url="http://localhost",
            safe_token=safe_token,
            unsafe_token=unsafe_token,
        )

    def test_raises_on_api_error_response(self):
        provider = self._make_provider()
        error_response = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}

        with pytest.raises(RuntimeError, match="API error"):
            provider._parse_guardian_response(error_response)

    def test_raises_when_choices_missing(self):
        provider = self._make_provider()
        incomplete_response = {"id": "chatcmpl-123", "object": "chat.completion"}

        with pytest.raises(RuntimeError, match="API error"):
            provider._parse_guardian_response(incomplete_response)

    def test_parses_chat_completion_biased_response(self):
        provider = self._make_provider(unsafe_token="BIASED")
        response = {"choices": [{"message": {"content": "BIASED"}}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is True

    def test_parses_chat_completion_safe_response(self):
        provider = self._make_provider(unsafe_token="BIASED")
        response = {"choices": [{"message": {"content": "SAFE"}}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is False

    def test_parses_completion_biased_response(self):
        provider = self._make_provider(unsafe_token="BIASED")
        response = {"choices": [{"text": "BIASED"}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is True


class TestOpenAIGuardianProviderProbability:
    """The probability reported for a verdict, which reads as P(violation)."""

    def _make_provider(self, **kwargs) -> OpenAIGuardianProvider:
        return OpenAIGuardianProvider(
            model="test-model",
            tokenizer=None,
            api_key="test",
            url="http://localhost",
            safe_token="No",
            unsafe_token="Yes",
            **kwargs,
        )

    def test_absent_logprobs_report_no_probability(self):
        """Without a distribution there is no score — and no invented one."""
        provider = self._make_provider(logprobs=False)
        response = {"choices": [{"message": {"content": "Yes"}}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is True
        assert infer.probability is None
        assert infer.method == SAMPLED_ANSWER_METHOD

    def test_provider_that_ignores_the_flag_reports_no_probability(self):
        """A provider answering "logprobs": null is a capability limit, not a crash."""
        provider = self._make_provider(logprobs=True)
        response = {"choices": [{"message": {"content": "Yes"}, "logprobs": None}]}

        infer = provider._parse_guardian_response(response)

        assert infer.probability is None
        assert infer.method == SAMPLED_ANSWER_METHOD

    def test_null_content_reports_no_probability(self):
        provider = self._make_provider(logprobs=True)
        response = {"choices": [{"message": {"content": None}, "logprobs": _chat_logprobs(("Yes", -0.1))}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is False
        assert infer.probability is None
        assert infer.method is None

    def test_violation_probability_is_the_verdict_token_probability(self):
        provider = self._make_provider(logprobs=True)
        response = {"choices": [{"message": {"content": "Yes"}, "logprobs": _chat_logprobs(("Yes", math.log(0.8)))}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is True
        assert infer.probability == pytest.approx(0.8)
        assert infer.method == LOGPROB_VERDICT_METHOD

    def test_a_confident_safe_verdict_is_a_low_violation_probability(self):
        """P(emitted token) is inverted for a safe verdict, so the number means one thing."""
        provider = self._make_provider(logprobs=True)
        response = {"choices": [{"message": {"content": "No"}, "logprobs": _chat_logprobs(("No", math.log(0.95)))}]}

        infer = provider._parse_guardian_response(response)

        assert infer.is_bias is False
        assert infer.probability == pytest.approx(0.05)

    def test_scores_the_last_verdict_token_not_the_preamble(self):
        provider = self._make_provider(logprobs=True)
        response = {
            "choices": [
                {
                    "message": {"content": "Let me think. Yes"},
                    "logprobs": _chat_logprobs(
                        ("Let", math.log(0.3)), (" think", math.log(0.4)), ("Yes", math.log(0.7))
                    ),
                }
            ]
        }

        infer = provider._parse_guardian_response(response)

        assert infer.probability == pytest.approx(0.7)

    def test_reads_the_completions_endpoint_logprob_shape(self):
        provider = self._make_provider(logprobs=True)
        response = {
            "choices": [
                {
                    "text": "Yes",
                    "logprobs": {"tokens": ["Yes"], "token_logprobs": [math.log(0.6)]},
                }
            ]
        }

        infer = provider._parse_guardian_response(response)

        assert infer.probability == pytest.approx(0.6)
        assert infer.method == LOGPROB_VERDICT_METHOD

    def test_an_answer_without_a_verdict_token_reports_no_probability(self):
        provider = self._make_provider(logprobs=True)
        response = {
            "choices": [
                {
                    "message": {"content": "Yes"},
                    "logprobs": _chat_logprobs(("Maybe", math.log(0.5)), ("?", math.log(0.5))),
                }
            ]
        }

        infer = provider._parse_guardian_response(response)

        assert infer.probability is None
        assert infer.method == SAMPLED_ANSWER_METHOD
