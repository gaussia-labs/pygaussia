"""Tests for OpenAIGuardianProvider."""

import pytest

from gaussia.guardians.llms.providers import OpenAIGuardianProvider


class TestOpenAIGuardianProviderParseResponse:
    """Tests for _parse_guardian_response error handling."""

    def _make_provider(self, unsafe_token: str = "BIASED") -> OpenAIGuardianProvider:
        return OpenAIGuardianProvider(
            model="test-model",
            tokenizer=None,
            api_key="test",
            url="http://localhost",
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

        is_biased, prob = provider._parse_guardian_response(response)

        assert is_biased is True
        assert prob == 1.0

    def test_parses_chat_completion_safe_response(self):
        provider = self._make_provider(unsafe_token="BIASED")
        response = {"choices": [{"message": {"content": "SAFE"}}]}

        is_biased, prob = provider._parse_guardian_response(response)

        assert is_biased is False
        assert prob == 1.0

    def test_parses_completion_biased_response(self):
        provider = self._make_provider(unsafe_token="BIASED")
        response = {"choices": [{"text": "BIASED"}]}

        is_biased, prob = provider._parse_guardian_response(response)

        assert is_biased is True
        assert prob == 1.0
