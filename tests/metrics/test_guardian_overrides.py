"""Tests for Guardian provider overrides and null content handling (Issue #2)."""

from unittest.mock import MagicMock, patch

from gaussia.guardians.llms.providers import OpenAIGuardianProvider
from gaussia.schemas.bias import GuardianLLMConfig


class TestGuardianLLMConfigOverrides:
    def test_guardian_llm_config_accepts_overrides(self):
        overrides = {"provider": {"ignore": ["DeepInfra"], "order": ["Together"]}}
        config = GuardianLLMConfig(
            model="meta-llama/llama-guard-4-12b",
            api_key="sk-test",
            url="https://openrouter.ai/api",
            temperature=0.0,
            provider=OpenAIGuardianProvider,
            overrides=overrides,
        )
        assert config.overrides == overrides

    def test_guardian_llm_config_overrides_defaults_to_empty(self):
        config = GuardianLLMConfig(
            model="meta-llama/llama-guard-4-12b",
            temperature=0.0,
            provider=OpenAIGuardianProvider,
        )
        assert config.overrides == {}


class TestOpenAIProviderOverrides:
    def _make_provider(self, overrides=None):
        mock_tokenizer = MagicMock()
        kwargs: dict = {
            "model": "meta-llama/llama-guard-4-12b",
            "tokenizer": mock_tokenizer,
            "api_key": "sk-test",
            "url": "https://openrouter.ai/api",
            "temperature": 0.0,
            "chat_completions": True,
        }
        if overrides is not None:
            kwargs["overrides"] = overrides
        return OpenAIGuardianProvider(**kwargs)

    def test_chat_completions_spreads_overrides_into_request_body(self):
        overrides = {"provider": {"ignore": ["DeepInfra"]}, "transforms": []}
        provider = self._make_provider(overrides=overrides)

        mock_prompt = MagicMock(return_value="<prompt>")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "safe"}}]
        }

        with patch("gaussia.guardians.llms.providers.requests.post", return_value=mock_response) as mock_post:
            provider._with_chat_completions(mock_prompt)

        posted_json = mock_post.call_args.kwargs["json"]
        assert "provider" in posted_json
        assert posted_json["provider"] == {"ignore": ["DeepInfra"]}
        assert "transforms" in posted_json

    def test_completions_spreads_overrides_into_request_body(self):
        overrides = {"provider": {"order": ["Together", "Fireworks"]}}
        provider = self._make_provider(overrides=overrides)

        mock_prompt = MagicMock(return_value="<prompt>")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "safe"}]
        }

        with patch("gaussia.guardians.llms.providers.requests.post", return_value=mock_response) as mock_post:
            provider._with_completions(mock_prompt)

        posted_json = mock_post.call_args.kwargs["json"]
        assert "provider" in posted_json
        assert posted_json["provider"] == {"order": ["Together", "Fireworks"]}

    def test_no_overrides_does_not_add_extra_keys(self):
        provider = self._make_provider()

        mock_prompt = MagicMock(return_value="<prompt>")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "safe"}}]
        }

        with patch("gaussia.guardians.llms.providers.requests.post", return_value=mock_response) as mock_post:
            provider._with_chat_completions(mock_prompt)

        posted_json = mock_post.call_args.kwargs["json"]
        expected_keys = {"model", "messages", "temperature", "max_tokens", "logprobs"}
        assert set(posted_json.keys()) == expected_keys


class TestParseGuardianResponseNullContent:
    def _make_provider(self):
        mock_tokenizer = MagicMock()
        return OpenAIGuardianProvider(
            model="meta-llama/llama-guard-4-12b",
            tokenizer=mock_tokenizer,
            api_key="sk-test",
            url="https://openrouter.ai/api",
            temperature=0.0,
            unsafe_token="unsafe",
            chat_completions=True,
        )

    def test_null_content_returns_not_biased(self):
        provider = self._make_provider()
        response = {"choices": [{"message": {"content": None}}]}
        is_biased, prob = provider._parse_guardian_response(response)
        assert is_biased is False
        assert prob == 1.0

    def test_valid_unsafe_content_returns_biased(self):
        provider = self._make_provider()
        response = {"choices": [{"message": {"content": "unsafe\nS1"}}]}
        is_biased, _prob = provider._parse_guardian_response(response)
        assert is_biased is True

    def test_valid_safe_content_returns_not_biased(self):
        provider = self._make_provider()
        response = {"choices": [{"message": {"content": "safe"}}]}
        is_biased, _prob = provider._parse_guardian_response(response)
        assert is_biased is False
