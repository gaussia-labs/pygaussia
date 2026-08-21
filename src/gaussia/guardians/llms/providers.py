import math
from functools import partial
from typing import Any

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from gaussia.schemas.bias import (
    LOGPROB_SOFTMAX_METHOD,
    LOGPROB_VERDICT_METHOD,
    SAMPLED_ANSWER_METHOD,
    LLMGuardianProvider,
    LLMGuardianProviderInfer,
)


class HuggingFaceGuardianProvider(LLMGuardianProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        url: str | None = None,
        temperature: float = 0.0,
        safe_token: str = "Yes",
        unsafe_token: str = "No",
        max_tokens: int = 5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model),
            api_key=api_key,
            url=url,
            temperature=temperature,
            safe_token=safe_token,
            unsafe_token=unsafe_token,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _parse_output(self, output: Any, input_len: int) -> tuple[bool, float]:
        import torch

        nlogprobs = 20
        is_bias, prob_of_bias = False, None

        if nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True) for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self._get_probabilities(list_index_logprobs_i)
                prob_of_bias = prob[1]

        res = self.tokenizer.decode(output.sequences[:, input_len:][0], skip_special_tokens=True).strip()

        is_bias = self.unsafe_token.lower() in res.lower()

        if prob_of_bias is None:
            raise ValueError("Probabilities are not available. Please check the model output.")

        return is_bias, prob_of_bias.item()

    def _get_probabilities(self, logprobs: list) -> Any:
        import torch

        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50
        for gen_token_i in logprobs:
            for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0], strict=False):
                decoded_token = self.tokenizer.convert_ids_to_tokens(index)
                if self.safe_token.lower() in decoded_token.strip().lower():
                    safe_token_prob += math.exp(logprob)
                if self.unsafe_token.lower() in decoded_token.strip().lower():
                    unsafe_token_prob += math.exp(logprob)

        return torch.softmax(
            torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]),
            dim=0,
        )

    def infer(self, prompt: partial) -> LLMGuardianProviderInfer:
        import torch

        model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.bfloat16)
        prompt = partial(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        input_ids = prompt().to(model_device)
        input_len = input_ids.shape[1]
        model.eval()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        is_bias, prob_of_bias = self._parse_output(output, input_len)
        # _get_probabilities normalises the safe and unsafe token mass against each other,
        # so this is already P(violation) rather than P(the emitted token).
        return LLMGuardianProviderInfer(probability=prob_of_bias, is_bias=is_bias, method=LOGPROB_SOFTMAX_METHOD)


class OpenAIGuardianProvider(LLMGuardianProvider):
    def __init__(
        self,
        model: str,
        tokenizer: AutoTokenizer,
        api_key: str | None = None,
        url: str | None = None,
        temperature: float = 0.0,
        safe_token: str = "Yes",
        unsafe_token: str = "No",
        max_tokens: int = 5,
        logprobs: bool = False,
        overrides: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            model, tokenizer, api_key, url, temperature, safe_token, unsafe_token, max_tokens, logprobs, **kwargs
        )
        self._overrides: dict[str, Any] = overrides if overrides is not None else {}
        self.chat_completions = bool(kwargs.get("chat_completions", False))

    def _endpoint(self, path: str) -> str:
        base_url = (self.url or "").rstrip("/")
        versioned_path = path.lstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/{versioned_path}"
        return f"{base_url}/v1/{versioned_path}"

    def _parse_guardian_response(self, response_json: dict) -> LLMGuardianProviderInfer:
        if "error" in response_json or "choices" not in response_json:
            raise RuntimeError(f"API error: {response_json.get('error', response_json)}")
        choice = response_json["choices"][0]
        answer = self._answer_text(choice)
        if answer is None:
            # A model that spent its whole budget on a reasoning trace answers with no
            # content: there is no verdict in it and no distribution behind it.
            return LLMGuardianProviderInfer(is_bias=False, probability=None)

        is_biased = self.unsafe_token in answer
        probability, method = self._graded_verdict(choice, is_biased)
        return LLMGuardianProviderInfer(is_bias=is_biased, probability=probability, method=method)

    @staticmethod
    def _answer_text(choice: dict) -> str | None:
        if "message" in choice:
            content = choice["message"]["content"]
            return None if content is None else str(content)
        return str(choice["text"])

    def _graded_verdict(self, choice: dict, is_biased: bool) -> tuple[float | None, str]:
        logprob = self._verdict_logprob(choice)
        if logprob is None:
            return None, SAMPLED_ANSWER_METHOD
        # exp(logprob) is P(the token the model emitted), which is P(violation) only when
        # the verdict was a violation. Conditioning here means the number reads the same
        # way whichever way the verdict went.
        emitted = math.exp(logprob)
        return (emitted if is_biased else 1.0 - emitted), LOGPROB_VERDICT_METHOD

    def _verdict_logprob(self, choice: dict) -> float | None:
        # The last verdict-shaped token rather than position 0: a reasoning model spends
        # its opening tokens on a preamble, and scoring those describes another token
        # entirely. Whether any logprobs come back is a property of the serving provider
        # and not of the requested flag, so their absence is reported, never invented.
        for token, logprob in reversed(_token_logprobs(choice.get("logprobs"))):
            if self._is_verdict(token):
                return logprob
        return None

    def _is_verdict(self, token: str) -> bool:
        return token.strip().lower() in (self.safe_token.lower(), self.unsafe_token.lower())

    def _with_chat_completions(self, prompt: partial) -> dict[str, Any]:
        messages = [{"role": "user", "content": partial(prompt, tokenize=False)()}]
        response = requests.post(
            self._endpoint("chat/completions"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "logprobs": self.logprobs,
                **self._overrides,
            },
        )
        result: dict[str, Any] = response.json()
        return result

    def _with_completions(self, prompt: partial) -> dict[str, Any]:
        response = requests.post(
            self._endpoint("completions"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "prompt": partial(prompt, tokenize=False)(),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "logprobs": self.logprobs,
                **self._overrides,
            },
        )
        result: dict[str, Any] = response.json()
        return result

    def infer(self, prompt: partial) -> LLMGuardianProviderInfer:
        if self.chat_completions:
            response = self._with_chat_completions(prompt)
        else:
            response = self._with_completions(prompt)
        return self._parse_guardian_response(response)


def _token_logprobs(logprobs: Any) -> list[tuple[str, float]]:
    """Every (token, logprob) the answer carries, in generated order.

    Both OpenAI-compatible shapes are read: chat/completions nests one entry per token
    under "content", while the completions endpoint returns parallel "tokens" and
    "token_logprobs" lists. A provider that accepts the parameter and ignores it answers
    with "logprobs": null, which reads as no tokens rather than raising.
    """
    if not logprobs:
        return []
    entries = logprobs.get("content")
    if entries is not None:
        return [(str(entry.get("token", "")), float(entry["logprob"])) for entry in entries if "logprob" in entry]
    tokens = logprobs.get("tokens") or []
    values = logprobs.get("token_logprobs") or []
    return [(str(token), float(value)) for token, value in zip(tokens, values, strict=False) if value is not None]
