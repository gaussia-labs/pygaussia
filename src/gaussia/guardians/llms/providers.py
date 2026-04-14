import math
from functools import partial
from typing import Any

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from gaussia.schemas.bias import LLMGuardianProvider, LLMGuardianProviderInfer


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
            tokenizer=AutoTokenizer.from_pretrained(model),  # type: ignore[arg-type]
            api_key=api_key,
            url=url,
            temperature=temperature,
            safe_token=safe_token,
            unsafe_token=unsafe_token,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _parse_output(self, output: Any, input_len: int) -> tuple[bool, float]:
        import torch  # type: ignore[import-not-found]

        nlogprobs = 20
        is_bias, prob_of_bias = False, None

        if nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True) for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self._get_probabilities(list_index_logprobs_i)
                prob_of_bias = prob[1]

        res = self.tokenizer.decode(output.sequences[:, input_len:][0], skip_special_tokens=True).strip()  # type: ignore[attr-defined]

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
                decoded_token = self.tokenizer.convert_ids_to_tokens(index)  # type: ignore[attr-defined]
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
        return LLMGuardianProviderInfer(probability=prob_of_bias, is_bias=is_bias)


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
        self.chat_completions = "chat_completions" in kwargs

    def _parse_guardian_response(self, response_json):
        if "error" in response_json or "choices" not in response_json:
            raise RuntimeError(f"API error: {response_json.get('error', response_json)}")
        choice = response_json["choices"][0]
        prob_token = 1.0
        if "message" in choice:
            message_content = choice["message"]["content"]
            if message_content is None:
                return False, 1.0
            is_biased = self.unsafe_token in message_content
        else:
            is_biased = self.unsafe_token in choice["text"]

        if self.logprobs:
            logprobs = choice["logprobs"]
            prob_token = logprobs["token_logprobs"][0]
            prob_token = math.exp(prob_token)

        return is_biased, prob_token

    def _with_chat_completions(self, prompt: partial) -> dict[str, Any]:
        messages = [{"role": "user", "content": partial(prompt, tokenize=False)()}]
        response = requests.post(
            f"{self.url}/v1/chat/completions",
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
        return response.json()  # type: ignore[no-any-return]

    def _with_completions(self, prompt: partial) -> dict[str, Any]:
        response = requests.post(
            f"{self.url}/v1/completions",
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
        return response.json()  # type: ignore[no-any-return]

    def infer(self, prompt: partial) -> LLMGuardianProviderInfer:
        if self.chat_completions:
            response = self._with_chat_completions(prompt)
        else:
            response = self._with_completions(prompt)
        answer_token, prob_token = self._parse_guardian_response(response)
        return LLMGuardianProviderInfer(is_bias=answer_token, probability=prob_token)
