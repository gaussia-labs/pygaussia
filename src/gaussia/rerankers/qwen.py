"""Qwen3 reranker model implementation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gaussia.core.reranker import Reranker


class QwenReranker(Reranker):
    """Reranker backed by a Qwen3-Reranker model.

    Scores query-document pairs using yes/no log-probability from the model.

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum token length for scoring.
        instruction: Task instruction prepended to each query-document pair.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 8192,
        instruction: str = (
            "Given the agent response as the Query, determine whether the Document "
            "SUPPORTS (yes) or CONTRADICTS (no) the agent response."
        ),
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._instruction = instruction
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                padding_side="left",
            )
        return self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model.eval()
        return self._model

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {doc}"

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [self._format_pair(query, doc) for doc in documents]

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Query meets the requirements based on the Document "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        effective_max = self._max_length - len(prefix_tokens) - len(suffix_tokens)

        inputs_enc = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=effective_max,
        )
        for i, ids in enumerate(inputs_enc["input_ids"]):
            inputs_enc["input_ids"][i] = prefix_tokens + ids + suffix_tokens

        padded = self.tokenizer.pad(
            inputs_enc,
            padding=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        padded = {k: v.to(self.model.device) for k, v in padded.items()}

        with torch.no_grad():
            logits = self.model(**padded).logits[:, -1, :]
            true_v = logits[:, token_true_id]
            false_v = logits[:, token_false_id]
            log_probs = torch.nn.functional.log_softmax(
                torch.stack([false_v, true_v], dim=1),
                dim=1,
            )

        return log_probs[:, 1].exp().tolist()
