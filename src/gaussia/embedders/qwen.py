"""Qwen3 embedding model implementation."""

import numpy as np
import torch  # type: ignore[import-not-found]
from torch.nn import functional as torch_functional  # type: ignore[import-not-found]
from transformers import AutoModel, AutoTokenizer

from gaussia.core.embedder import Embedder


class QwenEmbedder(Embedder):
    """Embedder backed by a Qwen3-Embedding model.

    Uses last-token pooling and instruction-prefixed query encoding.

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum token length for encoding.
        batch_size: Batch size for encoding.
        task: Instruction prefix used for query encoding via encode_query.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 8192,
        batch_size: int = 32,
        task: str = "Given a user query, retrieve relevant passages that answer the query",
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._task = task
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
                self._model_name,
                padding_side="left",
            )
        return self._tokenizer  # type: ignore[return-value]

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model.eval()  # type: ignore[union-attr]
        return self._model

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(  # type: ignore[operator]
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}  # type: ignore[attr-defined]

        with torch.no_grad():
            outputs = self.model(**batch)  # type: ignore[operator]
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"],
            )
            embeddings = torch_functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()  # type: ignore[no-any-return]

    def encode(self, sentences: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(sentences), self._batch_size):
            emb = self._encode_batch(sentences[i : i + self._batch_size])
            all_embeddings.append(emb)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0))

    def encode_query(self, sentences: list[str]) -> np.ndarray:
        prefixed = [f"Instruct: {self._task}\nQuery:{s}" for s in sentences]
        return self.encode(prefixed)
