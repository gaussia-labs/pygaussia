"""HuggingFace token-classification adapter.

Requires the ``privacy-huggingface`` extra; the imports below raise a clear
ImportError at module-import time when it is absent (FR-017).
"""

from typing import Any

from pydantic import PrivateAttr
from transformers import pipeline

from gaussia.core.detector import PIIDetector
from gaussia.detectors._label_canonicaliser import canonicalize, supported_classes_from_id2label
from gaussia.schemas.privacy import Span


class HuggingFacePIIDetector(PIIDetector):
    """Wraps a ``transformers`` token-classification pipeline, canonicalising its labels."""

    model_path: str
    device: int = -1
    aggregation_strategy: str = "simple"

    _pipeline: Any = PrivateAttr(default=None)

    def setup(self) -> None:
        self._ensure_pipeline()

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy,
            )
        return self._pipeline

    @property
    def supported_classes(self) -> frozenset[str]:
        pipe = self._ensure_pipeline()
        return supported_classes_from_id2label(pipe.model.config.id2label)

    def predict(self, text: str) -> list[Span]:
        pipe = self._ensure_pipeline()
        spans: list[Span] = []
        for entity in pipe(text):
            spans.append(
                Span(
                    label=canonicalize(entity["entity_group"]),
                    start=int(entity["start"]),
                    end=int(entity["end"]),
                    text=entity.get("word", ""),
                    score=float(entity["score"]),
                )
            )
        return spans
