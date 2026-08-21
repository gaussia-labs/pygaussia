"""Deterministic PII detector for arithmetic tests (no external dependencies).

`StubDetector` returns predictions fixed by the test, keyed on the input text,
so every score component can be asserted against hand-computed values.
"""

from gaussia.core.detector import PIIDetector
from gaussia.schemas.privacy import Span


class StubDetector(PIIDetector):
    """A `PIIDetector` whose predictions and advertised classes are supplied by the test."""

    classes_supported: frozenset[str]
    predictions: dict[str, list[Span]] = {}
    raises: str | None = None
    setup_calls: int = 0

    @property
    def supported_classes(self) -> frozenset[str]:
        return self.classes_supported

    def predict(self, text: str) -> list[Span]:
        if self.raises is not None:
            raise RuntimeError(self.raises)
        return self.predictions.get(text, [])

    def setup(self) -> None:
        self.setup_calls += 1
