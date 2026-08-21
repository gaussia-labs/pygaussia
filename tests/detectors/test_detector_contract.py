"""Contract tests for the abstract PIIDetector (T007)."""

import pytest
from pydantic import ValidationError

from gaussia.core.detector import PIIDetector
from gaussia.schemas.privacy import Span
from tests.fixtures.privacy.stub_detector import StubDetector


def test_cannot_instantiate_without_predict_and_supported_classes():
    class Incomplete(PIIDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete(name="x", domain_fit=0.5, regulatory_fit=0.5)


def test_missing_fits_raise_validation_error():
    with pytest.raises(ValidationError):
        StubDetector(name="x", classes_supported=frozenset({"a"}))


def test_fit_out_of_range_raises_validation_error():
    with pytest.raises(ValidationError):
        StubDetector(name="x", domain_fit=1.5, regulatory_fit=0.5, classes_supported=frozenset({"a"}))


def test_setup_is_noop_by_default():
    class Minimal(PIIDetector):
        @property
        def supported_classes(self) -> frozenset[str]:
            return frozenset()

        def predict(self, text: str) -> list[Span]:
            return []

    detector = Minimal(name="m", domain_fit=0.5, regulatory_fit=0.5)
    assert detector.setup() is None


def test_stub_predict_and_supported_classes():
    span = Span(label="a", start=0, end=2, text="ab", score=1.0)
    detector = StubDetector(
        name="stub",
        domain_fit=0.8,
        regulatory_fit=0.7,
        classes_supported=frozenset({"a", "b"}),
        predictions={"hello": [span]},
    )
    assert detector.supported_classes == frozenset({"a", "b"})
    assert detector.predict("hello") == [span]
    assert detector.predict("missing") == []
