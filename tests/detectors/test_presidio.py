"""Contract test for the Presidio adapter (T013). Skipped without the extra."""

import pytest

pytest.importorskip("presidio_analyzer")
pytest.importorskip("spacy")

from gaussia.detectors.presidio import PresidioDetector


@pytest.fixture(scope="module")
def detector() -> PresidioDetector:
    det = PresidioDetector(name="presidio", domain_fit=0.85, regulatory_fit=0.8)
    det.setup()
    return det


def test_supported_classes_are_domain_labels(detector: PresidioDetector):
    classes = detector.supported_classes
    assert "email_address" in classes
    assert all(label.islower() for label in classes)


def test_predicts_domain_projected_spans(detector: PresidioDetector):
    spans = detector.predict("Please email john@example.com about the invoice.")
    assert any(s.label == "email_address" for s in spans)
    for s in spans:
        assert s.start < s.end
        assert s.score is not None
