"""Contract test for the HuggingFace adapter (T013). Skipped without the extra.

The model is faked so the test asserts the id2label -> canonical-label projection
without downloading a real checkpoint.
"""

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from gaussia.detectors.huggingface import HuggingFacePIIDetector


class _FakeConfig:
    id2label = {0: "O", 1: "B-EMAIL", 2: "I-PERSON", 3: "B-PHONE"}


class _FakeModel:
    config = _FakeConfig()


class _FakePipeline:
    model = _FakeModel()

    def __init__(self, entities):
        self._entities = entities

    def __call__(self, text):
        return self._entities


def _make(entities=None) -> HuggingFacePIIDetector:
    detector = HuggingFacePIIDetector(
        name="hf", model_path="fake/model", domain_fit=0.7, regulatory_fit=0.6
    )
    detector._pipeline = _FakePipeline(entities or [])
    return detector


def test_supported_classes_canonicalised_and_filtered():
    detector = _make()
    # "O" dropped; B-EMAIL -> email_address; I-PERSON -> person; B-PHONE -> phone -> phone_number (alias)
    assert detector.supported_classes == frozenset({"email_address", "person", "phone_number"})


def test_predict_canonicalises_entity_labels():
    detector = _make([
        {"entity_group": "EMAIL", "start": 0, "end": 9, "word": "john@x.io", "score": 0.95},
    ])
    spans = detector.predict("john@x.io")
    assert len(spans) == 1
    assert spans[0].label == "email_address"
    assert spans[0].score == pytest.approx(0.95)
