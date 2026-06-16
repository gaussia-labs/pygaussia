"""Open/Closed verification (SC-005, T014).

A brand-new detector backend plugs into Privacy and PrivacyRanker by subclassing
PIIDetector only — no edits to Privacy, PrivacyRanker, or any schema.
"""

from gaussia.core.detector import PIIDetector
from gaussia.metrics.privacy import Privacy, PrivacyRanker
from gaussia.schemas.privacy import PrivacyDomainConfig, Span
from tests.fixtures.privacy.corpus import batch, dataset, make_retriever

EMAIL = "email_address"
QUERY = "dummy turn"


class RegexEmailDetector(PIIDetector):
    """A trivial third backend implemented entirely outside the library."""

    @property
    def supported_classes(self) -> frozenset[str]:
        return frozenset({EMAIL})

    def predict(self, text: str) -> list[Span]:
        idx = text.find("@")
        if idx == -1:
            return []
        return [Span(label=EMAIL, start=0, end=len(text), text=text, score=1.0)]


def _domain() -> PrivacyDomainConfig:
    return PrivacyDomainConfig(
        classes=frozenset({EMAIL}),
        criticality_weights={EMAIL: 1.0},
        fn_severity_weights={EMAIL: 1.0},
    )


def test_dummy_detector_runs_in_privacy():
    detector = RegexEmailDetector(name="regex", domain_fit=1.0, regulatory_fit=1.0)
    gt = [Span(label=EMAIL, start=0, end=9, text="john@x.io")]
    retriever = make_retriever([dataset("s", [batch("q", "john@x.io", gt)])])
    metric = Privacy.run(retriever, detector=detector, domain_config=_domain())[0]
    assert metric.coverage == 1.0
    assert metric.class_metrics[EMAIL].tp == 1


def test_dummy_detector_runs_in_ranker():
    detector = RegexEmailDetector(name="regex", domain_fit=1.0, regulatory_fit=1.0)
    gt = [Span(label=EMAIL, start=0, end=9, text="john@x.io")]
    retriever = make_retriever([dataset("s", [batch("q", "john@x.io", gt)])])
    ranking = PrivacyRanker.run(retriever, detectors=[detector], domain_config=_domain())[0]
    assert ranking.winning_detector == "regex"
