"""Minimal, self-contained Privacy metric example.

Runs with zero extra dependencies: the detector below is a tiny regex stand-in so
you can see the full input -> output shape without installing Presidio or
transformers. Swap in a real adapter (see the commented line) once you do.

    uv run python examples/privacy/run.py
"""

import re

from gaussia.core.detector import PIIDetector
from gaussia.core.retriever import Retriever
from gaussia.metrics.privacy import Privacy, PrivacyRanker
from gaussia.schemas.common import Dataset
from gaussia.schemas.privacy import PrivacyBatch, PrivacyDomainConfig, Span

# 1. Describe the evaluation domain: which PII classes matter, how critical each
#    one is (criticality_weights, used by the score) and how costly a miss is
#    (fn_severity_weights, used by the risk index). Both maps must sum to 1.0.
DOMAIN = PrivacyDomainConfig(
    classes=frozenset({"email_address", "phone_number"}),
    criticality_weights={"email_address": 0.6, "phone_number": 0.4},
    fn_severity_weights={"email_address": 0.7, "phone_number": 0.3},
    iou_threshold=0.5,
    regulatory_framework="GDPR",
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\d{3}-\d{4}")


class RegexDetector(PIIDetector):
    """A trivial detector so the example runs without optional backends."""

    @property
    def supported_classes(self) -> frozenset[str]:
        return frozenset({"email_address", "phone_number"})

    def predict(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for match in _EMAIL.finditer(text):
            spans.append(Span(label="email_address", start=match.start(), end=match.end(), text=match.group(), score=0.99))
        for match in _PHONE.finditer(text):
            spans.append(Span(label="phone_number", start=match.start(), end=match.end(), text=match.group(), score=0.90))
        return spans


class EmailOnlyDetector(PIIDetector):
    """A weaker detector that only finds emails -> lower coverage and a phone blind spot."""

    @property
    def supported_classes(self) -> frozenset[str]:
        return frozenset({"email_address"})

    def predict(self, text: str) -> list[Span]:
        return [
            Span(label="email_address", start=m.start(), end=m.end(), text=m.group(), score=0.99)
            for m in _EMAIL.finditer(text)
        ]


class InMemoryRetriever(Retriever):
    """Yields one labelled conversation; the query is the text shown to the detector."""

    def load_dataset(self) -> list[Dataset]:
        text = "Reach me at john@example.com or 555-1234."
        turn = PrivacyBatch(
            qa_id="turn-1",
            query=text,
            assistant="",
            ground_truth_assistant="",
            spans=[
                Span(label="email_address", start=12, end=28, text="john@example.com"),
                Span(label="phone_number", start=32, end=40, text="555-1234"),
            ],
        )
        return [Dataset(session_id="demo", assistant_id="bot", context="", conversation=[turn])]


def evaluate_one() -> None:
    detector = RegexDetector(name="regex-baseline", domain_fit=0.9, regulatory_fit=0.8)
    # Real backend instead of the regex stand-in (needs `pip install gaussia[privacy-presidio]`):
    #   from gaussia.detectors.presidio import PresidioDetector
    #   detector = PresidioDetector(name="presidio", domain_fit=0.9, regulatory_fit=0.8)

    metrics = Privacy.run(InMemoryRetriever, detector=detector, domain_config=DOMAIN)
    metric = metrics[0]

    print("== Privacy: single detector ==")
    print(f"detector        : {metric.name}")
    print(f"score (0-100)   : {metric.score_100:.2f}  -> {metric.interpretation}")
    print(f"detection_score : {metric.detection_score:.3f}")
    print(f"coverage        : {metric.coverage:.3f}")
    print(f"penalty_fn      : {metric.penalty_fn:.3f}")
    print(f"risk (r_final)  : {metric.r_final:.3f}  weakest class: {metric.r1_weakest_class}")
    for label, cm in metric.class_metrics.items():
        print(f"  {label:<14} f2={cm.f2:.2f} (tp={cm.tp} fp={cm.fp} fn={cm.fn})")


def rank_several() -> None:
    good = RegexDetector(name="regex-baseline", domain_fit=0.9, regulatory_fit=0.8)
    weak = EmailOnlyDetector(name="email-only", domain_fit=0.5, regulatory_fit=0.5)

    ranking = PrivacyRanker.run(InMemoryRetriever, detectors=[good, weak], domain_config=DOMAIN)[0]
    print("\n== PrivacyRanker: choose the best detector ==")
    print(f"winner: {ranking.winning_detector}")
    for rank, result in enumerate(ranking.results, start=1):
        print(f"  {rank}. {result.name:<16} score={result.score_100:6.2f}  success={result.success}")


if __name__ == "__main__":
    evaluate_one()
    rank_several()
