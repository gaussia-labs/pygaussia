"""Ranking tests for PIIDetectorRanker (T012)."""

import pytest

from gaussia.metrics.privacy import PIIDetectorRanker
from gaussia.schemas.privacy import PIIDetectionRanking, PrivacyDomainConfig, Span
from tests.fixtures.privacy.corpus import batch, dataset, make_retriever
from tests.fixtures.privacy.stub_detector import StubDetector

EMAIL = "email_address"
PERSON = "person"
QUERY = "ranker turn"

GT = [Span(label=EMAIL, start=0, end=6, text="a@b.co"), Span(label=PERSON, start=10, end=20, text="John Smith")]


def _domain() -> PrivacyDomainConfig:
    return PrivacyDomainConfig(
        classes=frozenset({EMAIL, PERSON}),
        criticality_weights={EMAIL: 0.5, PERSON: 0.5},
        fn_severity_weights={EMAIL: 0.5, PERSON: 0.5},
    )


def _high() -> StubDetector:
    return StubDetector(
        name="high",
        domain_fit=1.0,
        regulatory_fit=1.0,
        classes_supported=frozenset({EMAIL, PERSON}),
        predictions={
            QUERY: [
                Span(label=EMAIL, start=0, end=6, text="a@b.co", score=1.0),
                Span(label=PERSON, start=10, end=20, text="John Smith", score=1.0),
            ]
        },
    )


def _mid() -> StubDetector:
    return StubDetector(
        name="mid",
        domain_fit=1.0,
        regulatory_fit=1.0,
        classes_supported=frozenset({EMAIL, PERSON}),
        predictions={QUERY: [Span(label=EMAIL, start=0, end=6, text="a@b.co", score=1.0)]},
    )


def _broken() -> StubDetector:
    return StubDetector(
        name="broken",
        domain_fit=1.0,
        regulatory_fit=1.0,
        classes_supported=frozenset({EMAIL, PERSON}),
        raises="down",
    )


def _run(detectors) -> PIIDetectionRanking:
    retriever = make_retriever([dataset("s", [batch("q", QUERY, GT)])])
    rankings = PIIDetectorRanker.run(retriever, detectors=detectors, domain_config=_domain())
    assert len(rankings) == 1
    return rankings[0]


def test_ordered_by_score_descending():
    ranking = _run([_mid(), _high()])
    assert [r.name for r in ranking.results] == ["high", "mid"]
    assert ranking.results[0].score_100 >= ranking.results[1].score_100
    assert ranking.winning_detector == "high"


def test_failed_detector_sorted_to_tail():
    ranking = _run([_high(), _broken(), _mid()])
    assert [r.name for r in ranking.results] == ["high", "mid", "broken"]
    assert ranking.results[-1].success is False
    assert ranking.results[-1].error is not None
    assert ranking.results[-1].score_100 == 0.0
    assert ranking.winning_detector == "high"


def test_all_failed_has_no_winner():
    ranking = _run([_broken(), _broken()])
    assert all(r.success is False for r in ranking.results)
    assert ranking.winning_detector is None


def test_ranking_echoes_domain_metadata():
    ranking = _run([_high()])
    assert ranking.iou_threshold == 0.5
    assert sorted(ranking.domain_classes) == [EMAIL, PERSON]


def test_setup_runs_once_per_detector_across_sessions():
    """Each detector is set up once for the whole run, not once per session:
    re-loading a heavy backend per session would be a latent perf bug."""
    high, mid = _high(), _mid()
    turn = batch("q", QUERY, GT)
    retriever = make_retriever([dataset("s1", [turn]), dataset("s2", [turn])])
    rankings = PIIDetectorRanker.run(retriever, detectors=[high, mid], domain_config=_domain())
    assert len(rankings) == 2
    assert high.setup_calls == 1
    assert mid.setup_calls == 1


def test_invalid_corpus_fails_hard_not_per_detector():
    """A corpus data error is not a detector failure: it must raise, not produce
    a ranking with every detector marked failed."""
    bad = batch("q", QUERY, [Span(label="credit_card", start=0, end=5, text="xxxxx")])
    retriever = make_retriever([dataset("s", [bad])])
    with pytest.raises(ValueError, match="outside the domain classes"):
        PIIDetectorRanker.run(retriever, detectors=[_high(), _mid()], domain_config=_domain())
