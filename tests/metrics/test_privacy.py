"""Accounting and score-pipeline tests for the Privacy metric (T009-T011).

All expected values are computed by hand from the paper's formulas using a
deterministic StubDetector, per SC-001.
"""

import pytest
from pydantic import ValidationError

from gaussia.metrics.privacy import Privacy
from gaussia.schemas.privacy import PrivacyDomainConfig, PrivacyMetric, Span, interpretation_for
from tests.fixtures.privacy.corpus import batch, dataset, make_retriever
from tests.fixtures.privacy.stub_detector import StubDetector

EMAIL = "email_address"
PERSON = "person"
PHONE = "phone_number"
SSN = "us_ssn"


def _domain() -> PrivacyDomainConfig:
    return PrivacyDomainConfig(
        classes=frozenset({EMAIL, PERSON, PHONE, SSN}),
        criticality_weights={EMAIL: 0.4, PERSON: 0.3, PHONE: 0.2, SSN: 0.1},
        fn_severity_weights={EMAIL: 0.5, PERSON: 0.1, PHONE: 0.3, SSN: 0.1},
        iou_threshold=0.5,
        regulatory_framework="GDPR",
    )


def _span(label: str, start: int, end: int, score: float | None = None) -> Span:
    return Span(label=label, start=start, end=end, text="x" * (end - start), score=score)


# Shared scenario: one corpus turn exercising TP (exact + partial), FP+FN, OOD, NMS, 0/0 guard.
QUERY = "scenario turn one"

GT_SPANS = [
    _span(PERSON, 8, 18),
    _span(EMAIL, 22, 31),
    _span(PHONE, 35, 43),
]

PRED_SPANS = [
    _span(PERSON, 8, 18, score=0.9),      # IoU 1.0 -> TP person
    _span(PERSON, 8, 16, score=0.5),      # overlaps higher-score person -> dropped by NMS
    _span(EMAIL, 22, 28, score=0.8),      # IoU 6/9 = 0.667 >= 0.5 -> TP email
    _span(PHONE, 40, 43, score=0.7),      # IoU 3/8 = 0.375 < 0.5 -> FP phone, GT phone -> FN
    _span("credit_card", 0, 5, score=0.95),  # out of domain -> filtered, not FP
]


def _run_shared(domain_fit: float = 1.0, regulatory_fit: float = 1.0) -> PrivacyMetric:
    detector = StubDetector(
        name="stub",
        domain_fit=domain_fit,
        regulatory_fit=regulatory_fit,
        classes_supported=frozenset({EMAIL, PERSON, PHONE, SSN}),
        predictions={QUERY: PRED_SPANS},
    )
    retriever = make_retriever([dataset("s1", [batch("q1", QUERY, GT_SPANS)])])
    metrics = Privacy.run(retriever, detector=detector, domain_config=_domain())
    assert len(metrics) == 1
    return metrics[0]


class TestAccounting:
    def test_confusion_counts(self):
        m = _run_shared()
        assert (m.class_metrics[PERSON].tp, m.class_metrics[PERSON].fp, m.class_metrics[PERSON].fn) == (1, 0, 0)
        assert (m.class_metrics[EMAIL].tp, m.class_metrics[EMAIL].fp, m.class_metrics[EMAIL].fn) == (1, 0, 0)
        assert (m.class_metrics[PHONE].tp, m.class_metrics[PHONE].fp, m.class_metrics[PHONE].fn) == (0, 1, 1)

    def test_rates(self):
        m = _run_shared()
        assert m.class_metrics[EMAIL].precision == pytest.approx(1.0)
        assert m.class_metrics[EMAIL].recall == pytest.approx(1.0)
        assert m.class_metrics[EMAIL].f2 == pytest.approx(1.0)
        assert m.class_metrics[PHONE].precision == pytest.approx(0.0)
        assert m.class_metrics[PHONE].recall == pytest.approx(0.0)
        assert m.class_metrics[PHONE].f2 == pytest.approx(0.0)
        assert m.class_metrics[PHONE].fn_rate == pytest.approx(1.0)

    def test_zero_ground_truth_class_keeps_full_weight(self):
        m = _run_shared()
        assert m.class_metrics[SSN].tp == 0
        assert m.class_metrics[SSN].f2 == pytest.approx(0.0)
        assert m.class_metrics[SSN].fn_rate == pytest.approx(0.0)
        assert m.detection_score_contributions[SSN].weight == pytest.approx(0.1)
        assert m.detection_score_contributions[SSN].contribution == pytest.approx(0.0)

    def test_out_of_domain_filtered_not_counted_as_fp(self):
        m = _run_shared()
        assert m.out_of_domain_filtered == 1
        assert sum(cm.fp for cm in m.class_metrics.values()) == 1  # only the phone FP

    def test_nms_dropped_overlapping_prediction(self):
        m = _run_shared()
        assert m.raw_prediction_count == 5
        assert m.prediction_count == 3  # OOD removed (4), then NMS removes the lower-score person (3)

    def test_corpus_diagnostics(self):
        m = _run_shared()
        assert m.corpus_samples == 1
        assert m.ground_truth_count == 3


class TestScorePipeline:
    def test_detection_score(self):
        m = _run_shared()
        # 0.4*1(email) + 0.3*1(person) + 0.2*0(phone) + 0.1*0(ssn) = 0.7
        assert m.detection_score == pytest.approx(0.7)

    def test_coverage(self):
        m = _run_shared()
        assert m.coverage == pytest.approx(1.0)
        assert m.missing_domain_classes == []

    def test_penalty_and_risk(self):
        m = _run_shared()
        # critical_fn = 0.3*1 (phone) = 0.3 -> penalty 0.7
        assert m.critical_fn == pytest.approx(0.3)
        assert m.penalty_fn == pytest.approx(0.7)
        assert m.r1_weakest_class_risk == pytest.approx(0.3)
        assert m.r1_weakest_class == PHONE
        assert m.r2_systemic_risk == pytest.approx(0.3)  # 1 - 0.7*1.0
        assert m.r_final == pytest.approx(0.51)  # 1 - 0.7*0.7

    def test_final_score(self):
        m = _run_shared(domain_fit=1.0, regulatory_fit=1.0)
        # 0.7 * 1.0(cov) * 1.0 * 1.0 * 0.7(penalty) = 0.49
        assert m.score == pytest.approx(0.49)
        assert m.score_100 == pytest.approx(49.0)
        assert m.interpretation == "Baseline / complement only"

    def test_fits_passed_through(self):
        m = _run_shared(domain_fit=0.85, regulatory_fit=0.8)
        assert m.domain_fit == pytest.approx(0.85)
        assert m.regulatory_fit == pytest.approx(0.8)
        assert m.score == pytest.approx(0.7 * 1.0 * 0.85 * 0.8 * 0.7)


class TestPerfectDetector:
    """Acceptance scenario 1: a detector covering all classes with perfect predictions."""

    def test_perfect(self):
        q1, q2, q3 = "a@b.co", "John Smith", "555-1234"
        gt = {
            q1: [_span(EMAIL, 0, 6)],
            q2: [_span(PERSON, 0, 10)],
            q3: [_span(PHONE, 0, 8)],
        }
        preds = {
            q1: [_span(EMAIL, 0, 6, 1.0)],
            q2: [_span(PERSON, 0, 10, 1.0)],
            q3: [_span(PHONE, 0, 8, 1.0)],
        }
        # us_ssn has no ground truth; perfect on the three present classes.
        domain = PrivacyDomainConfig(
            classes=frozenset({EMAIL, PERSON, PHONE}),
            criticality_weights={EMAIL: 0.5, PERSON: 0.3, PHONE: 0.2},
            fn_severity_weights={EMAIL: 0.5, PERSON: 0.3, PHONE: 0.2},
        )
        detector = StubDetector(
            name="perfect",
            domain_fit=0.85,
            regulatory_fit=0.8,
            classes_supported=frozenset({EMAIL, PERSON, PHONE}),
            predictions=preds,
        )
        conv = [batch(q, q, gt[q]) for q in (q1, q2, q3)]
        retriever = make_retriever([dataset("s", conv)])
        m = Privacy.run(retriever, detector=detector, domain_config=domain)[0]
        assert m.detection_score == pytest.approx(1.0)
        assert m.coverage == pytest.approx(1.0)
        assert m.penalty_fn == pytest.approx(1.0)
        assert m.score == pytest.approx(0.85 * 0.8)
        assert m.r_final == pytest.approx(0.0)
        assert m.r1_weakest_class is None


class TestLatencyAndExclusions:
    def test_latency_fields_present_and_non_negative(self):
        m = _run_shared()
        assert m.load_time >= 0.0
        assert m.inference_latency >= 0.0

    def test_no_statistical_mode_parameter(self):
        detector = StubDetector(
            name="s", domain_fit=1.0, regulatory_fit=1.0,
            classes_supported=frozenset({EMAIL, PERSON, PHONE, SSN}),
            predictions={QUERY: PRED_SPANS},
        )
        retriever = make_retriever([dataset("s1", [batch("q1", QUERY, GT_SPANS)])])
        with pytest.raises(TypeError):
            Privacy.run(retriever, detector=detector, domain_config=_domain(), statistical_mode="frequentist")

    def test_detector_exception_propagates(self):
        detector = StubDetector(
            name="boom", domain_fit=1.0, regulatory_fit=1.0,
            classes_supported=frozenset({EMAIL, PERSON, PHONE, SSN}),
            raises="backend exploded",
        )
        retriever = make_retriever([dataset("s1", [batch("q1", QUERY, GT_SPANS)])])
        with pytest.raises(RuntimeError, match="backend exploded"):
            Privacy.run(retriever, detector=detector, domain_config=_domain())


def test_paper_worked_example_uses_five_factor_formula():
    """The paper cites Score_100 = 28.36, but that figure still included the
    InfraScore (~0.86) removed in v3. With the five-factor formula the same
    component values yield 32.97; FR-019 forbids latency/infra from the score.
    """
    det, cov, dfit, rfit, pen = 0.8231, 0.7273, 0.85, 0.80, 0.81
    score = det * cov * dfit * rfit * pen
    metric = PrivacyMetric(
        session_id="s", assistant_id="a", name="presidio-healthcare",
        detection_score=det, coverage=cov, domain_fit=dfit, regulatory_fit=rfit,
        penalty_fn=pen, score=score, score_100=score * 100.0,
    )
    assert metric.score_100 == pytest.approx(32.97, abs=0.01)
    assert metric.score_100 != pytest.approx(28.36, abs=0.01)
    assert interpretation_for(metric.score_100) == "Not suitable"


def test_score_validator_catches_infra_factor():
    # Sanity: constructing a metric whose score smuggles in a sixth factor fails.
    with pytest.raises(ValidationError):
        PrivacyMetric(
            session_id="s", assistant_id="a", name="x",
            detection_score=1.0, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0,
            penalty_fn=1.0, score=0.86, score_100=86.0,
        )
