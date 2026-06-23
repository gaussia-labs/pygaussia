"""Schema validation tests for the privacy metric (T006)."""

import pytest
from pydantic import ValidationError

from gaussia.schemas.privacy import (
    ClassMetrics,
    CriticalFNContribution,
    DetectionScoreContribution,
    PrivacyDomainConfig,
    PrivacyMetric,
    PrivacyRanking,
    Span,
    interpretation_for,
)


def _metric(name: str = "d", **overrides) -> PrivacyMetric:
    base: dict = {"session_id": "s", "assistant_id": "a", "name": name}
    base.update(overrides)
    return PrivacyMetric(**base)


class TestSpan:
    def test_valid_span(self):
        span = Span(label="email_address", start=0, end=5, text="a@b.c", score=0.9)
        assert span.start < span.end

    def test_start_not_less_than_end_rejected(self):
        with pytest.raises(ValidationError):
            Span(label="x", start=5, end=5, text="")

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            Span(label="x", start=-1, end=3, text="abc")

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Span(label="x", start=0, end=3, text="abc", score=1.5)

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            Span(label="", start=0, end=3, text="abc")


class TestPrivacyDomainConfig:
    def test_valid_config(self):
        cfg = PrivacyDomainConfig(
            classes=frozenset({"a", "b"}),
            criticality_weights={"a": 0.5, "b": 0.5},
            fn_severity_weights={"a": 0.3, "b": 0.7},
        )
        assert cfg.iou_threshold == 0.50

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(ValidationError):
            PrivacyDomainConfig(
                classes=frozenset({"a", "b"}),
                criticality_weights={"a": 0.5, "b": 0.4},
                fn_severity_weights={"a": 0.5, "b": 0.5},
            )

    def test_weight_keys_not_matching_classes_rejected(self):
        with pytest.raises(ValidationError):
            PrivacyDomainConfig(
                classes=frozenset({"a", "b"}),
                criticality_weights={"a": 1.0},
                fn_severity_weights={"a": 0.5, "b": 0.5},
            )

    def test_iou_threshold_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PrivacyDomainConfig(
                classes=frozenset({"a"}),
                criticality_weights={"a": 1.0},
                fn_severity_weights={"a": 1.0},
                iou_threshold=0.0,
            )

    def test_frozen(self):
        cfg = PrivacyDomainConfig(
            classes=frozenset({"a"}),
            criticality_weights={"a": 1.0},
            fn_severity_weights={"a": 1.0},
        )
        with pytest.raises(ValidationError):
            cfg.iou_threshold = 0.7

    def test_weight_maps_immutable(self):
        cfg = PrivacyDomainConfig(
            classes=frozenset({"a", "b"}),
            criticality_weights={"a": 0.5, "b": 0.5},
            fn_severity_weights={"a": 0.3, "b": 0.7},
        )
        with pytest.raises(TypeError):
            cfg.criticality_weights["a"] = 0.9
        with pytest.raises(TypeError):
            cfg.fn_severity_weights["a"] = 0.9

    def test_weight_maps_serialise_to_plain_dicts(self):
        cfg = PrivacyDomainConfig(
            classes=frozenset({"a", "b"}),
            criticality_weights={"a": 0.5, "b": 0.5},
            fn_severity_weights={"a": 0.3, "b": 0.7},
        )
        dumped = cfg.model_dump(mode="json")
        assert dumped["criticality_weights"] == {"a": 0.5, "b": 0.5}
        assert isinstance(dumped["fn_severity_weights"], dict)


class TestContributions:
    def test_detection_contribution_product_enforced(self):
        DetectionScoreContribution(weight=0.5, f2=0.4, contribution=0.2)
        with pytest.raises(ValidationError):
            DetectionScoreContribution(weight=0.5, f2=0.4, contribution=0.99)

    def test_critical_fn_contribution_product_enforced(self):
        CriticalFNContribution(severity_weight=0.5, fn_rate=0.4, contribution=0.2)
        with pytest.raises(ValidationError):
            CriticalFNContribution(severity_weight=0.5, fn_rate=0.4, contribution=0.99)


class TestPrivacyMetricValidators:
    def test_score_100_inconsistency_rejected(self):
        with pytest.raises(ValidationError):
            _metric(score=0.5, score_100=49.0, detection_score=0.5, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0)

    def test_score_product_inconsistency_rejected(self):
        with pytest.raises(ValidationError):
            _metric(score=0.99, score_100=99.0, detection_score=0.5, coverage=0.5, domain_fit=1.0, regulatory_fit=1.0)

    def test_r_final_inconsistency_rejected(self):
        with pytest.raises(ValidationError):
            _metric(r1_weakest_class_risk=0.5, r2_systemic_risk=0.5, r_final=0.0)

    def test_class_metrics_keys_must_match_taxonomy(self):
        cm = ClassMetrics(
            tp=1,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f2=1.0,
            fn_rate=0.0,
            criticality_weight=1.0,
            fn_severity_weight=1.0,
        )
        with pytest.raises(ValidationError):
            _metric(class_metrics={"a": cm}, covered_domain_classes=["b"], missing_domain_classes=[])

    def test_failed_metric_skips_consistency_checks(self):
        metric = _metric(
            name="broken",
            success=False,
            error="boom",
            score=0.0,
            score_100=0.0,
            detection_score=0.0,
            coverage=0.0,
        )
        assert metric.success is False
        assert metric.interpretation == "Not suitable"

    def test_interpretation_recomputed_from_score(self):
        metric = _metric(
            score=0.9,
            score_100=90.0,
            detection_score=0.9,
            coverage=1.0,
            domain_fit=1.0,
            regulatory_fit=1.0,
        )
        assert metric.interpretation == "Recommended"
        assert metric.r_final_100 == metric.r_final * 100.0


class TestPrivacyRanking:
    def test_descending_order_enforced(self):
        a = _metric(
            name="a", score=0.9, score_100=90.0, detection_score=0.9, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0
        )
        b = _metric(
            name="b", score=0.5, score_100=50.0, detection_score=0.5, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0
        )
        PrivacyRanking(session_id="s", assistant_id="a", results=[a, b], winning_detector="a", iou_threshold=0.5)
        with pytest.raises(ValidationError):
            PrivacyRanking(session_id="s", assistant_id="a", results=[b, a], winning_detector="b", iou_threshold=0.5)

    def test_failed_entries_must_be_at_tail(self):
        ok = _metric(
            name="ok", score=0.5, score_100=50.0, detection_score=0.5, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0
        )
        failed = _metric(name="bad", success=False, error="x")
        with pytest.raises(ValidationError):
            PrivacyRanking(
                session_id="s", assistant_id="a", results=[failed, ok], winning_detector=None, iou_threshold=0.5
            )

    def test_winner_must_match_top(self):
        ok = _metric(
            name="ok", score=0.5, score_100=50.0, detection_score=0.5, coverage=1.0, domain_fit=1.0, regulatory_fit=1.0
        )
        with pytest.raises(ValidationError):
            PrivacyRanking(session_id="s", assistant_id="a", results=[ok], winning_detector="wrong", iou_threshold=0.5)

    def test_all_failed_has_no_winner(self):
        failed = _metric(name="bad", success=False, error="x")
        ranking = PrivacyRanking(
            session_id="s", assistant_id="a", results=[failed], winning_detector=None, iou_threshold=0.5
        )
        assert ranking.winning_detector is None


def test_interpretation_bands():
    assert interpretation_for(0.0) == "Not suitable"
    assert interpretation_for(39.999) == "Not suitable"
    assert interpretation_for(40.0) == "Baseline / complement only"
    assert interpretation_for(60.0) == "Suitable with mitigations"
    assert interpretation_for(75.0) == "Operationally suitable"
    assert interpretation_for(85.0) == "Recommended"
    assert interpretation_for(95.0) == "Recommended with strong local evidence"
