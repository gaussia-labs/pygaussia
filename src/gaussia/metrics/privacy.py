"""Domain-Adjusted Privacy Detection metric and ranker.

`Privacy` evaluates one injected `PIIDetector` against a labelled corpus and
emits one `PrivacyMetric` per dataset. `PrivacyRanker` evaluates several
detectors against the same corpus and emits a score-ordered `PrivacyRanking`.

The score and risk formulas are deterministic functions of corpus-wide class
counts (no `StatisticalMode`, FR-020). Latency is measured for diagnostics only
and never enters the score (FR-019).
"""

import time
from collections import defaultdict
from collections.abc import Iterable

from gaussia.core.base import Gaussia
from gaussia.core.detector import PIIDetector
from gaussia.core.retriever import Retriever
from gaussia.schemas.common import Batch
from gaussia.schemas.privacy import (
    ClassMetrics,
    CriticalFNContribution,
    DetectionScoreContribution,
    PrivacyBatch,
    PrivacyDomainConfig,
    PrivacyMetric,
    PrivacyRanking,
    Span,
)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    denominator = beta**2 * precision + recall
    if not denominator:
        return 0.0
    return (1 + beta**2) * precision * recall / denominator


def _spans_overlap(a: Span, b: Span) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)


def _span_iou(a: Span, b: Span) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    if not intersection:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union


def _suppress_overlaps(predictions: list[Span]) -> list[Span]:
    """Greedy non-maximum suppression: keep the highest-confidence of overlapping spans."""
    selected: list[Span] = []
    for pred in sorted(predictions, key=lambda s: (-(s.score or 0.0), s.start)):
        if not any(_spans_overlap(pred, kept) for kept in selected):
            selected.append(pred)
    return selected


class _Counts:
    __slots__ = ("fn", "fp", "tp")

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0


class _CorpusEvaluation:
    """Accumulates TP/FP/FN over a corpus for one detector and exposes diagnostics."""

    def __init__(self, config: PrivacyDomainConfig) -> None:
        self._config = config
        self._counts: dict[str, _Counts] = defaultdict(_Counts)
        self.corpus_samples = 0
        self.ground_truth_count = 0
        self.raw_prediction_count = 0
        self.prediction_count = 0
        self.out_of_domain_filtered = 0

    def add_turn(self, predictions: list[Span], ground_truth: list[Span]) -> None:
        self.corpus_samples += 1
        self.ground_truth_count += len(ground_truth)
        self.raw_prediction_count += len(predictions)

        in_domain = [p for p in predictions if p.label in self._config.classes]
        self.out_of_domain_filtered += len(predictions) - len(in_domain)
        kept = _suppress_overlaps(in_domain)
        self.prediction_count += len(kept)

        unmatched = set(range(len(ground_truth)))
        for pred in sorted(kept, key=lambda s: -(s.score or 0.0)):
            best_iou = 0.0
            best_gt = -1
            for gt_i in unmatched:
                gt = ground_truth[gt_i]
                if pred.label == gt.label:
                    iou = _span_iou(pred, gt)
                    if iou >= self._config.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt = gt_i
            if best_gt >= 0:
                unmatched.remove(best_gt)
                self._counts[ground_truth[best_gt].label].tp += 1
            else:
                self._counts[pred.label].fp += 1

        for gt_i in unmatched:
            self._counts[ground_truth[gt_i].label].fn += 1

    def class_metrics(self) -> dict[str, ClassMetrics]:
        metrics: dict[str, ClassMetrics] = {}
        for label in sorted(self._config.classes):
            counts = self._counts[label]
            precision = _safe_div(counts.tp, counts.tp + counts.fp)
            recall = _safe_div(counts.tp, counts.tp + counts.fn)
            metrics[label] = ClassMetrics(
                tp=counts.tp,
                fp=counts.fp,
                fn=counts.fn,
                precision=precision,
                recall=recall,
                f2=_f_beta(precision, recall),
                fn_rate=_safe_div(counts.fn, counts.tp + counts.fn),
                criticality_weight=self._config.criticality_weights[label],
                fn_severity_weight=self._config.fn_severity_weights[label],
            )
        return metrics


def _validate_corpus(conversation: Iterable[Batch], config: PrivacyDomainConfig) -> None:
    for turn in conversation:
        if not isinstance(turn, PrivacyBatch):
            raise TypeError("Privacy corpus turns must be PrivacyBatch carrying ground-truth spans")
        for span in turn.spans:
            if span.label not in config.classes:
                raise ValueError(f"ground-truth span label {span.label!r} is outside the domain classes")


def _evaluate(
    detector: PIIDetector,
    config: PrivacyDomainConfig,
    conversation: Iterable[Batch],
    session_id: str,
    assistant_id: str,
    load_time: float,
) -> PrivacyMetric:
    evaluation = _CorpusEvaluation(config)
    inference_latency = 0.0
    for turn in conversation:
        ground_truth = turn.spans if isinstance(turn, PrivacyBatch) else []
        start = time.perf_counter()
        predictions = detector.predict(turn.query)
        inference_latency += time.perf_counter() - start
        evaluation.add_turn(predictions, ground_truth)

    class_metrics = evaluation.class_metrics()

    detection_score = 0.0
    detection_contributions: dict[str, DetectionScoreContribution] = {}
    critical_fn = 0.0
    critical_contributions: dict[str, CriticalFNContribution] = {}
    r1 = 0.0
    r1_class: str | None = None
    for label in sorted(config.classes):
        cm = class_metrics[label]
        det_contrib = cm.criticality_weight * cm.f2
        detection_score += det_contrib
        detection_contributions[label] = DetectionScoreContribution(
            weight=cm.criticality_weight, f2=cm.f2, contribution=det_contrib
        )
        cfn_contrib = cm.fn_severity_weight * cm.fn_rate
        critical_fn += cfn_contrib
        critical_contributions[label] = CriticalFNContribution(
            severity_weight=cm.fn_severity_weight, fn_rate=cm.fn_rate, contribution=cfn_contrib
        )
        if cfn_contrib > r1:
            r1 = cfn_contrib
            r1_class = label

    supported = detector.supported_classes
    covered = sorted(supported & config.classes)
    missing = sorted(config.classes - supported)
    coverage = len(covered) / len(config.classes)

    penalty_fn = max(0.0, 1.0 - critical_fn)
    r2 = 1.0 - penalty_fn * coverage
    r_final = 1.0 - (1.0 - r1) * (1.0 - r2)
    score = detection_score * coverage * detector.domain_fit * detector.regulatory_fit * penalty_fn

    return PrivacyMetric(
        session_id=session_id,
        assistant_id=assistant_id,
        name=detector.name,
        load_time=load_time,
        inference_latency=inference_latency,
        corpus_samples=evaluation.corpus_samples,
        ground_truth_count=evaluation.ground_truth_count,
        raw_prediction_count=evaluation.raw_prediction_count,
        prediction_count=evaluation.prediction_count,
        out_of_domain_filtered=evaluation.out_of_domain_filtered,
        class_metrics=class_metrics,
        detection_score=detection_score,
        detection_score_contributions=detection_contributions,
        coverage=coverage,
        supported_classes=sorted(supported),
        covered_domain_classes=covered,
        missing_domain_classes=missing,
        domain_fit=detector.domain_fit,
        regulatory_fit=detector.regulatory_fit,
        critical_fn=critical_fn,
        penalty_fn=penalty_fn,
        critical_fn_contributions=critical_contributions,
        r1_weakest_class_risk=r1,
        r1_weakest_class=r1_class,
        r2_systemic_risk=r2,
        r_final=r_final,
        score=score,
        score_100=score * 100.0,
    )


def _timed_setup(detector: PIIDetector) -> float:
    start = time.perf_counter()
    detector.setup()
    return time.perf_counter() - start


def _reject_statistical_mode(kwargs: dict) -> None:
    # FR-020: the score/risk are deterministic functions of corpus-wide counts;
    # there is no per-batch distribution to aggregate, so no StatisticalMode applies.
    if "statistical_mode" in kwargs:
        raise TypeError("Privacy metrics do not accept 'statistical_mode' (FR-020): the score is deterministic")


class Privacy(Gaussia):
    """Evaluate a single `PIIDetector` against a labelled corpus (one `PrivacyMetric` per dataset)."""

    def __init__(
        self,
        retriever: type[Retriever],
        detector: PIIDetector,
        domain_config: PrivacyDomainConfig,
        **kwargs,
    ) -> None:
        _reject_statistical_mode(kwargs)
        super().__init__(retriever, **kwargs)
        self.detector = detector
        self.domain_config = domain_config
        self._load_time = _timed_setup(detector)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ) -> None:
        _validate_corpus(batch, self.domain_config)
        self.metrics.append(
            _evaluate(self.detector, self.domain_config, batch, session_id, assistant_id, self._load_time)
        )


class PrivacyRanker(Gaussia):
    """Rank several `PIIDetector`s against the same corpus (one `PrivacyRanking` per dataset)."""

    def __init__(
        self,
        retriever: type[Retriever],
        detectors: list[PIIDetector],
        domain_config: PrivacyDomainConfig,
        **kwargs,
    ) -> None:
        _reject_statistical_mode(kwargs)
        super().__init__(retriever, **kwargs)
        self.detectors = detectors
        self.domain_config = domain_config
        self._load_times: dict[int, float] = {}

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ) -> None:
        _validate_corpus(batch, self.domain_config)
        results: list[PrivacyMetric] = []
        for detector in self.detectors:
            results.append(self._evaluate_one(detector, batch, session_id, assistant_id))

        ranked = sorted(
            results,
            key=lambda m: (m.success, m.score_100),
            reverse=True,
        )
        winner = ranked[0].name if ranked and ranked[0].success else None
        self.metrics.append(
            PrivacyRanking(
                session_id=session_id,
                assistant_id=assistant_id,
                results=ranked,
                winning_detector=winner,
                iou_threshold=self.domain_config.iou_threshold,
                domain_classes=sorted(self.domain_config.classes),
                regulatory_framework=self.domain_config.regulatory_framework,
            )
        )

    def _setup_once(self, detector: PIIDetector) -> float:
        key = id(detector)
        if key not in self._load_times:
            self._load_times[key] = _timed_setup(detector)
        return self._load_times[key]

    def _evaluate_one(
        self, detector: PIIDetector, batch: list[Batch], session_id: str, assistant_id: str
    ) -> PrivacyMetric:
        try:
            load_time = self._setup_once(detector)
            return _evaluate(detector, self.domain_config, batch, session_id, assistant_id, load_time)
        except Exception as error:
            return PrivacyMetric(
                session_id=session_id,
                assistant_id=assistant_id,
                name=detector.name,
                success=False,
                error=str(error),
                domain_fit=detector.domain_fit,
                regulatory_fit=detector.regulatory_fit,
            )
