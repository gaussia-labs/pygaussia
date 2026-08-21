"""The output boundary: Roast Dataset records into the framework's dataset shapes (FR-034).

This is the only module of the subsystem where the framework's ``Batch`` and ``Dataset``
appear. Everything upstream of it computes; this converts, so that an existing metric consumes
a run unmodified.

The framework's models require fields a trap has no natural value for, and `data-model.md`
states what each one gets rather than leaving it to the conversion: the expected answer is
``""`` because a trap has no correct answer and inventing one would let a metric score against
it, the turn weight is left unset because it is the framework's aggregation weight and has
nothing to do with principle weights, and the record itself travels on ``RoastBatch.roast``
because ``Batch`` has no free-form slot.

``evidence_available`` crosses on the record rather than being inferred from ``evidence``.
Absent evidence and evidence sought and not found are different findings, and a single
``None`` would collapse them into one (FR-015, US2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gaussia.schemas.common import Dataset
from gaussia.schemas.roastme import RecordProvenance, RoastBatch, RoastDatasetRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from gaussia.schemas.roastme import CategoryEvaluation, FailureReport, GradedOutcome, PrincipleGrade, Probe

DEFAULT_LANGUAGE = "english"

EVIDENCE_META_KEY = "evidence"
"""Where a probe engine records the supporting or contradicting source text, when it has one.

A probe generated with no knowledge base has none, which is what makes ``evidence`` ``None``
in black-box mode; a knowledge-grounded probe whose engine recorded none is the other case,
and ``evidence_available`` is what keeps the two apart.
"""


def charged_principles(grades: Sequence[PrincipleGrade]) -> list[str]:
    """The principles a response is charged with: those its grades score above zero.

    Shared with the Exploiter, which builds records from generated queries that have no probe
    behind them. The charging rule has to be one function, or the same response would be
    charged differently depending on which component wrote the record.
    """
    return [grade.principle for grade in grades if grade.score > 0.0]


def grading_methods(grades: Iterable[PrincipleGrade]) -> dict[str, int]:
    """How many grades each judging method produced, counted over a whole run.

    ``PrincipleGrade.method`` already says how one verdict was reached, and one at a time is not the
    scale the question is asked at. The shipped grader reads a continuous probability out of the
    verdict token's distribution in a single call, and falls back to voting across ``k`` samples
    when it cannot — two estimators, one of them able to land only on multiples of ``1/k``. A rate
    that averages over both is an average over two different measurements, and until this existed
    the only way to notice was to walk every grade of every outcome by hand.

    Counted rather than flagged, because the two readings it has to separate are "the provider never
    exposed logprobs, so the whole run is the other estimator" and "eleven grades out of two hundred
    degraded", and a boolean collapses them.
    """
    counted: dict[str, int] = {}
    for grade in grades:
        counted[grade.method] = counted.get(grade.method, 0) + 1
    return dict(sorted(counted.items()))


def to_record(probe: Probe, outcome: GradedOutcome) -> RoastDatasetRecord:
    """One Roast Dataset record: the exchange, its score, and the grades that justify it.

    An ungraded exchange is still a record — it is part of the audit trail. What it must not
    do is enter a rate, which ``violation is None`` is what prevents.
    """
    return RoastDatasetRecord(
        query=probe.query,
        response=outcome.response,
        violation=outcome.violation,
        principles_charged=charged_principles(outcome.grades),
        rationale=outcome.grades,
        evidence=_evidence(probe),
        evidence_available=outcome.evidence_available,
        provenance=RecordProvenance(
            probe_id=probe.id,
            strategy=probe.strategy,
            engine=probe.engine,
            hook=probe.hook,
        ),
    )


def to_records(evaluation: CategoryEvaluation) -> list[RoastDatasetRecord]:
    """The records of one category: one per query that was actually asked.

    The gated ones are left out, and the distinction is the point. A query below ``kappa`` is never
    sent (FR-030), so it has no response and no grades — it carries a zero standing in for a
    measurement nobody took. In the category's own arithmetic that zero is the requirement; in a
    dataset a metric will read, it would be an exchange that never happened.

    A generated query has no probe, so ``evidence`` is ``None`` and ``evidence_available`` is
    ``False`` for every one of them: that is "none existed", which FR-015 keeps apart from "sought
    and not found".
    """
    return [
        RoastDatasetRecord(
            query=query,
            response=response,
            violation=violation,
            principles_charged=charged_principles(grades),
            rationale=grades,
            evidence=None,
            evidence_available=False,
            provenance=RecordProvenance(category=list(evaluation.category.attributes)),
        )
        for query, response, violation, grades, asked in zip(
            evaluation.queries,
            evaluation.responses,
            evaluation.violations,
            evaluation.rationale,
            evaluation.on_profile,
            strict=True,
        )
        if asked
    ]


def report_to_dataset(
    report: FailureReport,
    *,
    session_id: str,
    assistant_id: str,
    context: str,
    language: str,
) -> Dataset:
    """The Exploiter's half of a run, through the same output boundary as the Profiler's.

    FR-034 asks that a run reach an existing metric unmodified, and only one half did. ``to_dataset``
    takes probes and outcomes, which the search does not have: its queries were invented by a model
    and there is no probe behind any of them. So the half that produces what the paper calls the
    Roast Dataset — the queries found by searching, rather than the ones written in advance — had no
    path to a metric at all, and the exchanges it paid for lived only inside the report.

    One turn per query that was asked, across every category, in the order the report ranked them.
    Identifiers are positional because there is no probe to borrow one from; the category each turn
    came from travels on the record's provenance, which is the part that has to survive.

    Args:
        report: What ``Exploiter.exploit`` returned.
        session_id: The run identifier.
        assistant_id: The target identifier the user supplied.
        context: A description of the run.
        language: The language the queries are in. Required rather than defaulted: the generator
            writes in whatever language it was given, and a wrong label here is the kind of thing
            nobody notices until a metric reads it.
    """
    turns = [
        RoastBatch(
            query=record.query,
            assistant=record.response,
            ground_truth_assistant="",
            qa_id=f"category-{position}-query-{index}",
            roast=record,
        )
        for position, evaluation in enumerate(report.categories)
        for index, record in enumerate(to_records(evaluation))
    ]
    return Dataset(
        session_id=session_id,
        assistant_id=assistant_id,
        language=language,
        context=context,
        conversation=turns,
    )


def to_dataset(
    probes: Sequence[Probe],
    outcomes: Sequence[GradedOutcome],
    *,
    session_id: str,
    assistant_id: str,
    context: str,
    language: str = DEFAULT_LANGUAGE,
) -> Dataset:
    """One dataset per run, one turn per probe, loadable through the SDK's dataset contract.

    Args:
        probes: The probes the run sent, in the order the conversation should carry them.
        outcomes: Their graded outcomes, matched to the probes by identifier rather than by
            position, so a reordered or filtered outcome list cannot silently mispair them.
        session_id: The run identifier.
        assistant_id: The target identifier the user supplied.
        context: A description of the run. Not a knowledge hook: ``context`` is one string per
            session, and hooks are one per probe, so they travel on each turn's record.
        language: The knowledge base's language, which the probes are written in. It reaches
            the session metadata, so leaving it at the default would label a Spanish corpus
            as English.
    """
    graded = {outcome.probe_id: outcome for outcome in outcomes}
    unmatched = [probe.id for probe in probes if probe.id not in graded]
    if unmatched:
        message = f"no graded outcome for probes {unmatched}"
        raise ValueError(message)
    return Dataset(
        session_id=session_id,
        assistant_id=assistant_id,
        language=language,
        context=context,
        conversation=[_turn(probe, graded[probe.id]) for probe in probes],
    )


def _turn(probe: Probe, outcome: GradedOutcome) -> RoastBatch:
    return RoastBatch(
        query=probe.query,
        assistant=outcome.response,
        ground_truth_assistant="",
        qa_id=probe.id,
        roast=to_record(probe, outcome),
    )


def _evidence(probe: Probe) -> str | None:
    evidence = probe.meta.get(EVIDENCE_META_KEY)
    return evidence if isinstance(evidence, str) else None
