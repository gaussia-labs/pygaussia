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
from gaussia.schemas.roastme import RoastBatch, RoastDatasetRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.schemas.roastme import GradedOutcome, Probe

DEFAULT_LANGUAGE = "english"

EVIDENCE_META_KEY = "evidence"
"""Where a probe engine records the supporting or contradicting source text, when it has one.

A probe generated with no knowledge base has none, which is what makes ``evidence`` ``None``
in black-box mode; a knowledge-grounded probe whose engine recorded none is the other case,
and ``evidence_available`` is what keeps the two apart.
"""


def to_record(probe: Probe, outcome: GradedOutcome) -> RoastDatasetRecord:
    """One Roast Dataset record: the exchange, its score, and the grades that justify it.

    An ungraded exchange is still a record — it is part of the audit trail. What it must not
    do is enter a rate, which ``violation is None`` is what prevents.
    """
    return RoastDatasetRecord(
        query=probe.query,
        response=outcome.response,
        violation=outcome.violation,
        principles_charged=[grade.principle for grade in outcome.grades if grade.score > 0.0],
        rationale=outcome.grades,
        evidence=_evidence(probe),
        evidence_available=outcome.evidence_available,
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
