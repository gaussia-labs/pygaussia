"""The Profiler: a probe set and a target assistant to the profile ``theta = (omega, H)``.

It drives the injected target over the probes, grades every response against every principle
of the contract, and aggregates what survives into the weakness map and the retained hooks
(FR-010...FR-016). Three things it does not do, each load-bearing:

* it never reaches the knowledge base — nothing on its surface accepts a document, so paper
  invariant 2 is structural rather than documented (FR-010);
* it never contacts the assistant except through the target interface, so replaying recorded
  responses is the same code as a live run rather than a separate mode (FR-014, spec D16);
* it sends no identifier across to the Exploiter. Aggregation is keyed internally by the
  probe's strategy, and only the prose of a probe's attributes lands on the profile
  (FR-013, paper invariant 3).

A control is recognised by one thing only: its probe carries no plugin, because the strategy
that produced it named none (FR-011, FR-026). Its outcome is graded and kept in the record and
excluded from every rate. An exchange the target marks as failed is recorded ungraded, and
``violation is None`` is the only representation of that, so it moves neither the numerator nor
the denominator of anything (FR-016).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from gaussia.schemas.roastme import AssistantProfile, GradedOutcome, ProfilerResult

from .dataset import grading_methods
from .searches.scoring import violation_score, weakness_entry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.target_assistant import TargetAssistant
    from gaussia.schemas.roastme import (
        BehavioralContract,
        KnowledgeHook,
        PrincipleGrade,
        Probe,
        TargetResponse,
        WeaknessEntry,
    )

DESCRIPTOR_SEPARATOR = ", "


class _Scored(NamedTuple):
    """A graded, scoreable exchange, with its violation already narrowed out of ``None``."""

    probe: Probe
    outcome: GradedOutcome
    violation: float


class Profiler:
    """Builds the assistant profile the Exploiter searches from.

    Args:
        contract: The principles under test, each with the grader bound to it. Grading runs
            over every principle for every response, so a failure traces to the principle it
            breaks (FR-004).
        target: The user's adapter over the assistant. The only path to it (FR-017).
    """

    def __init__(self, contract: BehavioralContract, target: TargetAssistant) -> None:
        self._contract = contract
        self._target = target

    def profile(self, probes: Sequence[Probe]) -> ProfilerResult:
        outcomes = [self._exchange(probe) for probe in probes]
        scored = _scoreable(probes, outcomes)
        return ProfilerResult(
            profile=AssistantProfile(
                weaknesses=self._weaknesses(scored),
                hooks=_retained_hooks(scored),
            ),
            outcomes=outcomes,
            probes=list(probes),
            overall_rate=_mean([item.violation for item in scored]),
            n_scoreable=len(scored),
            n_ungraded=sum(1 for outcome in outcomes if outcome.violation is None),
            grading_methods=grading_methods(grade for outcome in outcomes for grade in outcome.grades),
        )

    def _exchange(self, probe: Probe) -> GradedOutcome:
        response = self._target.send(probe.query)
        grades, ungraded_reason = self._grades(probe, response)
        return GradedOutcome(
            probe_id=probe.id,
            response=response.content,
            grades=grades,
            # No grades is exactly the failed exchange: a contract carries at least one
            # principle, so a graded outcome always has at least one grade behind it (FR-016).
            violation=violation_score(grades, self._contract) if grades else None,
            evidence_available=probe.hook is not None,
            scoreable=probe.plugin is not None,
            ungraded_reason=ungraded_reason,
        )

    def _grades(self, probe: Probe, response: TargetResponse) -> tuple[list[PrincipleGrade], str | None]:
        """Every principle graded, or none of them and why.

        FR-016 already covers the assistant: an exchange it failed is recorded ungraded rather than
        as a pass, because a transport error read as good behaviour is a silent free mark. **The
        same was not true of the judge.** Nothing here caught anything, so one unparseable verdict
        on probe four hundred of five hundred ended the run — taking with it every assistant call
        already paid for, which is the half that cannot be reproduced: the assistant is not
        deterministic, so a lost response is not recovered, it is replaced, and that is a different
        run.

        Partial grades are discarded rather than kept, and that is deliberate: ``v`` is a weighted
        sum over **every** principle of the contract, so a subset cannot produce one. Keeping them
        would invite an average over whichever principles happened to answer.
        """
        if response.failed:
            return [], response.failure_reason or "the target reported the exchange failed"
        try:
            return [
                principle.grader.grade(probe.query, response.content, principle, probe.meta)
                for principle in self._contract.principles
            ], None
        except Exception as unruled:  # a judge fails through whichever client it wraps
            return [], f"the judge failed: {type(unruled).__name__}: {unruled}"

    def _weaknesses(self, scored: Sequence[_Scored]) -> list[WeaknessEntry]:
        grouped: dict[str, list[_Scored]] = defaultdict(list)
        for item in scored:
            grouped[item.probe.strategy].append(item)
        return [
            weakness_entry(
                principle.id,
                _descriptor(group),
                [_grade_for(item.outcome, principle.id) for item in group],
            )
            for group in grouped.values()
            for principle in self._contract.principles
        ]


def _scoreable(probes: Sequence[Probe], outcomes: Sequence[GradedOutcome]) -> list[_Scored]:
    scored = []
    for probe, outcome in zip(probes, outcomes, strict=True):
        violation = outcome.violation
        if outcome.scoreable and violation is not None:
            scored.append(_Scored(probe, outcome, violation))
    return scored


def _descriptor(group: Sequence[_Scored]) -> str:
    """The prose of what a strategy's probes have in common, which is the ``z`` of ``omega``.

    Built from the probes' own attributes rather than from the strategy identifier: the
    identifier is the user's private vocabulary and may not cross to the Exploiter (FR-013).
    Probes carrying no attributes leave nothing sayable, and inventing a placeholder would put
    an identifier back on the profile by another route, so it fails instead.
    """
    attributes = list(dict.fromkeys(attribute for item in group for attribute in item.probe.attrs))
    if not attributes:
        message = "probes carrying no attributes leave the weakness map with no prose descriptor"
        raise ValueError(message)
    return DESCRIPTOR_SEPARATOR.join(attributes)


def _grade_for(outcome: GradedOutcome, principle: str) -> float:
    for grade in outcome.grades:
        if grade.principle == principle:
            return grade.score
    message = f"outcome {outcome.probe_id} carries no grade for principle {principle}"
    raise ValueError(message)


def _retained_hooks(scored: Sequence[_Scored]) -> list[KnowledgeHook]:
    """``H``: the hooks of the probes that actually broke the assistant, deduplicated.

    An evaluator needs to know which specific entities broke it, and the Exploiter grounds
    categories on these — so a hook whose probe drew no violation is not evidence of a
    weakness and does not belong in ``H``.
    """
    retained: dict[str, KnowledgeHook] = {}
    for item in scored:
        hook = item.probe.hook
        if item.violation > 0.0 and hook is not None:
            retained.setdefault(hook.model_dump_json(), hook)
    return list(retained.values())


def _mean(values: Sequence[float]) -> float:
    # An empty run has no rate; ``n_scoreable`` is what says so, which is why zero here is not
    # readable as "nothing violated".
    return sum(values) / len(values) if values else 0.0
