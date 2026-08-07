"""The training-free category search: attributes of the highest-scoring pool, then subsets.

The default of the two procedures behind ``CategorySearch``, and the one that needs no GPU, no
trained model and no optimiser (FR-033). It has no published result behind it — the paper's
headline procedure is the policy-gradient one — so what it offers is a run that costs only target
calls.

Three steps:

1. **Ground.** Every attribute comes from the profile: a weakness whose rate reaches ``eta``
   contributes its descriptor, a retained hook contributes the entity it leans on. Each carries
   the entry that induced it, so no attribute ever lacks provenance (FR-028). Nothing else is
   invented — a category grounded in something the Profiler never observed is a guess.
2. **Pool.** Each grounded attribute is evaluated on its own, and every query/response pair those
   evaluations produced goes into one pool ranked by violation. The attributes behind the highest
   scoring ``pool_size`` pairs are conjoined into one candidate: whatever the assistant broke on
   most, asked all at once.
3. **Refine.** The candidate is reduced to the minimal sub-conjunction still passing ``tau`` and
   ``delta``, and what came off is reported as incidental (FR-032). This is the step that turns a
   pile of co-occurring attributes into a category an evaluator can act on.

Refinement is exhaustive over sub-conjunctions, so the candidate's length is what bounds the cost
of a run — hence ``max_attributes``, and hence a default rather than an unbounded conjunction.

The single-attribute evaluations are kept in the result, not discarded. A run where the conjunction
fails and one attribute alone reaches ``tau`` has found something, and dropping the seeds would
leave the report saying nothing happened.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple

from gaussia.core.category_search import CategorySearch
from gaussia.schemas.roastme import Category

from .evaluation import CategoryEvaluator
from .scoring import refine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.on_profile_filter import OnProfileFilter
    from gaussia.core.query_generator import QueryGenerator
    from gaussia.core.realism_estimator import RealismEstimator
    from gaussia.core.target_assistant import TargetAssistant
    from gaussia.schemas.roastme import (
        AssistantProfile,
        BehavioralContract,
        CategoryEvaluation,
        ExploiterConfig,
    )

DEFAULT_MAX_ATTRIBUTES = 3

_HOOK_ATTRIBUTE = "concerns {reference}"
_WEAKNESS_PROVENANCE = "weakness {principle}: {descriptor}"
_HOOK_PROVENANCE = "hook: {reference}"


class _Attribute(NamedTuple):
    """One natural-language attribute and the profile entry that induced it (FR-028)."""

    text: str
    provenance: str


class AttributeIterationSearch(CategorySearch):
    """Proposes categories by conjoining the attributes that scored highest on their own.

    Args:
        max_attributes: How many of the pool's attributes the candidate conjunction may carry.
            Refinement tries every sub-conjunction, so a candidate of length ``l`` costs
            ``2^l - 1`` evaluations and this is what keeps a run finite. A knob of gaussia's own
            search rather than a parameter of the method, so gaussia owns its default (FR-040).
    """

    def __init__(self, max_attributes: int = DEFAULT_MAX_ATTRIBUTES) -> None:
        self._max_attributes = max_attributes

    def search(
        self,
        profile: AssistantProfile,
        contract: BehavioralContract,
        config: ExploiterConfig,
        target: TargetAssistant,
        query_generator: QueryGenerator,
        on_profile_filter: OnProfileFilter,
        realism_estimator: RealismEstimator,
    ) -> list[CategoryEvaluation]:
        evaluator = CategoryEvaluator(
            profile,
            contract,
            config,
            target,
            query_generator,
            on_profile_filter,
            realism_estimator,
        )
        seeded = _seeded(evaluator, _grounded(profile, config.eta))
        reported = {tuple(evaluation.category.attributes): evaluation for _, evaluation in seeded}
        candidate = _candidate(seeded, config.pool_size, self._max_attributes)
        if candidate is not None:
            refined, dropped = refine(candidate, evaluator.verdict, config.tau, evaluator.delta)
            evaluation = evaluator.evaluate(refined)
            if evaluation is not None:
                reported[tuple(refined.attributes)] = evaluation.model_copy(update={"dropped_attributes": dropped})
        return list(reported.values())


def _grounded(profile: AssistantProfile, eta: float) -> list[_Attribute]:
    """The attributes the profile justifies: weaknesses at or above ``eta``, and retained hooks."""
    grounded = [
        _Attribute(
            entry.descriptor,
            _WEAKNESS_PROVENANCE.format(principle=entry.principle, descriptor=entry.descriptor),
        )
        for entry in profile.weaknesses
        if entry.rate >= eta
    ]
    grounded += [
        _Attribute(
            _HOOK_ATTRIBUTE.format(reference=hook.references),
            _HOOK_PROVENANCE.format(reference=hook.references),
        )
        for hook in profile.hooks
    ]
    # Two entries of the weakness map may describe the same behaviour under different principles,
    # and the same attribute twice in one conjunction says nothing the once did not.
    unique: dict[str, _Attribute] = {}
    for attribute in grounded:
        unique.setdefault(attribute.text, attribute)
    return list(unique.values())


def _seeded(
    evaluator: CategoryEvaluator,
    grounded: Sequence[_Attribute],
) -> list[tuple[_Attribute, CategoryEvaluation]]:
    """Every grounded attribute evaluated on its own, keeping the ones that produced a score."""
    evaluated = ((attribute, evaluator.evaluate(_conjunction([attribute]))) for attribute in grounded)
    return [(attribute, evaluation) for attribute, evaluation in evaluated if evaluation is not None]


def _candidate(
    seeded: Sequence[tuple[_Attribute, CategoryEvaluation]],
    pool_size: int,
    width: int,
) -> Category | None:
    """The conjunction of the attributes behind the highest-scoring pairs, best first."""
    pool = sorted(
        ((violation, attribute) for attribute, evaluation in seeded for violation in evaluation.violations),
        key=itemgetter(0),
        reverse=True,
    )[:pool_size]
    attributes = list(dict.fromkeys(attribute for _, attribute in pool))[:width]
    return _conjunction(attributes) if attributes else None


def _conjunction(attributes: Sequence[_Attribute]) -> Category:
    return Category(
        attributes=[attribute.text for attribute in attributes],
        provenance=[attribute.provenance for attribute in attributes],
    )
