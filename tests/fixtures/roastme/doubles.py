"""Deterministic doubles for the ten Roast Me interfaces (T016).

One double per `core/` abstraction, plus the three behaviours the plan singles out as easy
to get silently wrong: a target that replays recorded responses, a target that reports a
failed exchange (SC-008), and a grader whose provider exposes no usable logprobs (SC-009).
The on-profile filter and the realism estimator come in two variants each — one declaring a
recommended threshold, one declaring none — because the four resolution paths of
`data-model.md` need both sides (SC-012).

Every double returns values prescribed by the test, so each arithmetic component can be
asserted against hand computation. Nothing here opens a socket, reads a credential or
touches a GPU (SC-010).
"""

from typing import Any

from gaussia.core.category_search import CategorySearch
from gaussia.core.entity_enumerator import EntityEnumerator
from gaussia.core.grader import Grader
from gaussia.core.hook_verifier import HookVerifier
from gaussia.core.on_profile_filter import OnProfileFilter
from gaussia.core.probe_engine import ProbeEngine
from gaussia.core.query_generator import QueryGenerator
from gaussia.core.realism_estimator import RealismEstimator
from gaussia.core.target_assistant import TargetAssistant
from gaussia.core.transform import Transform
from gaussia.schemas.roastme import (
    AssistantProfile,
    BehavioralContract,
    Catalogue,
    Category,
    CategoryEvaluation,
    Document,
    ExploiterConfig,
    KnowledgeHook,
    Principle,
    PrincipleGrade,
    Probe,
    TargetResponse,
)

LOGPROB_METHOD = "stub-logprob"
FALLBACK_METHOD = "stub-sampling-fallback"


class StubGrader(Grader):
    """A grader whose per-principle score is prescribed per `(query, principle id)`.

    Keyed on the query rather than on a probe identifier because `Grader.grade` never
    receives the probe: the Exploiter grades generated queries that have no probe behind
    them.
    """

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        method: str = LOGPROB_METHOD,
        model: str | None = "stub-model",
    ):
        self.scores = scores
        self.method = method
        self.model = model
        self.calls: list[tuple[str, str, str, dict[str, Any] | None]] = []

    def grade(
        self,
        query: str,
        response: str,
        principle: Principle,
        meta: dict[str, Any] | None = None,
    ) -> PrincipleGrade:
        self.calls.append((query, response, principle.id, meta))
        return PrincipleGrade(
            principle=principle.id,
            score=self.scores[(query, principle.id)],
            grader=type(self).__name__,
            method=self.method,
            model=self.model,
            evidence={"rubric": principle.rubric, "meta": meta or {}},
        )


class NoLogprobStubGrader(StubGrader):
    """A grader standing in for a provider that exposes no usable logprobs (SC-009).

    It still produces a graded outcome — the violation-rate denominator must not depend on
    provider behaviour (spec D13) — and marks the grade as fallback-derived (FR-008).
    """

    def __init__(self, scores: dict[tuple[str, str], float]):
        super().__init__(scores, method=FALLBACK_METHOD, model="stub-model")


class RecordedTarget(TargetAssistant):
    """A target that replays recorded responses, so a run needs no credentials (FR-014).

    Replaying is an implementation of the same interface rather than a separate mode, which
    is what keeps the credential-free path from being a special case.
    """

    def __init__(
        self,
        responses: dict[str, str],
        failures: dict[str, str] | None = None,
        session_id: str | None = None,
    ):
        self.responses = responses
        self.failures = failures or {}
        self.session_id = session_id
        self.sent: list[tuple[str, str | None]] = []

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        self.sent.append((query, session_id))
        if query in self.failures:
            return TargetResponse(
                content="",
                failed=True,
                failure_reason=self.failures[query],
                session_id=self.session_id,
            )
        return TargetResponse(content=self.responses[query], session_id=self.session_id)


class FailingTarget(TargetAssistant):
    """A target that reports every exchange as failed at transport level (SC-008)."""

    def __init__(self, reason: str = "stub transport failure"):
        self.reason = reason
        self.sent: list[tuple[str, str | None]] = []

    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        self.sent.append((query, session_id))
        return TargetResponse(content="", failed=True, failure_reason=self.reason)


class StubProbeEngine(ProbeEngine):
    """A probe engine returning prescribed probes for the documents it declares it handles."""

    def __init__(
        self,
        engine_name: str,
        kinds: frozenset[str],
        probes: list[Probe],
        handles: frozenset[str] | None = None,
    ):
        self._name = engine_name
        self._kinds = kinds
        self._probes = probes
        self._handles = handles
        self.generate_calls: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def entity_kinds(self) -> frozenset[str]:
        return self._kinds

    def can_handle(self, document: Document) -> bool:
        return self._handles is None or document.id in self._handles

    def generate(self, documents: list[Document], catalogue: Catalogue) -> list[Probe]:
        self.generate_calls.append((len(documents), len(catalogue.strategies)))
        return [probe.model_copy(update={"engine": self._name}) for probe in self._probes]


class StubEntityEnumerator(EntityEnumerator):
    """An enumerator with a complete, prescribed entity set per kind.

    Completeness is the whole contract: this is the enumeration a test scores absence
    labels against, and the engine under test never sees it (FR-021).
    """

    def __init__(self, entities: dict[str, frozenset[str]]):
        self.entities = entities
        self.calls: list[str] = []

    def enumerate_entities(self, kind: str, documents: list[Document]) -> frozenset[str]:
        self.calls.append(kind)
        return self.entities.get(kind, frozenset())


class StubHookVerifier(HookVerifier):
    """A verifier whose verdict is prescribed per hook reference."""

    def __init__(self, verdicts: dict[str, bool]):
        self.verdicts = verdicts
        self.calls: list[str] = []

    def verify(self, hook: KnowledgeHook, documents: list[Document]) -> bool:
        self.calls.append(hook.references)
        return self.verdicts[hook.references]


class StubTransform(Transform):
    """A transform answering to a prescribed catalogue key with a prescribed rewrite."""

    def __init__(self, transform_key: str, results: dict[str, str]):
        self._key = transform_key
        self.results = results

    @property
    def key(self) -> str:
        return self._key

    def apply(self, entity: str) -> str:
        return self.results.get(entity, entity)


class StubQueryGenerator(QueryGenerator):
    """A query generator returning prescribed queries per category.

    It returns exactly ``count`` queries: returning fewer would shrink the denominator of
    ``S(c)`` without saying so. It also records every call, so a search can be checked for
    having left this collaborator unmodified (paper invariant 5).
    """

    def __init__(self, queries: dict[tuple[str, ...], list[str]]):
        self.queries = queries
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def generate(self, category: Category, count: int) -> list[str]:
        key = tuple(category.attributes)
        self.calls.append((key, count))
        pool = self.queries[key]
        return list(pool[:count])


class RecommendingOnProfileFilter(OnProfileFilter):
    """An on-profile filter that declares the ``kappa`` it recommends on its own scale."""

    recommended_threshold: float | None = 0.5

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[str] = []

    def score(self, query: str, profile: AssistantProfile) -> float:
        self.calls.append(query)
        return self.scores[query]


class SilentOnProfileFilter(RecommendingOnProfileFilter):
    """An on-profile filter that declares no recommended ``kappa`` (SC-012).

    Declaring none is legitimate and obliges the user to supply ``kappa`` explicitly.
    """

    recommended_threshold: float | None = None


class RecommendingRealismEstimator(RealismEstimator):
    """A realism estimator that declares the ``delta`` it recommends on its own scale."""

    recommended_threshold: float | None = 0.3

    def __init__(self, gaps: dict[tuple[str, ...], float]):
        self.gaps = gaps
        self.calls: list[tuple[str, ...]] = []

    def estimate(self, queries: list[str]) -> float:
        key = tuple(queries)
        self.calls.append(key)
        return self.gaps[key]


class SilentRealismEstimator(RecommendingRealismEstimator):
    """A realism estimator that declares no recommended ``delta`` (SC-012)."""

    recommended_threshold: float | None = None


class StubCategorySearch(CategorySearch):
    """A search returning prescribed evaluations and recording what it was handed.

    Recording the config is what lets a test assert the Exploiter resolved ``kappa`` and
    ``delta`` once, before the search ran, rather than leaving the search to resolve them.
    """

    def __init__(self, evaluations: list[CategoryEvaluation]):
        self.evaluations = evaluations
        self.configs: list[ExploiterConfig] = []
        self.profiles: list[AssistantProfile] = []
        self.collaborators: list[tuple[object, object, object, object]] = []

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
        self.profiles.append(profile)
        self.configs.append(config)
        self.collaborators.append((target, query_generator, on_profile_filter, realism_estimator))
        return list(self.evaluations)
