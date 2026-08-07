"""Threshold resolution, and the floor on the sample size behind `S(c)` (T025).

`data-model.md` gives four resolution paths and the third is the point of the mechanism: a
component that recommends nothing, used with nothing supplied, must refuse to construct. A
`kappa` calibrated for a filter scoring `[0, 1]` is a no-op against one scoring `[0, 100]` — it
admits every query, the run completes, and the report looks populated. Validating a range would
not catch it, because the value is inside both ranges (FR-041, SC-012).

The second half is SC-013: `queries_per_category = 1` is rejected at configuration, and the
hand-computed fixture shows why — at `n = 1` the penalised score is the unpenalised mean, so the
inconsistency penalty stops existing rather than becoming small.
"""

import pytest
from pydantic import ValidationError

from gaussia.generators.roastme.exploiter import Exploiter
from gaussia.generators.roastme.searches.scoring import category_score
from gaussia.generators.roastme.searches.thresholds import resolve_delta, resolve_kappa
from gaussia.schemas.roastme import AssistantProfile, ExploiterConfig
from tests.fixtures.roastme import expected as fx
from tests.fixtures.roastme.doubles import (
    RecommendingOnProfileFilter,
    RecommendingRealismEstimator,
    SilentOnProfileFilter,
    SilentRealismEstimator,
    StubCategorySearch,
    StubQueryGenerator,
)

TOLERANCE = 1e-9

TAU = 0.5
ETA = 0.25
SUPPLIED_KAPPA = 0.75
SUPPLIED_DELTA = 0.125


def _config(**overrides) -> ExploiterConfig:
    fields = {"tau": TAU, "eta": ETA}
    fields.update(overrides)
    return ExploiterConfig(**fields)


def _filter(scores: dict[str, float] | None = None) -> RecommendingOnProfileFilter:
    return RecommendingOnProfileFilter(scores or {})


def _estimator() -> RecommendingRealismEstimator:
    return RecommendingRealismEstimator({})


class TestKappaResolution:
    def test_the_component_s_recommendation_is_used_when_nothing_is_supplied(self):
        component = _filter()
        assert resolve_kappa(_config(), component) == pytest.approx(component.recommended_threshold, abs=TOLERANCE)

    def test_a_supplied_value_always_wins(self):
        component = _filter()
        assert component.recommended_threshold != SUPPLIED_KAPPA
        assert resolve_kappa(_config(kappa=SUPPLIED_KAPPA), component) == pytest.approx(SUPPLIED_KAPPA, abs=TOLERANCE)

    def test_a_component_recommending_nothing_with_nothing_supplied_refuses(self):
        with pytest.raises(ValueError, match=r"(?i)kappa") as excinfo:
            resolve_kappa(_config(), SilentOnProfileFilter({}))

        assert SilentOnProfileFilter.__name__ in str(excinfo.value)

    def test_a_component_recommending_nothing_accepts_a_supplied_value(self):
        assert resolve_kappa(_config(kappa=SUPPLIED_KAPPA), SilentOnProfileFilter({})) == pytest.approx(
            SUPPLIED_KAPPA, abs=TOLERANCE
        )

    def test_no_recommendation_travels_across_a_substitution(self):
        """A recommendation is meaningless outside the scale it came from, so a filter that has
        none must never inherit one from a filter that has."""
        with pytest.raises(ValueError, match=r"(?i)kappa"):
            resolve_kappa(_config(), SilentOnProfileFilter({}))

        assert resolve_kappa(_config(), _filter()) == pytest.approx(
            RecommendingOnProfileFilter.recommended_threshold, abs=TOLERANCE
        )


class TestDeltaResolution:
    def test_the_component_s_recommendation_is_used_when_nothing_is_supplied(self):
        component = _estimator()
        assert resolve_delta(_config(), component) == pytest.approx(component.recommended_threshold, abs=TOLERANCE)

    def test_a_supplied_value_always_wins(self):
        component = _estimator()
        assert component.recommended_threshold != SUPPLIED_DELTA
        assert resolve_delta(_config(delta=SUPPLIED_DELTA), component) == pytest.approx(SUPPLIED_DELTA, abs=TOLERANCE)

    def test_a_component_recommending_nothing_with_nothing_supplied_refuses(self):
        with pytest.raises(ValueError, match=r"(?i)delta") as excinfo:
            resolve_delta(_config(), SilentRealismEstimator({}))

        assert SilentRealismEstimator.__name__ in str(excinfo.value)

    def test_a_component_recommending_nothing_accepts_a_supplied_value(self):
        assert resolve_delta(_config(delta=SUPPLIED_DELTA), SilentRealismEstimator({})) == pytest.approx(
            SUPPLIED_DELTA, abs=TOLERANCE
        )


def _exploiter(on_profile_filter, realism_estimator, config):
    return Exploiter(
        contract=fx.contract(fx.stub_grader()),
        target=fx.recorded_target(),
        search=StubCategorySearch([]),
        query_generator=StubQueryGenerator({}),
        on_profile_filter=on_profile_filter,
        realism_estimator=realism_estimator,
        config=config,
    )


class TestResolutionHappensAtExploiterConstruction:
    def test_it_constructs_with_only_tau_and_eta_supplied(self):
        """SC-012: with components that recommend, the user never sees the parameter."""
        exploiter = _exploiter(_filter(), _estimator(), _config())
        assert exploiter is not None

    def test_it_refuses_when_a_configured_component_recommends_nothing(self):
        """SC-012: naming both the component and the parameter, and not quietly running with the
        other component's number."""
        with pytest.raises(ValueError, match=r"(?i)kappa") as excinfo:
            _exploiter(SilentOnProfileFilter({}), _estimator(), _config())

        assert SilentOnProfileFilter.__name__ in str(excinfo.value)

    def test_it_refuses_when_the_estimator_recommends_nothing(self):
        with pytest.raises(ValueError, match=r"(?i)delta") as excinfo:
            _exploiter(_filter(), SilentRealismEstimator({}), _config())

        assert SilentRealismEstimator.__name__ in str(excinfo.value)

    def test_the_resolved_pair_is_what_the_search_receives(self):
        """Resolved once, at construction, so one run has one pair of thresholds in force."""
        search = StubCategorySearch([])
        exploiter = Exploiter(
            contract=fx.contract(fx.stub_grader()),
            target=fx.recorded_target(),
            search=search,
            query_generator=StubQueryGenerator({}),
            on_profile_filter=_filter(),
            realism_estimator=_estimator(),
            config=_config(),
        )
        exploiter.exploit(AssistantProfile())

        assert len(search.configs) == 1
        assert search.configs[0].kappa == pytest.approx(
            RecommendingOnProfileFilter.recommended_threshold, abs=TOLERANCE
        )
        assert search.configs[0].delta == pytest.approx(
            RecommendingRealismEstimator.recommended_threshold, abs=TOLERANCE
        )


class TestTheSampleSizeFloor:
    def test_one_query_per_category_is_rejected_at_configuration(self):
        with pytest.raises(ValidationError, match=r"(?i)queries_per_category|greater than or equal"):
            _config(queries_per_category=1)

    def test_two_is_accepted(self):
        assert _config(queries_per_category=2).queries_per_category == 2

    def test_the_default_is_above_the_floor(self):
        assert _config().queries_per_category >= 2

    def test_at_one_query_the_penalty_stops_existing(self):
        """SC-013's fixture, computed against the scoring functions directly because the config
        that would carry `n = 1` cannot be constructed."""
        mean = sum(fx.SINGLE_VIOLATIONS) / len(fx.SINGLE_VIOLATIONS)

        assert category_score(fx.SINGLE_VIOLATIONS, 1.0) == pytest.approx(mean, abs=TOLERANCE)
        assert category_score(fx.SINGLE_VIOLATIONS, 25.0) == pytest.approx(mean, abs=TOLERANCE)

    def test_above_the_floor_the_penalty_exists(self):
        assert category_score(fx.SPIKY_VIOLATIONS, 1.0) < sum(fx.SPIKY_VIOLATIONS) / len(fx.SPIKY_VIOLATIONS)


class TestTheShippedComponentsRecommend:
    def test_the_shipped_filter_declares_a_kappa(self):
        """SC-012: a user who keeps the shipped component never sees the parameter."""
        from gaussia.generators.roastme.searches.on_profile import JudgeOnProfileFilter

        assert JudgeOnProfileFilter.recommended_threshold is not None

    def test_the_shipped_estimator_declares_a_delta(self):
        from gaussia.generators.roastme.searches.realism import EmbeddingRealismEstimator

        assert EmbeddingRealismEstimator.recommended_threshold is not None
