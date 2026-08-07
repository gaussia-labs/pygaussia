"""The behavioral contract's own admissibility rules (FR-001).

`BehavioralContract` is the one place gaussia decides whether a contract may be used at all, and
each of its two rejections stops a `v` that would be quietly wrong rather than obviously broken:

* weights that do not sum to 1 leave `v` on a scale `tau` was never set against, so a run
  completes and reports numbers that mean nothing;
* a duplicate principle id makes `v` depend on which of two entries a lookup happens to find,
  which is the same failure mode `violation_score` refuses a duplicate grade for.

The tolerance is the third case and the one worth pinning from both sides: `1 ± 1e-9` exists so a
contract assembled from decimals is not rejected for float noise, which is a different thing from
letting a score leave `[0, 1]` — the bound that keeps those apart is in `test_scoring.py`.
"""

import pytest
from pydantic import ValidationError

from gaussia.schemas.roastme import BehavioralContract, Principle
from tests.fixtures.roastme import expected as fx

# Just outside the tolerance: 0.5 + 0.3 + 0.200000002 sums to 1.000000002, which is 2e-9 away.
OVER_TOLERANCE_WEIGHT_C = 0.200000002


def _principle(identifier: str, weight: float) -> Principle:
    return Principle(id=identifier, weight=weight, rubric=f"rubric for {identifier}", grader=fx.stub_grader())


def _contract(weights: dict[str, float]) -> BehavioralContract:
    return BehavioralContract(principles=[_principle(identifier, weight) for identifier, weight in weights.items()])


class TestTheWeightsMustSumToOne:
    @pytest.mark.parametrize(
        "weights",
        [
            {fx.PRINCIPLE_A: 0.5, fx.PRINCIPLE_B: 0.3, fx.PRINCIPLE_C: 0.1},
            {fx.PRINCIPLE_A: 0.5, fx.PRINCIPLE_B: 0.3, fx.PRINCIPLE_C: 0.4},
            {fx.PRINCIPLE_A: 1.0, fx.PRINCIPLE_B: 1.0},
        ],
    )
    def test_a_sum_that_is_not_one_is_rejected(self, weights):
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            _contract(weights)

    def test_a_sum_just_outside_the_tolerance_is_rejected(self):
        """The tolerance is float noise, not a licence to weight a contract to 1.000000002."""
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            _contract({fx.PRINCIPLE_A: 0.5, fx.PRINCIPLE_B: 0.3, fx.PRINCIPLE_C: OVER_TOLERANCE_WEIGHT_C})

    def test_a_sum_inside_the_tolerance_is_accepted(self):
        """FR-001: a contract assembled from decimals must not be rejected for float noise."""
        contract = fx.tolerance_edge_contract(fx.stub_grader())
        total = sum(principle.weight for principle in contract.principles)

        assert total == fx.EDGE_WEIGHT_SUM
        assert total > 1.0

    def test_a_sum_of_exactly_one_is_accepted(self):
        assert len(fx.contract(fx.stub_grader()).principles) == 3


class TestTheIdentifiersMustBeUnique:
    def test_a_repeated_principle_id_is_rejected(self):
        """Two entries under one id: `v` would charge whichever of them the lookup kept.

        The weights sum to exactly 1.0, so it is the identifier rule that rejects this contract
        and not the weight rule catching it by accident.
        """
        with pytest.raises(ValidationError, match="unique"):
            BehavioralContract(
                principles=[
                    _principle(fx.PRINCIPLE_A, 0.5),
                    _principle(fx.PRINCIPLE_B, 0.3),
                    _principle(fx.PRINCIPLE_A, 0.2),
                ]
            )
