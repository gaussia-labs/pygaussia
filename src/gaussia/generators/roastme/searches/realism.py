"""The base realism estimator: the paper's divergence, on the paper's own estimator of it.

``delta`` bounds ``D_hat(Q_c || N)``: how far a category's sampled queries sit from the prior over
natural traffic. ``eq:realism-cos`` writes it ``1 - E[cos(E(q), E(q_N))]`` with the expectation over
**both** draws — the mean cosine to the whole pool — and FR-039 singles this component out as the
one whose construction comes from the paper rather than from gaussia.

**It used to take each query's distance to its nearest pool member instead.** The argument for that
was not empty: a query is natural if it resembles *something* real, and a mean distance to a diverse
pool says as much about how spread the pool is as about the query. But a maximum is never below a
mean, so that estimator's ``D`` was never above the paper's and the gate was never stricter and
usually looser — and it was not what the plan this subsystem was approved against said would ship.
The paper's cheaper centroid variant remains the third option, for a user who prefers the
nearest-neighbour argument and wants to substitute it.

Two properties the construction has to keep:

* it never contacts the assistant (FR-031). Realism is a property of the queries and the prior, so
  spending target calls on it would make the budget cost exactly what it exists to protect;
* the scale is this estimator's own, so the ``delta`` it recommends is declared here rather than
  defaulted in the config (FR-041, spec D18).

The embedder is injected. The framework already specifies one, and encoding vectors is not this
estimator's job — so the same estimator runs against whatever model the user already has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gaussia.core.realism_estimator import RealismEstimator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gaussia.core.embedder import Embedder

# Keeps a zero-norm embedding from turning a similarity into a warning and then into a NaN. It
# floors the divisor instead of being added to it: `norm + eps` would shrink every other row by
# `1/(1 + eps/norm)`, enough to leave a query pointing exactly where a natural one points reading
# as a whisker away from the prior rather than on it.
_EPSILON = 1e-12

_LONE_PRIOR_DELTA = 0.5
"""What to recommend when the pool holds one query, which has no spread to measure."""


class EmbeddingRealismEstimator(RealismEstimator):
    """Expected cosine distance from a pool of natural queries — ``eq:realism-cos``.

    Args:
        embedder: The framework's embedding interface, composed rather than built.
        prior: The natural-query pool ``N``. The user's own traffic is what makes the estimate
            mean anything, so gaussia ships no pool: an empty one is refused rather than
            silently making every category look realistic.
    """

    def __init__(self, embedder: Embedder, prior: Sequence[str]) -> None:
        if not prior:
            message = "a realism gap is measured against a prior pool of natural queries, and none was given"
            raise ValueError(message)
        self._embedder = embedder
        # Encoded once: the pool is fixed for the run, and re-encoding it per category would make
        # the budget cost grow with the search.
        self._prior = _unit_rows(embedder.encode(list(prior)))
        self.recommended_threshold: float | None = self._prior_spread()

    def estimate(self, queries: list[str]) -> float:
        if not queries:
            message = "a realism gap needs at least one query to measure"
            raise ValueError(message)
        sampled = _unit_rows(self._embedder.encode_query(queries))
        return float(np.mean(1.0 - sampled @ self._prior.T))

    def _prior_spread(self) -> float:
        """The pool's mean distance to itself, recommended as ``delta``.

        A constant cannot be recommended here. ``delta`` is compared against a mean cosine distance,
        and what that mean comes out at depends on the embedder and on how varied the pool is —
        a tight pool of one intent and a broad pool spanning a whole product line do not put natural
        traffic at the same number. The previous fixed ``0.5`` was chosen against the nearest-
        neighbour estimator, where distances are systematically smaller; carried over unchanged it
        would have rejected almost every category, and a search that evaluates nothing returns an
        empty report that reads like a well-behaved assistant.

        So the yardstick comes from the pool: **as far from the prior as the prior is from itself**.
        A category of genuinely natural queries scores about what a natural query scores, and one
        that drifts scores worse. Self-pairs are excluded — the diagonal is zero by construction and
        would pull the recommendation down in proportion to how small the pool is.

        Free to compute: the pool is already encoded, and nothing here calls a model.
        """
        count = len(self._prior)
        if count < 2:
            return _LONE_PRIOR_DELTA
        distances = 1.0 - self._prior @ self._prior.T
        # Sum over the off-diagonal, which is every pair but a query with itself.
        return float(distances.sum() / (count * (count - 1)))


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    normalised: np.ndarray = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), _EPSILON)
    return normalised
