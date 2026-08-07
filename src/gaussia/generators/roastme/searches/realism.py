"""The base realism estimator: the paper's divergence, on a **nearest-neighbour** estimator of it.

``delta`` bounds ``D_hat(Q_c || N)``: how far a category's sampled queries sit from the prior over
natural traffic. The quantity is the paper's; **this estimator of it is not**, and FR-039 makes the
distinction matter. The paper writes ``1 - E[cos(E(q), E(q_N))]`` with the expectation over both
draws — the mean cosine to the *whole* pool — and offers a cheaper centroid variant. This module
takes each query's distance to its *nearest* pool member instead, for the reason below.

The consequence, stated because it is systematic rather than incidental: a maximum is never below a
mean, so this estimator's ``D`` is never above the paper's, and the realism gate is therefore never
stricter than the paper's and usually looser. Over a deliberately diverse pool the gap is enough to
flip the gate. Whether that is the right trade is a live question and not a settled one.

Two properties the construction has to keep:

* it never contacts the assistant (FR-031). Realism is a property of the queries and the prior, so
  spending target calls on it would make the budget cost exactly what it exists to protect;
* the scale is this estimator's own, so the ``delta`` it recommends is declared here rather than
  defaulted in the config (FR-041). Cosine distance to the nearest prior query lands in ``[0, 2]``
  and in practice in ``[0, 1]``; the recommendation below is a starting point on that scale and
  means nothing on another one.

Why nearest rather than the paper's mean: a query is natural if it resembles *something* real, and
its mean distance to a diverse pool says more about how spread the pool is than about the query. The
paper's centroid variant is a third option, and the one to reach for if the looseness above matters
more than that argument.

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

RECOMMENDED_DELTA = 0.5
"""The ``delta`` recommended on this estimator's cosine-distance scale.

At ``0.5`` a category whose queries share no more than half their direction with anything natural
is discarded. It is a starting point, not a calibrated constant: no grader or estimator here has
been calibrated against human labels (FR-038).
"""

# Keeps a zero-norm embedding from turning a similarity into a warning and then into a NaN. It
# floors the divisor instead of being added to it: `norm + eps` would shrink every other row by
# `1/(1 + eps/norm)`, enough to leave a query pointing exactly where a natural one points reading
# as a whisker away from the prior rather than on it.
_EPSILON = 1e-12


class EmbeddingRealismEstimator(RealismEstimator):
    """Expected cosine distance from a pool of natural queries.

    Args:
        embedder: The framework's embedding interface, composed rather than built.
        prior: The natural-query pool ``N``. The user's own traffic is what makes the estimate
            mean anything, so gaussia ships no pool: an empty one is refused rather than
            silently making every category look realistic.
    """

    recommended_threshold: float | None = RECOMMENDED_DELTA

    def __init__(self, embedder: Embedder, prior: Sequence[str]) -> None:
        if not prior:
            message = "a realism gap is measured against a prior pool of natural queries, and none was given"
            raise ValueError(message)
        self._embedder = embedder
        # Encoded once: the pool is fixed for the run, and re-encoding it per category would make
        # the budget cost grow with the search.
        self._prior = _unit_rows(embedder.encode(list(prior)))

    def estimate(self, queries: list[str]) -> float:
        if not queries:
            message = "a realism gap needs at least one query to measure"
            raise ValueError(message)
        sampled = _unit_rows(self._embedder.encode_query(queries))
        nearest = (sampled @ self._prior.T).max(axis=1)
        return float(np.mean(1.0 - nearest))


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    normalised: np.ndarray = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), _EPSILON)
    return normalised
