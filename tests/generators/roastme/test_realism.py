"""The base realism estimator's arithmetic (FR-031).

The estimator is the paper's own construction, so what is asserted here is the construction and
not a calibration: the gap of a query pointing exactly where a prior query points is zero, and a
query the embedder maps to nothing at all is maximally far rather than a NaN.

Those two are in tension, and the normalisation is where they meet. Dividing by `norm + eps`
would keep the zero vector finite at the cost of shrinking every other row by a factor of
`1/(1 + eps/norm)` — small, but enough that an identical query reads as `3e-13` away from
natural instead of `0`. Dividing by `max(norm, eps)` leaves a unit row exactly where it was and
still never divides by zero.

The embedder is injected, so this runs with no model download (SC-010).
"""

import numpy as np
import pytest

from gaussia.core.embedder import Embedder
from gaussia.generators.roastme.searches.realism import EmbeddingRealismEstimator

NATURAL = "a question a real user would ask"
SAME_DIRECTION = "the same question, worded twice as loudly"
ORTHOGONAL = "a question about something else entirely"
UNENCODABLE = "a query the embedder maps to nothing"

VECTORS = {
    NATURAL: [3.0, 4.0],
    SAME_DIRECTION: [6.0, 8.0],
    ORTHOGONAL: [-4.0, 3.0],
    UNENCODABLE: [0.0, 0.0],
}


class StubEmbedder(Embedder):
    """An embedder with a prescribed vector per sentence, so the geometry is hand-checkable."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str]) -> np.ndarray:
        self.calls.append(list(sentences))
        return np.array([self.vectors[sentence] for sentence in sentences], dtype=float)


def _estimator() -> EmbeddingRealismEstimator:
    return EmbeddingRealismEstimator(StubEmbedder(VECTORS), [NATURAL])


class TestTheGap:
    def test_a_query_pointing_exactly_where_a_prior_query_points_has_no_gap(self):
        """Cosine distance is scale-free, so a longer vector in the same direction is the same
        query as far as realism is concerned — exactly, not nearly."""
        assert _estimator().estimate([SAME_DIRECTION]) == 0.0

    def test_an_orthogonal_query_sits_at_cosine_distance_one(self):
        assert _estimator().estimate([ORTHOGONAL]) == 1.0

    def test_the_expectation_is_taken_over_the_category_s_queries(self):
        assert _estimator().estimate([SAME_DIRECTION, ORTHOGONAL]) == 0.5

    def test_a_vector_of_zeros_is_maximally_far_rather_than_a_nan(self):
        """A zero-norm embedding must not turn a similarity into a warning and then into a NaN."""
        gap = _estimator().estimate([UNENCODABLE])

        assert gap == 1.0
        assert not np.isnan(gap)


class TestWhatItRefuses:
    def test_no_prior_pool_at_all(self):
        """The user's own traffic is what makes the estimate mean anything; an empty pool would
        silently make every category look realistic."""
        with pytest.raises(ValueError, match="prior pool"):
            EmbeddingRealismEstimator(StubEmbedder(VECTORS), [])

    def test_no_queries_to_measure(self):
        with pytest.raises(ValueError, match="at least one query"):
            _estimator().estimate([])


class TestItNeverReachesTheAssistant:
    def test_the_prior_is_encoded_once_and_only_queries_follow(self):
        """FR-031: realism is a property of the queries and the prior, so the estimate spends no
        target call — and re-encoding the pool per category would make the budget cost grow with
        the search."""
        embedder = StubEmbedder(VECTORS)
        estimator = EmbeddingRealismEstimator(embedder, [NATURAL])
        estimator.estimate([SAME_DIRECTION])
        estimator.estimate([ORTHOGONAL])

        assert embedder.calls == [[NATURAL], [SAME_DIRECTION], [ORTHOGONAL]]
