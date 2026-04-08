"""Contradiction checker for evaluating document-response alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pygaussia.core.document_retriever import RetrievedChunk  # noqa: TC001
from pygaussia.core.reranker import Reranker  # noqa: TC001


@dataclass
class RankedChunk:
    """A chunk with reranker verdict."""

    text: str
    source: str
    chunk_index: int
    similarity: float
    reranker_score: float
    verdict: Literal["SUPPORTS", "CONTRADICTS"]


class ContradictionChecker:
    """Checks if retrieved chunks support or contradict an agent response.

    Args:
        reranker: Reranker instance for scoring query-document pairs.
        contradiction_threshold: Score below which a chunk is considered contradicting.
    """

    def __init__(
        self,
        reranker: Reranker,
        contradiction_threshold: float = 0.6,
    ):
        self._reranker = reranker
        self._contradiction_threshold = contradiction_threshold

    def check(
        self,
        agent_response: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RankedChunk]:
        """Check if retrieved chunks contradict the agent response.

        Args:
            agent_response: The agent's response to check.
            retrieved_chunks: List of retrieved chunks to evaluate.

        Returns:
            List of ranked chunks with support/contradiction verdicts.
        """
        if not retrieved_chunks:
            return []

        documents = [chunk.text for chunk in retrieved_chunks]
        scores = self._reranker.score(agent_response, documents)

        results = []
        for chunk, score in zip(retrieved_chunks, scores, strict=True):
            verdict: Literal["SUPPORTS", "CONTRADICTS"] = (
                "SUPPORTS" if score >= self._contradiction_threshold else "CONTRADICTS"
            )
            results.append(
                RankedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    similarity=chunk.similarity,
                    reranker_score=round(score, 4),
                    verdict=verdict,
                )
            )

        return results


__all__ = [
    "ContradictionChecker",
    "RankedChunk",
]
