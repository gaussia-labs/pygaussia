"""Document retriever with chunk-based similarity search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gaussia.core.embedder import Embedder  # noqa: TC001

if TYPE_CHECKING:
    from gaussia.connectors import RegulatoryDocument


@dataclass
class Chunk:
    """A text chunk with source metadata."""

    text: str
    source: str
    chunk_index: int


@dataclass
class RetrievedChunk(Chunk):
    """A retrieved chunk with similarity score."""

    similarity: float = 0.0


@dataclass
class ChunkerConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 100


@dataclass
class DocumentRetrieverConfig:
    """Configuration for document retrieval."""

    top_k: int = 10
    similarity_threshold: float = 0.3
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)


class DocumentRetriever:
    """Chunks documents and retrieves relevant chunks by embedding similarity.

    Args:
        embedder: Embedder instance for encoding texts.
        config: Retrieval configuration.
    """

    def __init__(
        self,
        embedder: Embedder,
        config: DocumentRetrieverConfig | None = None,
    ):
        self._embedder = embedder
        self._config = config or DocumentRetrieverConfig()
        self._chunks: list[Chunk] = []
        self._doc_embeddings: np.ndarray | None = None

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        text = re.sub(r"\n{3,}", "\n\n", text)

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._config.chunker.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        chunk_index=idx,
                    )
                )
            start += self._config.chunker.chunk_size - self._config.chunker.chunk_overlap
            idx += 1

        return chunks

    def load_corpus(self, documents: list[RegulatoryDocument]) -> int:
        """Load and chunk documents for retrieval.

        Args:
            documents: List of regulatory documents to process.

        Returns:
            Total number of chunks created.
        """
        self._chunks = []
        for doc in documents:
            doc_chunks = self._chunk_text(doc.text, doc.source)
            self._chunks.extend(doc_chunks)

        self._doc_embeddings = None
        return len(self._chunks)

    def _ensure_embeddings(self) -> None:
        if self._doc_embeddings is not None:
            return

        if not self._chunks:
            self._doc_embeddings = np.empty((0, 0))
            return

        chunk_texts = [c.text for c in self._chunks]
        self._doc_embeddings = self._embedder.encode(chunk_texts)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The query text.

        Returns:
            List of retrieved chunks above similarity threshold.
        """
        self._ensure_embeddings()

        if self._doc_embeddings is None or self._doc_embeddings.size == 0:
            return []

        query_embedding = self._embedder.encode_query([query])
        scores = (query_embedding @ self._doc_embeddings.T).flatten().tolist()

        ranked = sorted(
            [(score, chunk) for score, chunk in zip(scores, self._chunks, strict=True)],
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in ranked[: self._config.top_k]:
            if score >= self._config.similarity_threshold:
                results.append(
                    RetrievedChunk(
                        text=chunk.text,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        similarity=round(score, 4),
                    )
                )

        return results

    def retrieve_merged(
        self,
        user_query: str,
        agent_response: str,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks for both query and response, merged and deduplicated.

        Args:
            user_query: The user's query.
            agent_response: The agent's response.

        Returns:
            Merged list of unique chunks with maximum similarity scores.
        """
        query_chunks = self.retrieve(user_query)
        response_chunks = self.retrieve(agent_response)

        seen: dict[tuple[str, int], RetrievedChunk] = {}
        for chunk in query_chunks + response_chunks:
            key = (chunk.source, chunk.chunk_index)
            if key not in seen or chunk.similarity > seen[key].similarity:
                seen[key] = chunk

        return sorted(seen.values(), key=lambda c: c.similarity, reverse=True)


__all__ = [
    "Chunk",
    "ChunkerConfig",
    "DocumentRetriever",
    "DocumentRetrieverConfig",
    "RetrievedChunk",
]
