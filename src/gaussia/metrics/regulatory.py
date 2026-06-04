"""Regulatory compliance metric for evaluating AI responses against regulatory corpus."""

from __future__ import annotations

from typing import Literal

from gaussia.connectors import CorpusConnector  # noqa: TC001
from gaussia.core import Gaussia, Retriever
from gaussia.core.contradiction_checker import ContradictionChecker
from gaussia.core.document_retriever import (
    ChunkerConfig,
    DocumentRetriever,
    DocumentRetrieverConfig,
)
from gaussia.core.embedder import Embedder  # noqa: TC001
from gaussia.core.reranker import Reranker  # noqa: TC001
from gaussia.schemas import Batch  # noqa: TC001
from gaussia.schemas.regulatory import RegulatoryChunk, RegulatoryInteraction, RegulatoryMetric
from gaussia.statistical import FrequentistMode, StatisticalMode


class Regulatory(Gaussia):
    """Evaluates AI assistant responses against a regulatory corpus.

    Accumulates per-interaction compliance scores and emits one session-level
    RegulatoryMetric in on_process_complete(). The aggregated compliance score
    uses the configured StatisticalMode -- frequentist returns a weighted mean,
    Bayesian returns a bootstrapped credible interval.

    Args:
        retriever: Retriever class for loading conversation datasets.
        corpus_connector: Connector for loading regulatory documents.
        embedder: Embedder for encoding documents and queries.
        reranker: Reranker for scoring document-response alignment.
        statistical_mode: Statistical computation mode (defaults to FrequentistMode).
        chunk_size: Character size for text chunks.
        chunk_overlap: Character overlap between chunks.
        top_k: Maximum chunks to retrieve per query.
        similarity_threshold: Minimum cosine similarity for retrieval.
        contradiction_threshold: Score below which a chunk is considered contradicting.
        compliance_threshold: Minimum compliance score to consider a response compliant.
        **kwargs: Additional arguments passed to Gaussia base class.
    """

    def __init__(
        self,
        retriever: type[Retriever],
        corpus_connector: CorpusConnector,
        embedder: Embedder,
        reranker: Reranker,
        statistical_mode: StatisticalMode | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        contradiction_threshold: float = 0.6,
        compliance_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)

        self.corpus_connector = corpus_connector
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()
        self.compliance_threshold = compliance_threshold

        retriever_config = DocumentRetrieverConfig(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            chunker=ChunkerConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
        )
        self.document_retriever = DocumentRetriever(embedder, retriever_config)
        self.contradiction_checker = ContradictionChecker(reranker, contradiction_threshold)

        self._corpus_loaded = False
        self._session_data: dict[str, dict] = {}

        self.logger.info("--REGULATORY CONFIGURATION--")
        self.logger.info(f"Embedder: {type(embedder).__name__}")
        self.logger.info(f"Reranker: {type(reranker).__name__}")
        self.logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        self.logger.info(f"Top-K: {top_k}, Similarity threshold: {similarity_threshold}")
        self.logger.info(f"Contradiction threshold: {contradiction_threshold}")
        self.logger.info(f"Compliance threshold: {compliance_threshold}")
        self.logger.info(f"Statistical mode: {self.statistical_mode.get_result_type()}")

    def _ensure_corpus_loaded(self) -> None:
        if self._corpus_loaded:
            return
        documents = self.corpus_connector.load_documents()
        num_chunks = self.document_retriever.load_corpus(documents)
        self.logger.info(f"Loaded corpus: {len(documents)} documents -> {num_chunks} chunks")
        self._corpus_loaded = True

    def _compute_verdict(
        self,
        supporting: int,
        contradicting: int,
    ) -> tuple[Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"], float]:
        total = supporting + contradicting

        if total == 0:
            return "IRRELEVANT", self.compliance_threshold

        compliance_score = supporting / total

        if contradicting > 0 and supporting == 0:
            return "NON_COMPLIANT", compliance_score

        if compliance_score >= self.compliance_threshold:
            return "COMPLIANT", compliance_score

        return "NON_COMPLIANT", compliance_score

    def _generate_insight(self, verdict: str, supporting: int, contradicting: int) -> str:
        total = supporting + contradicting

        if verdict == "IRRELEVANT":
            return "No relevant regulatory chunks were retrieved for this interaction."

        if verdict == "COMPLIANT":
            return f"Response is COMPLIANT. {supporting} of {total} relevant chunk(s) support the response."

        return f"Response is NON-COMPLIANT. {contradicting} of {total} relevant chunk(s) contradict the response."

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        self._ensure_corpus_loaded()

        if session_id not in self._session_data:
            self._session_data[session_id] = {
                "assistant_id": assistant_id,
                "batches": [],
                "scores": [],
                "interactions": [],
            }

        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            retrieved = self.document_retriever.retrieve_merged(
                user_query=interaction.query,
                agent_response=interaction.assistant,
            )

            if not retrieved:
                interaction_result = RegulatoryInteraction(
                    qa_id=interaction.qa_id,
                    query=interaction.query,
                    assistant=interaction.assistant,
                    compliance_score=self.compliance_threshold,
                    verdict="IRRELEVANT",
                    supporting_chunks=0,
                    contradicting_chunks=0,
                    retrieved_chunks=[],
                    insight="No relevant regulatory chunks were retrieved for this interaction.",
                )
            else:
                ranked = self.contradiction_checker.check(
                    agent_response=interaction.assistant,
                    retrieved_chunks=retrieved,
                )

                supporting = sum(1 for r in ranked if r.verdict == "SUPPORTS")
                contradicting = sum(1 for r in ranked if r.verdict == "CONTRADICTS")
                verdict, compliance_score = self._compute_verdict(supporting, contradicting)
                insight = self._generate_insight(verdict, supporting, contradicting)

                interaction_result = RegulatoryInteraction(
                    qa_id=interaction.qa_id,
                    query=interaction.query,
                    assistant=interaction.assistant,
                    compliance_score=round(compliance_score, 4),
                    verdict=verdict,
                    supporting_chunks=supporting,
                    contradicting_chunks=contradicting,
                    retrieved_chunks=[
                        RegulatoryChunk(
                            text=r.text,
                            source=r.source,
                            chunk_index=r.chunk_index,
                            similarity=r.similarity,
                            reranker_score=r.reranker_score,
                            verdict=r.verdict,
                        )
                        for r in ranked
                    ],
                    insight=insight,
                )

            self._session_data[session_id]["batches"].append(interaction)
            self._session_data[session_id]["scores"].append(interaction_result.compliance_score)
            self._session_data[session_id]["interactions"].append(interaction_result)

    def on_process_complete(self):
        for session_id, data in self._session_data.items():
            batches = data["batches"]
            interactions = data["interactions"]
            weights = self._resolve_weights(batches)

            mean, ci_low, ci_high = self._aggregate_scores(data["scores"], batches, weights, self.statistical_mode)

            total_supporting = sum(i.supporting_chunks for i in interactions)
            total_contradicting = sum(i.contradicting_chunks for i in interactions)

            session_verdict: Literal["COMPLIANT", "NON_COMPLIANT", "IRRELEVANT"]
            if all(i.verdict == "IRRELEVANT" for i in interactions):
                session_verdict = "IRRELEVANT"
            elif mean >= self.compliance_threshold:
                session_verdict = "COMPLIANT"
            else:
                session_verdict = "NON_COMPLIANT"

            metric = RegulatoryMetric(
                session_id=session_id,
                assistant_id=data["assistant_id"],
                n_interactions=len(batches),
                compliance_score=round(mean, 4),
                compliance_score_ci_low=round(ci_low, 4) if ci_low is not None else None,
                compliance_score_ci_high=round(ci_high, 4) if ci_high is not None else None,
                verdict=session_verdict,
                total_supporting_chunks=total_supporting,
                total_contradicting_chunks=total_contradicting,
                interactions=interactions,
            )
            self.metrics.append(metric)


__all__ = ["Regulatory"]
