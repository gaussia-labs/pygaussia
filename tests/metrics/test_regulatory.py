"""Unit tests for Regulatory metric."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pygaussia.connectors import LocalCorpusConnector, RegulatoryDocument
from pygaussia.core.contradiction_checker import ContradictionChecker, RankedChunk
from pygaussia.core.document_retriever import DocumentRetriever, RetrievedChunk
from pygaussia.core.embedder import Embedder
from pygaussia.core.reranker import Reranker
from pygaussia.metrics.regulatory import Regulatory
from pygaussia.schemas.regulatory import RegulatoryChunk, RegulatoryInteraction, RegulatoryMetric


class MockCorpusConnector:
    """Mock corpus connector for testing."""

    def load_documents(self) -> list[RegulatoryDocument]:
        return [
            RegulatoryDocument(
                text="Customers have the right to request removal from call lists. "
                "All requests must be honored within 24 hours. "
                "No calls should be made between 9 PM and 8 AM local time.",
                source="call_policy.md",
            ),
            RegulatoryDocument(
                text="Refund policy: Full refunds are available within 30 days of purchase. "
                "Original receipt is required for all refund requests. "
                "Refunds will be processed within 5-7 business days.",
                source="refund_policy.md",
            ),
        ]


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        return np.array([[0.1] * 10 for _ in sentences])


class MockReranker(Reranker):
    """Mock reranker for testing."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.8] * len(documents)


def _make_mock_embedder() -> Embedder:
    return MockEmbedder()


def _make_mock_reranker() -> Reranker:
    return MockReranker()


class TestRegulatoryMetricSchema:
    """Test suite for RegulatoryMetric schema."""

    def test_regulatory_chunk_creation(self):
        chunk = RegulatoryChunk(
            text="Test regulatory text",
            source="test.md",
            chunk_index=0,
            similarity=0.85,
            reranker_score=0.7,
            verdict="SUPPORTS",
        )
        assert chunk.text == "Test regulatory text"
        assert chunk.source == "test.md"
        assert chunk.similarity == 0.85
        assert chunk.verdict == "SUPPORTS"

    def test_regulatory_metric_creation(self):
        interaction = RegulatoryInteraction(
            qa_id="qa_001",
            query="Test query",
            assistant="Test response",
            compliance_score=0.8,
            verdict="COMPLIANT",
            supporting_chunks=2,
            contradicting_chunks=0,
            retrieved_chunks=[],
            insight="Test insight",
        )
        metric = RegulatoryMetric(
            session_id="test_session",
            assistant_id="test_assistant",
            n_interactions=1,
            compliance_score=0.8,
            verdict="COMPLIANT",
            total_supporting_chunks=2,
            total_contradicting_chunks=0,
            interactions=[interaction],
        )
        assert metric.compliance_score == 0.8
        assert metric.verdict == "COMPLIANT"
        assert metric.interactions[0].qa_id == "qa_001"


class TestLocalCorpusConnector:
    """Test suite for LocalCorpusConnector."""

    def test_load_documents_from_directory(self):
        with TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir)
            (corpus_dir / "policy1.md").write_text("Policy one content")
            (corpus_dir / "policy2.md").write_text("Policy two content")

            connector = LocalCorpusConnector(corpus_dir)
            documents = connector.load_documents()

            assert len(documents) == 2
            sources = {doc.source for doc in documents}
            assert "policy1.md" in sources
            assert "policy2.md" in sources

    def test_load_documents_nonexistent_directory(self):
        connector = LocalCorpusConnector("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            connector.load_documents()

    def test_load_documents_empty_directory(self):
        with TemporaryDirectory() as tmpdir:
            connector = LocalCorpusConnector(tmpdir)
            documents = connector.load_documents()
            assert len(documents) == 0


class TestDocumentRetriever:
    """Test suite for DocumentRetriever."""

    def test_retriever_initialization(self):
        from pygaussia.core.document_retriever import DocumentRetrieverConfig

        config = DocumentRetrieverConfig(
            top_k=5,
            similarity_threshold=0.5,
        )
        retriever = DocumentRetriever(_make_mock_embedder(), config)
        assert retriever._config.top_k == 5
        assert retriever._config.similarity_threshold == 0.5

    def test_load_corpus(self):
        retriever = DocumentRetriever(_make_mock_embedder())
        documents = [
            RegulatoryDocument(text="First document content", source="doc1.md"),
            RegulatoryDocument(text="Second document content", source="doc2.md"),
        ]
        num_chunks = retriever.load_corpus(documents)
        assert num_chunks >= 2


class TestContradictionChecker:
    """Test suite for ContradictionChecker."""

    def test_checker_initialization(self):
        checker = ContradictionChecker(_make_mock_reranker(), contradiction_threshold=0.5)
        assert checker._contradiction_threshold == 0.5


class TestRegulatoryMetric:
    """Test suite for Regulatory metric."""

    @patch.object(DocumentRetriever, "retrieve_merged")
    @patch.object(ContradictionChecker, "check")
    @patch.object(DocumentRetriever, "load_corpus")
    def test_batch_processing_compliant(
        self,
        mock_load_corpus,
        mock_check,
        mock_retrieve,
        regulatory_dataset_retriever,
        regulatory_dataset,
    ):
        mock_load_corpus.return_value = 10

        mock_retrieve.return_value = [
            RetrievedChunk(
                text="Customers can request removal from call lists",
                source="policy.md",
                chunk_index=0,
                similarity=0.85,
            ),
        ]

        mock_check.return_value = [
            RankedChunk(
                text="Customers can request removal from call lists",
                source="policy.md",
                chunk_index=0,
                similarity=0.85,
                reranker_score=0.8,
                verdict="SUPPORTS",
            ),
        ]

        connector = MockCorpusConnector()
        metric = Regulatory(
            regulatory_dataset_retriever,
            corpus_connector=connector,
            embedder=_make_mock_embedder(),
            reranker=_make_mock_reranker(),
            verbose=False,
        )

        dataset = regulatory_dataset
        metric.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation,
            language=dataset.language,
        )
        metric.on_process_complete()

        assert len(metric.metrics) == 1
        m = metric.metrics[0]
        assert isinstance(m, RegulatoryMetric)
        assert m.n_interactions == len(dataset.conversation)
        assert hasattr(m, "compliance_score")
        assert hasattr(m, "verdict")
        assert len(m.interactions) == len(dataset.conversation)

    @patch.object(DocumentRetriever, "retrieve_merged")
    @patch.object(ContradictionChecker, "check")
    @patch.object(DocumentRetriever, "load_corpus")
    def test_batch_processing_non_compliant(
        self,
        mock_load_corpus,
        mock_check,
        mock_retrieve,
        regulatory_dataset_retriever,
        regulatory_dataset,
    ):
        mock_load_corpus.return_value = 10

        mock_retrieve.return_value = [
            RetrievedChunk(
                text="No calls should be made at night",
                source="policy.md",
                chunk_index=0,
                similarity=0.85,
            ),
        ]

        mock_check.return_value = [
            RankedChunk(
                text="No calls should be made at night",
                source="policy.md",
                chunk_index=0,
                similarity=0.85,
                reranker_score=0.2,
                verdict="CONTRADICTS",
            ),
        ]

        connector = MockCorpusConnector()
        metric = Regulatory(
            regulatory_dataset_retriever,
            corpus_connector=connector,
            embedder=_make_mock_embedder(),
            reranker=_make_mock_reranker(),
            verbose=False,
        )

        dataset = regulatory_dataset
        metric.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation[:1],
            language=dataset.language,
        )
        metric.on_process_complete()

        assert len(metric.metrics) == 1
        assert metric.metrics[0].verdict == "NON_COMPLIANT"
        assert metric.metrics[0].total_contradicting_chunks == 1

    @patch.object(DocumentRetriever, "retrieve_merged")
    @patch.object(DocumentRetriever, "load_corpus")
    def test_batch_processing_irrelevant(
        self,
        mock_load_corpus,
        mock_retrieve,
        regulatory_dataset_retriever,
        regulatory_dataset,
    ):
        mock_load_corpus.return_value = 10
        mock_retrieve.return_value = []

        connector = MockCorpusConnector()
        metric = Regulatory(
            regulatory_dataset_retriever,
            corpus_connector=connector,
            embedder=_make_mock_embedder(),
            reranker=_make_mock_reranker(),
            verbose=False,
        )

        dataset = regulatory_dataset
        metric.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation[:1],
            language=dataset.language,
        )
        metric.on_process_complete()

        assert len(metric.metrics) == 1
        assert metric.metrics[0].verdict == "IRRELEVANT"
        assert metric.metrics[0].interactions[0].compliance_score == 0.5

    @patch.object(DocumentRetriever, "retrieve_merged")
    @patch.object(ContradictionChecker, "check")
    @patch.object(DocumentRetriever, "load_corpus")
    def test_run_method(
        self,
        mock_load_corpus,
        mock_check,
        mock_retrieve,
        regulatory_dataset_retriever,
    ):
        mock_load_corpus.return_value = 10

        mock_retrieve.return_value = [
            RetrievedChunk(
                text="Test chunk",
                source="test.md",
                chunk_index=0,
                similarity=0.8,
            ),
        ]

        mock_check.return_value = [
            RankedChunk(
                text="Test chunk",
                source="test.md",
                chunk_index=0,
                similarity=0.8,
                reranker_score=0.7,
                verdict="SUPPORTS",
            ),
        ]

        connector = MockCorpusConnector()
        metrics = Regulatory.run(
            regulatory_dataset_retriever,
            corpus_connector=connector,
            embedder=_make_mock_embedder(),
            reranker=_make_mock_reranker(),
            verbose=False,
        )

        assert isinstance(metrics, list)
        assert len(metrics) > 0

        for m in metrics:
            assert isinstance(m, RegulatoryMetric)

    def test_metric_attributes(self, regulatory_dataset_retriever):
        with patch.object(DocumentRetriever, "retrieve_merged") as mock_retrieve:
            with patch.object(ContradictionChecker, "check") as mock_check:
                with patch.object(DocumentRetriever, "load_corpus") as mock_load:
                    mock_load.return_value = 5
                    mock_retrieve.return_value = [
                        RetrievedChunk(
                            text="Test",
                            source="test.md",
                            chunk_index=0,
                            similarity=0.9,
                        ),
                    ]
                    mock_check.return_value = [
                        RankedChunk(
                            text="Test",
                            source="test.md",
                            chunk_index=0,
                            similarity=0.9,
                            reranker_score=0.8,
                            verdict="SUPPORTS",
                        ),
                    ]

                    connector = MockCorpusConnector()
                    metrics = Regulatory.run(
                        regulatory_dataset_retriever,
                        corpus_connector=connector,
                        embedder=_make_mock_embedder(),
                        reranker=_make_mock_reranker(),
                        verbose=False,
                    )

                    assert len(metrics) > 0
                    m = metrics[0]

                    required_attributes = [
                        "session_id",
                        "assistant_id",
                        "n_interactions",
                        "compliance_score",
                        "verdict",
                        "total_supporting_chunks",
                        "total_contradicting_chunks",
                        "interactions",
                    ]

                    for attr in required_attributes:
                        assert hasattr(m, attr), f"Missing attribute: {attr}"

                    assert len(m.interactions) > 0
                    interaction = m.interactions[0]
                    for attr in [
                        "qa_id",
                        "query",
                        "assistant",
                        "compliance_score",
                        "verdict",
                        "supporting_chunks",
                        "contradicting_chunks",
                        "retrieved_chunks",
                        "insight",
                    ]:
                        assert hasattr(interaction, attr), f"Missing interaction attribute: {attr}"


class TestComplianceScoring:
    """Test suite for compliance score computation."""

    def _make_metric(self) -> Regulatory:
        metric = Regulatory.__new__(Regulatory)
        metric.logger = MagicMock()
        metric.compliance_threshold = 0.5
        return metric

    def test_compute_verdict_all_supporting(self):
        metric = self._make_metric()
        verdict, score = metric._compute_verdict(supporting=5, contradicting=0)
        assert verdict == "COMPLIANT"
        assert score == 1.0

    def test_compute_verdict_all_contradicting(self):
        metric = self._make_metric()
        verdict, score = metric._compute_verdict(supporting=0, contradicting=3)
        assert verdict == "NON_COMPLIANT"
        assert score == 0.0

    def test_compute_verdict_mixed(self):
        metric = self._make_metric()
        verdict, score = metric._compute_verdict(supporting=3, contradicting=1)
        assert verdict == "COMPLIANT"
        assert score == 0.75

    def test_compute_verdict_irrelevant(self):
        metric = self._make_metric()
        verdict, score = metric._compute_verdict(supporting=0, contradicting=0)
        assert verdict == "IRRELEVANT"
        assert score == 0.5
