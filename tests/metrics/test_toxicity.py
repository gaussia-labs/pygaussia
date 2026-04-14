"""Unit tests for Toxicity metric."""

from collections import Counter
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from gaussia.core.embedder import Embedder
from gaussia.metrics.toxicity import Toxicity
from gaussia.schemas.toxicity import ToxicityMetric
from gaussia.statistical import BayesianMode, FrequentistMode
from tests.fixtures.mock_data import create_sample_batch, create_sample_dataset


class MockToxicityDataset:
    """Mock toxicity dataset entry."""

    def __init__(self, word: str, category: str = "offensive"):
        self.word = word
        self.category = category


class MockToxicityLoader:
    """Mock toxicity loader."""

    def load(self, language: str):
        return [
            MockToxicityDataset("hate", "offensive"),
            MockToxicityDataset("stupid", "offensive"),
            MockToxicityDataset("idiot", "offensive"),
            MockToxicityDataset("kill", "violence"),
            MockToxicityDataset("destroy", "violence"),
        ]


class MockGroupExtractor:
    """Mock group extractor."""

    def __init__(self, *args, **kwargs):
        pass

    def detect_batch(self, texts):
        return [
            {
                "male": Mock(present=True, confidence=0.9),
                "female": Mock(present=False, confidence=0.1),
            }
            for _ in texts
        ]


def _make_mock_embedder() -> MagicMock:
    mock = MagicMock(spec=Embedder)
    mock.encode.return_value = np.array([[0.1, 0.2]])
    return mock


class TestToxicityMetric:
    """Test suite for Toxicity metric."""

    def test_toxicity_initialization_frequentist(self, toxicity_dataset_retriever):
        """Test that Toxicity metric initializes correctly with frequentist mode."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=FrequentistMode(),
        )

        assert toxicity is not None
        assert isinstance(toxicity.statistical_mode, FrequentistMode)
        assert toxicity.statistical_mode.get_result_type() == "point_estimate"
        assert toxicity.min_cluster_size == 5
        assert toxicity.umap_n_components == 2

    def test_toxicity_initialization_bayesian(self, toxicity_dataset_retriever):
        """Test that Toxicity metric initializes correctly with Bayesian mode."""
        bayesian_mode = BayesianMode(mc_samples=5000, ci_level=0.95)
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=bayesian_mode,
        )

        assert toxicity is not None
        assert isinstance(toxicity.statistical_mode, BayesianMode)
        assert toxicity.statistical_mode.get_result_type() == "distribution"

    def test_toxicity_default_statistical_mode(self, toxicity_dataset_retriever):
        """Test that Toxicity defaults to FrequentistMode."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        assert isinstance(toxicity.statistical_mode, FrequentistMode)

    def test_toxicity_custom_configuration(self, toxicity_dataset_retriever):
        """Test Toxicity metric with custom configuration."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            toxicity_min_cluster_size=10,
            toxicity_cluster_selection_epsilon=0.05,
            umap_n_components=3,
            umap_n_neighbors=20,
            group_prototypes={"gender": ["male", "female"]},
            w_DR=0.5,
            w_ASB=0.3,
            w_DTO=0.2,
        )

        assert toxicity.min_cluster_size == 10
        assert toxicity.cluster_selection_epsilon == 0.05
        assert toxicity.umap_n_components == 3
        assert toxicity.umap_n_neighbors == 20
        assert toxicity.w_DR == 0.5
        assert toxicity.w_ASB == 0.3
        assert toxicity.w_DTO == 0.2

    def test_tokenize(self, toxicity_dataset_retriever):
        """Test text tokenization."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        text = "Hello, World! This is a test."
        tokens = toxicity._tokenize(text)

        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_unicode(self, toxicity_dataset_retriever):
        """Test tokenization with unicode characters."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        text = "Hello 世界 Здравствуй"
        tokens = toxicity._tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_build_toxic_set(self, toxicity_dataset_retriever):
        """Test building toxic word set."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        toxic_set = toxicity._build_toxic_set("english")

        assert isinstance(toxic_set, set)
        assert "hate" in toxic_set
        assert "stupid" in toxic_set
        assert "idiot" in toxic_set
        assert "kill" in toxic_set
        assert "destroy" in toxic_set

    def test_count_toxic_in_counter(self, toxicity_dataset_retriever):
        """Test counting toxic words in Counter."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        counter = Counter(["hello", "hate", "world", "stupid", "test"])
        toxic_set = {"hate", "stupid", "idiot"}

        toxic_words, total_words = toxicity._count_toxic_in_counter(counter, toxic_set)

        assert toxic_words == 2
        assert total_words == 5

    def test_count_toxic_in_counter_no_toxic(self, toxicity_dataset_retriever):
        """Test counting when no toxic words present."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        counter = Counter(["hello", "world", "test"])
        toxic_set = {"hate", "stupid", "idiot"}

        toxic_words, total_words = toxicity._count_toxic_in_counter(counter, toxic_set)

        assert toxic_words == 0
        assert total_words == 3

    def test_toxicity_score_text(self, toxicity_dataset_retriever):
        """Test toxicity score calculation for text."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        toxic_set = {"hate", "stupid", "idiot"}

        text = "I hate stupid people and love coding"
        score = toxicity._toxicity_score_text(text, toxic_set)

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score == pytest.approx(2 / 7, abs=0.01)

    def test_toxicity_score_text_no_toxic(self, toxicity_dataset_retriever):
        """Test toxicity score with no toxic words."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        toxic_set = {"hate", "stupid", "idiot"}
        text = "I love coding and enjoy learning"
        score = toxicity._toxicity_score_text(text, toxic_set)

        assert score == 0.0

    def test_toxicity_score_text_empty(self, toxicity_dataset_retriever):
        """Test toxicity score with empty text."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        toxic_set = {"hate", "stupid", "idiot"}
        score = toxicity._toxicity_score_text("", toxic_set)

        assert score == 0.0

    def test_is_toxic_text(self, toxicity_dataset_retriever):
        """Test binary toxicity classification."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
        )

        toxic_set = {"hate", "stupid", "idiot"}

        toxic_text = "I hate stupid people"
        clean_text = "I love programming"

        assert toxicity._is_toxic_text(toxic_text, toxic_set, threshold=0.1) is True
        assert toxicity._is_toxic_text(clean_text, toxic_set, threshold=0.1) is False

    def test_normalize_weights(self, toxicity_dataset_retriever):
        """Test weight normalization."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            w_DR=2.0,
            w_ASB=3.0,
            w_DTO=1.0,
        )

        w_DR, w_ASB, w_DTO = toxicity._normalize_weights()

        assert w_DR == pytest.approx(2.0 / 6.0, abs=0.01)
        assert w_ASB == pytest.approx(3.0 / 6.0, abs=0.01)
        assert w_DTO == pytest.approx(1.0 / 6.0, abs=0.01)
        assert w_DR + w_ASB + w_DTO == pytest.approx(1.0, abs=0.01)

    def test_normalize_weights_zero(self, toxicity_dataset_retriever):
        """Test weight normalization when all weights are zero."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            w_DR=0.0,
            w_ASB=0.0,
            w_DTO=0.0,
        )

        w_DR, w_ASB, w_DTO = toxicity._normalize_weights()

        assert w_DR == pytest.approx(1.0 / 3.0, abs=0.01)
        assert w_ASB == pytest.approx(1.0 / 3.0, abs=0.01)
        assert w_DTO == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_compute_DR_frequentist(self, toxicity_dataset_retriever):
        """Test DR computation in frequentist mode."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=FrequentistMode(),
        )

        N_i = {"male": 60, "female": 40}
        groups = ["male", "female"]

        DR = toxicity._compute_DR(N_i, groups)

        assert isinstance(DR, float)
        assert DR >= 0

    def test_compute_DR_bayesian(self, toxicity_dataset_retriever):
        """Test DR computation in Bayesian mode."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=BayesianMode(mc_samples=1000),
        )

        N_i = {"male": 60, "female": 40}
        groups = ["male", "female"]

        DR = toxicity._compute_DR(N_i, groups)

        assert isinstance(DR, dict)
        assert "mean" in DR
        assert "ci_low" in DR
        assert "ci_high" in DR

    def test_compute_DTO_frequentist(self, toxicity_dataset_retriever):
        """Test DTO computation in frequentist mode."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=FrequentistMode(),
        )

        N_i = {"male": 100, "female": 100}
        K_i = {"male": 20, "female": 10}
        groups = ["male", "female"]

        DTO = toxicity._compute_DTO(N_i, K_i, groups)

        assert isinstance(DTO, float)
        assert DTO >= 0

    def test_compute_ASB(self, toxicity_dataset_retriever):
        """Test ASB computation (placeholder)."""
        toxicity = Toxicity(
            toxicity_dataset_retriever,
            embedder=_make_mock_embedder(),
            toxicity_loader=MockToxicityLoader,
            group_prototypes={"gender": ["male", "female"]},
            statistical_mode=FrequentistMode(),
        )

        ASB = toxicity._compute_asb([], [], [])

        assert isinstance(ASB, float)
        assert ASB == 0.0

    @patch("gaussia.metrics.toxicity.hdbscan")
    @patch("gaussia.metrics.toxicity.umap")
    def test_batch_processing(self, mock_umap, mock_hdbscan):
        """Test batch processing with mocked dependencies."""
        from tests.fixtures.mock_retriever import MockRetriever

        mock_embedder = _make_mock_embedder()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_umap.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([0, 0])
        mock_hdbscan.HDBSCAN.return_value = mock_hdbscan_instance

        batch_data = [
            create_sample_batch(qa_id="qa_001", assistant="This is a clean response"),
            create_sample_batch(qa_id="qa_002", assistant="Another clean response"),
        ]

        dataset = create_sample_dataset(conversation=batch_data)
        retriever = type("TestRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})

        with patch("gaussia.metrics.toxicity.EmbeddingGroupExtractor", MockGroupExtractor):
            toxicity = Toxicity(
                retriever,
                embedder=mock_embedder,
                toxicity_loader=MockToxicityLoader,
                group_prototypes={"gender": ["male", "female"]},
            )

            toxicity.batch(
                session_id=dataset.session_id,
                context=dataset.context,
                assistant_id=dataset.assistant_id,
                batch=dataset.conversation,
                language=dataset.language,
            )
            toxicity.on_process_complete()

            assert len(toxicity.metrics) == 1
            metric = toxicity.metrics[0]
            assert isinstance(metric, ToxicityMetric)

    def test_metric_attributes(self):
        """Test that all expected attributes exist in ToxicityMetric."""
        from tests.fixtures.mock_retriever import MockRetriever

        mock_embedder = _make_mock_embedder()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        batch_data = [
            create_sample_batch(qa_id="qa_001", assistant="Clean response one"),
            create_sample_batch(qa_id="qa_002", assistant="Clean response two"),
        ]

        dataset = create_sample_dataset(conversation=batch_data)
        retriever = type("TestRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})

        with (
            patch("gaussia.metrics.toxicity.hdbscan") as mock_hdbscan,
            patch("gaussia.metrics.toxicity.umap") as mock_umap,
            patch("gaussia.metrics.toxicity.EmbeddingGroupExtractor", MockGroupExtractor),
        ):
            mock_umap_instance = MagicMock()
            mock_umap_instance.fit_transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_umap.UMAP.return_value = mock_umap_instance

            mock_hdbscan_instance = MagicMock()
            mock_hdbscan_instance.fit_predict.return_value = np.array([0, 0])
            mock_hdbscan.HDBSCAN.return_value = mock_hdbscan_instance

            toxicity = Toxicity(
                retriever,
                embedder=mock_embedder,
                toxicity_loader=MockToxicityLoader,
                group_prototypes={"gender": ["male", "female"]},
            )

            toxicity.batch(
                session_id=dataset.session_id,
                context=dataset.context,
                assistant_id=dataset.assistant_id,
                batch=dataset.conversation,
                language=dataset.language,
            )
            toxicity.on_process_complete()

            metric = toxicity.metrics[0]

            required_attributes = [
                "session_id",
                "assistant_id",
                "cluster_profiling",
                "assistant_space",
                "group_profiling",
            ]

            for attr in required_attributes:
                assert hasattr(metric, attr)

            assert hasattr(metric.assistant_space, "embeddings")
            assert hasattr(metric.assistant_space, "latent_space")
            assert hasattr(metric.assistant_space, "cluster_labels")

    def test_group_prototypes_required(self, toxicity_dataset_retriever):
        """Test that group_prototypes is required when group_extractor is None."""
        with pytest.raises(ValueError, match="group_prototypes must be provided"):
            Toxicity(
                toxicity_dataset_retriever,
                embedder=_make_mock_embedder(),
                toxicity_loader=MockToxicityLoader,
                group_prototypes=None,
                group_extractor=None,
            )
