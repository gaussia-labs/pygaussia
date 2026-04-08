"""Tests for Embedding Group Extractor."""

import numpy as np
import pytest

from pygaussia.core.embedder import Embedder
from pygaussia.extractors.embedding import EmbeddingGroupExtractor
from pygaussia.schemas.toxicity import GroupDetection


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self._call_count = 0

    def encode(self, sentences: list[str]) -> np.ndarray:
        embeddings = []
        for text in sentences:
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self.embedding_dim).astype(np.float32)
            embeddings.append(emb)
        return np.array(embeddings)


class TestEmbeddingGroupExtractor:
    """Test suite for EmbeddingGroupExtractor."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        embedder = MockEmbedder()
        group_prototypes = {
            "gender": ["man", "woman"],
            "race": ["white", "black", "asian"],
        }

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        assert extractor.default_threshold == 0.50
        assert extractor.normalize_embeddings is True
        assert len(extractor._proto_embs) == 2
        assert "gender" in extractor._proto_embs
        assert "race" in extractor._proto_embs

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        embedder = MockEmbedder()
        group_prototypes = {"category": ["proto1", "proto2"]}
        thresholds = {"category": 0.7}

        extractor = EmbeddingGroupExtractor(
            embedder=embedder,
            group_prototypes=group_prototypes,
            thresholds=thresholds,
            default_threshold=0.6,
            normalize_embeddings=False,
        )

        assert extractor.default_threshold == 0.6
        assert extractor.normalize_embeddings is False
        assert extractor.thresholds == {"category": 0.7}

    def test_initialization_empty_prototypes_raises(self):
        """Test initialization raises error with empty prototypes."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="group_prototypes must be non-empty"):
            EmbeddingGroupExtractor(embedder=embedder, group_prototypes={})

    def test_initialization_empty_group_prototypes_raises(self):
        """Test initialization raises error when a group has empty prototypes."""
        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="group_prototypes\\['empty_group'\\] is empty"):
            EmbeddingGroupExtractor(embedder=embedder, group_prototypes={"empty_group": []})

    def test_detect_one_basic(self):
        """Test detect_one with basic input."""
        embedder = MockEmbedder()
        group_prototypes = {
            "gender": ["man", "woman", "person"],
        }

        extractor = EmbeddingGroupExtractor(
            embedder=embedder,
            group_prototypes=group_prototypes,
            default_threshold=0.0,
        )

        result = extractor.detect_one("The man walked to the store")

        assert "gender" in result
        assert isinstance(result["gender"], GroupDetection)
        assert hasattr(result["gender"], "present")
        assert hasattr(result["gender"], "score")
        assert hasattr(result["gender"], "best_prototype")
        assert hasattr(result["gender"], "best_prototype_index")

    def test_detect_one_non_string_raises(self):
        """Test detect_one raises error for non-string input."""
        embedder = MockEmbedder()
        group_prototypes = {"gender": ["man", "woman"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        with pytest.raises(TypeError, match="text must be a string"):
            extractor.detect_one(123)

    def test_detect_one_with_thresholds(self):
        """Test detect_one respects per-group thresholds."""
        embedder = MockEmbedder()
        group_prototypes = {
            "gender": ["man", "woman"],
            "race": ["white", "black"],
        }
        thresholds = {
            "gender": 0.9,
            "race": -1.0,
        }

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes, thresholds=thresholds)

        result = extractor.detect_one("some text")

        assert result["race"].present is True

    def test_detect_batch_basic(self):
        """Test detect_batch with basic input."""
        embedder = MockEmbedder()
        group_prototypes = {
            "gender": ["man", "woman"],
        }

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes, default_threshold=0.0)

        texts = ["The man is here", "The woman is there", "A person walks"]
        results = extractor.detect_batch(texts)

        assert len(results) == 3
        for result in results:
            assert "gender" in result
            assert isinstance(result["gender"], GroupDetection)

    def test_detect_batch_non_list_raises(self):
        """Test detect_batch raises error for non-list input."""
        embedder = MockEmbedder()
        group_prototypes = {"gender": ["man", "woman"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        with pytest.raises(TypeError, match="texts must be a list"):
            extractor.detect_batch("not a list")

    def test_detect_batch_non_string_elements_raises(self):
        """Test detect_batch raises error for non-string elements."""
        embedder = MockEmbedder()
        group_prototypes = {"gender": ["man", "woman"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        with pytest.raises(TypeError, match="texts must be a list"):
            extractor.detect_batch(["valid", 123, "also valid"])

    def test_detect_batch_single_item(self):
        """Test detect_batch with single item list."""
        embedder = MockEmbedder()
        group_prototypes = {"gender": ["man", "woman"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes, default_threshold=0.0)

        results = extractor.detect_batch(["hello world"])

        assert len(results) == 1
        assert "gender" in results[0]

    def test_encode_normalizes_1d_embedding(self):
        """Test _encode handles 1D embedding arrays."""
        embedder = MockEmbedder()
        group_prototypes = {"test": ["single"]}

        extractor = EmbeddingGroupExtractor(
            embedder=embedder, group_prototypes=group_prototypes, normalize_embeddings=True
        )

        assert "test" in extractor._proto_embs
        assert extractor._proto_embs["test"].shape[0] == 1

    def test_encode_with_non_normalizing_embedder(self):
        """Test that extractor normalizes even when embedder returns raw vectors."""
        embedder = MockEmbedder()
        group_prototypes = {"test": ["prototype1", "prototype2"]}

        extractor = EmbeddingGroupExtractor(
            embedder=embedder, group_prototypes=group_prototypes, normalize_embeddings=True
        )

        result = extractor.detect_one("test text")
        assert "test" in result
        assert -1.0 <= result["test"].score <= 1.0

    def test_group_detection_score_range(self):
        """Test that detection scores are in valid range."""
        embedder = MockEmbedder()
        group_prototypes = {"category": ["word1", "word2", "word3"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        result = extractor.detect_one("test sentence")

        assert -1.0 <= result["category"].score <= 1.0

    def test_best_prototype_index_valid(self):
        """Test that best_prototype_index is valid."""
        embedder = MockEmbedder()
        prototypes = ["proto1", "proto2", "proto3"]
        group_prototypes = {"category": prototypes}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        result = extractor.detect_one("test text")

        assert 0 <= result["category"].best_prototype_index < len(prototypes)
        assert result["category"].best_prototype in prototypes

    def test_multiple_groups(self):
        """Test detection with multiple groups."""
        embedder = MockEmbedder()
        group_prototypes = {
            "gender": ["male", "female"],
            "age": ["young", "old"],
            "occupation": ["doctor", "engineer", "teacher"],
        }

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes, default_threshold=0.0)

        result = extractor.detect_one("The young female doctor")

        assert len(result) == 3
        assert all(group in result for group in ["gender", "age", "occupation"])

    def test_batch_consistency(self):
        """Test that batch and single detection give same results."""
        embedder = MockEmbedder()
        group_prototypes = {"category": ["word1", "word2"]}

        extractor = EmbeddingGroupExtractor(embedder=embedder, group_prototypes=group_prototypes)

        texts = ["text one", "text two"]
        batch_results = extractor.detect_batch(texts)
        single_results = [extractor.detect_one(t) for t in texts]

        for i in range(len(texts)):
            assert batch_results[i]["category"].score == pytest.approx(single_results[i]["category"].score, abs=0.01)
