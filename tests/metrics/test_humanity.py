"""Unit tests for Humanity metric."""

from pygaussia.metrics.humanity import Humanity
from pygaussia.schemas.humanity import HumanityMetric
from tests.fixtures.mock_data import create_sample_batch
from tests.fixtures.mock_retriever import MockRetriever


class TestHumanityMetric:
    """Test suite for Humanity metric."""

    def test_humanity_initialization(self, emotional_dataset_retriever):
        """Test that Humanity metric initializes correctly."""
        humanity = Humanity(emotional_dataset_retriever)
        assert humanity is not None
        assert hasattr(humanity, "emotion_columns")
        assert len(humanity.emotion_columns) == 8
        expected_emotions = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]
        assert humanity.emotion_columns == expected_emotions

    def test_tokenize(self, emotional_dataset_retriever):
        """Test text tokenization."""
        humanity = Humanity(emotional_dataset_retriever)

        text = "Hello, World! This is a test."
        tokens = humanity._tokenize(text)

        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert len(tokens) == 6

    def test_tokenize_empty_string(self, emotional_dataset_retriever):
        """Test tokenization of empty string."""
        humanity = Humanity(emotional_dataset_retriever)
        tokens = humanity._tokenize("")
        assert tokens == []

    def test_load_emotion_lexicon(self, emotional_dataset_retriever):
        """Test loading emotion lexicon."""
        humanity = Humanity(emotional_dataset_retriever)
        lexicon = humanity._load_emotion_lexicon()

        assert isinstance(lexicon, dict)
        assert len(lexicon) > 0

        for word, emotions in lexicon.items():
            assert isinstance(word, str)
            assert isinstance(emotions, list)
            for emotion in emotions:
                assert emotion in humanity.emotion_columns

    def test_get_emotion_distribution(self, emotional_dataset_retriever):
        """Test emotion distribution calculation."""
        humanity = Humanity(emotional_dataset_retriever)
        lexicon = humanity._load_emotion_lexicon()

        text = "I feel joyful and happy today"
        distribution = humanity._get_emotion_distribution(text, lexicon, humanity.emotion_columns)

        assert isinstance(distribution, dict)
        assert len(distribution) == 8

        for emotion in humanity.emotion_columns:
            assert emotion in distribution
            assert isinstance(distribution[emotion], (int, float))
            assert 0 <= distribution[emotion] <= 1

        total = sum(distribution.values())
        assert total <= 1.0 or total == 0.0

    def test_get_emotion_distribution_empty_text(self, emotional_dataset_retriever):
        """Test emotion distribution with empty text."""
        humanity = Humanity(emotional_dataset_retriever)
        lexicon = humanity._load_emotion_lexicon()

        distribution = humanity._get_emotion_distribution("", lexicon, humanity.emotion_columns)

        assert isinstance(distribution, dict)
        assert all(v == 0 for v in distribution.values())

    def test_get_emotion_distribution_no_emotional_words(self, emotional_dataset_retriever):
        """Test emotion distribution with text containing no emotional words."""
        humanity = Humanity(emotional_dataset_retriever)
        lexicon = humanity._load_emotion_lexicon()

        text = "xyz abc def"
        distribution = humanity._get_emotion_distribution(text, lexicon, humanity.emotion_columns)

        assert isinstance(distribution, dict)
        assert all(v == 0 for v in distribution.values())

    def test_emotional_entropy(self, emotional_dataset_retriever):
        """Test emotional entropy calculation."""
        humanity = Humanity(emotional_dataset_retriever)

        uniform_distribution = dict.fromkeys(humanity.emotion_columns, 1 / 8)
        entropy = humanity._emotional_entropy(uniform_distribution)

        assert isinstance(entropy, float)
        assert entropy >= 0
        assert entropy <= 3.0
        assert abs(entropy - 3.0) < 0.01

    def test_emotional_entropy_zero_distribution(self, emotional_dataset_retriever):
        """Test emotional entropy with zero distribution."""
        humanity = Humanity(emotional_dataset_retriever)

        zero_distribution = dict.fromkeys(humanity.emotion_columns, 0)
        entropy = humanity._emotional_entropy(zero_distribution)

        assert entropy == 0.0

    def test_emotional_entropy_single_emotion(self, emotional_dataset_retriever):
        """Test emotional entropy with single dominant emotion."""
        humanity = Humanity(emotional_dataset_retriever)

        single_emotion_distribution = {
            "Joy": 1.0,
            "Anger": 0.0,
            "Anticipation": 0.0,
            "Disgust": 0.0,
            "Fear": 0.0,
            "Sadness": 0.0,
            "Surprise": 0.0,
            "Trust": 0.0,
        }
        entropy = humanity._emotional_entropy(single_emotion_distribution)

        assert entropy == 0.0

    def test_batch_processing(self, emotional_dataset_retriever, emotional_dataset):
        """Test batch processing of interactions."""
        humanity = Humanity(emotional_dataset_retriever)

        dataset = emotional_dataset
        batch_interactions = dataset.conversation

        humanity.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=batch_interactions,
            language=dataset.language,
        )

        assert len(humanity.metrics) == len(batch_interactions)

        for metric in humanity.metrics:
            assert isinstance(metric, HumanityMetric)
            assert hasattr(metric, "humanity_assistant_emotional_entropy")
            assert hasattr(metric, "humanity_ground_truth_spearman")

            assert isinstance(metric.humanity_assistant_emotional_entropy, float)
            assert isinstance(metric.humanity_ground_truth_spearman, float)

            assert metric.humanity_assistant_emotional_entropy >= 0
            assert -1.0 <= metric.humanity_ground_truth_spearman <= 1.0

    def test_batch_with_emotional_content(self, emotional_dataset_retriever, emotional_dataset):
        """Test batch processing with emotionally rich content."""
        humanity = Humanity(emotional_dataset_retriever)

        dataset = emotional_dataset

        humanity.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation,
            language=dataset.language,
        )

        assert len(humanity.metrics) > 0

        for metric in humanity.metrics:
            for emotion in ["joy", "sadness", "fear", "anger", "trust", "anticipation", "disgust", "surprise"]:
                emotion_attr = f"humanity_assistant_{emotion}"
                assert hasattr(metric, emotion_attr)
                value = getattr(metric, emotion_attr)
                assert isinstance(value, (int, float))
                assert value >= 0

    def test_run_method(self, emotional_dataset_retriever):
        """Test the run class method."""
        metrics = Humanity.run(emotional_dataset_retriever, verbose=False)

        assert isinstance(metrics, list)
        assert len(metrics) > 0

        for metric in metrics:
            assert isinstance(metric, HumanityMetric)

    def test_spearman_correlation_calculation(self, mock_retriever):
        """Test Spearman correlation calculation between assistant and ground truth."""
        from tests.fixtures.mock_data import create_sample_dataset

        batch = create_sample_batch(
            qa_id="qa_001",
            query="How are you?",
            assistant="I feel joyful and happy with anticipation and trust",
            ground_truth_assistant="I am joyful and happy with trust and anticipation",
        )

        dataset = create_sample_dataset(conversation=[batch])
        retriever = type("TestRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})

        humanity = Humanity(retriever)
        humanity.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation,
            language=dataset.language,
        )

        assert len(humanity.metrics) == 1
        metric = humanity.metrics[0]

        assert metric.humanity_ground_truth_spearman is not None
        assert isinstance(metric.humanity_ground_truth_spearman, float)

    def test_spearman_zero_std(self, mock_retriever):
        """Test Spearman correlation when standard deviation is zero."""
        from tests.fixtures.mock_data import create_sample_dataset

        batch = create_sample_batch(
            qa_id="qa_001",
            query="Test query",
            assistant="xyz abc def",
            ground_truth_assistant="uvw rst lmn",
        )

        dataset = create_sample_dataset(conversation=[batch])
        retriever = type("TestRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})

        humanity = Humanity(retriever)
        humanity.batch(
            session_id=dataset.session_id,
            context=dataset.context,
            assistant_id=dataset.assistant_id,
            batch=dataset.conversation,
            language=dataset.language,
        )

        assert len(humanity.metrics) == 1
        metric = humanity.metrics[0]

        assert metric.humanity_ground_truth_spearman == 0

    def test_metrics_attributes(self, emotional_dataset_retriever):
        """Test that all expected attributes exist in HumanityMetric."""
        metrics = Humanity.run(emotional_dataset_retriever, verbose=False)

        assert len(metrics) > 0
        metric = metrics[0]

        required_attributes = [
            "session_id",
            "qa_id",
            "assistant_id",
            "humanity_assistant_emotional_entropy",
            "humanity_ground_truth_spearman",
        ]

        for attr in required_attributes:
            assert hasattr(metric, attr)

        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        for emotion in emotions:
            emotion_attr = f"humanity_assistant_{emotion}"
            assert hasattr(metric, emotion_attr)

    def test_verbose_mode(self, emotional_dataset_retriever):
        """Test that verbose mode works without errors."""
        metrics = Humanity.run(emotional_dataset_retriever, verbose=True)
        assert isinstance(metrics, list)

    def test_no_ground_truth(self, mock_retriever):
        """Test behavior when ground truth is not provided."""
        from tests.fixtures.mock_data import create_sample_dataset

        batch = create_sample_batch(
            qa_id="qa_001",
            query="Test query",
            assistant="I feel joyful today",
            ground_truth_assistant="",
        )

        dataset = create_sample_dataset(conversation=[batch])
        retriever = type("TestRetriever", (MockRetriever,), {"load_dataset": lambda self: [dataset]})

        metrics = Humanity.run(retriever, verbose=False)

        assert len(metrics) > 0
        metric = metrics[0]
        assert hasattr(metric, "humanity_ground_truth_spearman")
