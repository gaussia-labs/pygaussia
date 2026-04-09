"""Pytest configuration and shared fixtures for Gaussia tests."""

import warnings

# Suppress all problematic warnings from third-party libraries before importing
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", message=".*Tensorflow not installed.*")
warnings.filterwarnings("ignore", message=".*ParametricUMAP will be unavailable.*")

try:
    import hdbscan  # noqa: F401
    import umap  # noqa: F401
except Exception:
    pass

import pytest
from pydantic import SecretStr

from gaussia.schemas.common import Batch, Dataset


def pytest_configure(config):
    """Configure pytest to skip assertion rewriting for problematic modules."""
    assertrewrite = config.pluginmanager.get_plugin("assertion")
    if assertrewrite and hasattr(assertrewrite, "_dont_rewrite"):
        assertrewrite._dont_rewrite.add("hdbscan")
        assertrewrite._dont_rewrite.add("umap")


from tests.fixtures.mock_data import (
    create_agentic_dataset,
    create_bestof_dataset,
    create_bias_dataset,
    create_context_dataset,
    create_conversational_dataset,
    create_emotional_dataset,
    create_regulatory_dataset,
    create_sample_batch,
    create_sample_dataset,
    create_toxicity_dataset,
    create_vision_dataset,
)
from tests.fixtures.mock_retriever import (
    AgenticDatasetRetriever,
    BestOfDatasetRetriever,
    BiasDatasetRetriever,
    ContextDatasetRetriever,
    ConversationalDatasetRetriever,
    EmotionalDatasetRetriever,
    EmptyRetriever,
    MockRetriever,
    MultipleDatasetRetriever,
    RegulatoryDatasetRetriever,
    SingleDatasetRetriever,
    ToxicityDatasetRetriever,
    VisionDatasetRetriever,
)


@pytest.fixture
def sample_batch() -> Batch:
    """Fixture providing a sample Batch object."""
    return create_sample_batch()


@pytest.fixture
def sample_dataset() -> Dataset:
    """Fixture providing a sample Dataset object."""
    return create_sample_dataset()


@pytest.fixture
def emotional_dataset() -> Dataset:
    """Fixture providing an emotionally rich dataset."""
    return create_emotional_dataset()


@pytest.fixture
def conversational_dataset() -> Dataset:
    """Fixture providing a conversational dataset."""
    return create_conversational_dataset()


@pytest.fixture
def bias_dataset() -> Dataset:
    """Fixture providing a bias testing dataset."""
    return create_bias_dataset()


@pytest.fixture
def toxicity_dataset() -> Dataset:
    """Fixture providing a toxicity testing dataset."""
    return create_toxicity_dataset()


@pytest.fixture
def context_dataset() -> Dataset:
    """Fixture providing a context testing dataset."""
    return create_context_dataset()


@pytest.fixture
def bestof_dataset() -> Dataset:
    """Fixture providing a best-of testing dataset."""
    return create_bestof_dataset()


@pytest.fixture
def agentic_dataset() -> list[Dataset]:
    """Fixture providing an agentic testing dataset."""
    return create_agentic_dataset()


@pytest.fixture
def regulatory_dataset() -> Dataset:
    """Fixture providing a regulatory testing dataset."""
    return create_regulatory_dataset()


@pytest.fixture
def mock_retriever() -> type[MockRetriever]:
    """Fixture providing MockRetriever class."""
    return MockRetriever


@pytest.fixture
def empty_retriever() -> type[EmptyRetriever]:
    """Fixture providing EmptyRetriever class."""
    return EmptyRetriever


@pytest.fixture
def single_dataset_retriever() -> type[SingleDatasetRetriever]:
    """Fixture providing SingleDatasetRetriever class."""
    return SingleDatasetRetriever


@pytest.fixture
def multiple_dataset_retriever() -> type[MultipleDatasetRetriever]:
    """Fixture providing MultipleDatasetRetriever class."""
    return MultipleDatasetRetriever


@pytest.fixture
def emotional_dataset_retriever() -> type[EmotionalDatasetRetriever]:
    """Fixture providing EmotionalDatasetRetriever class."""
    return EmotionalDatasetRetriever


@pytest.fixture
def conversational_dataset_retriever() -> type[ConversationalDatasetRetriever]:
    """Fixture providing ConversationalDatasetRetriever class."""
    return ConversationalDatasetRetriever


@pytest.fixture
def bias_dataset_retriever() -> type[BiasDatasetRetriever]:
    """Fixture providing BiasDatasetRetriever class."""
    return BiasDatasetRetriever


@pytest.fixture
def toxicity_dataset_retriever() -> type[ToxicityDatasetRetriever]:
    """Fixture providing ToxicityDatasetRetriever class."""
    return ToxicityDatasetRetriever


@pytest.fixture
def context_dataset_retriever() -> type[ContextDatasetRetriever]:
    """Fixture providing ContextDatasetRetriever class."""
    return ContextDatasetRetriever


@pytest.fixture
def bestof_dataset_retriever() -> type[BestOfDatasetRetriever]:
    """Fixture providing BestOfDatasetRetriever class."""
    return BestOfDatasetRetriever


@pytest.fixture
def agentic_dataset_retriever() -> type[AgenticDatasetRetriever]:
    """Fixture providing AgenticDatasetRetriever class."""
    return AgenticDatasetRetriever


@pytest.fixture
def vision_dataset() -> Dataset:
    """Fixture providing a vision testing dataset."""
    return create_vision_dataset()


@pytest.fixture
def vision_dataset_retriever() -> type[VisionDatasetRetriever]:
    """Fixture providing VisionDatasetRetriever class."""
    return VisionDatasetRetriever


@pytest.fixture
def regulatory_dataset_retriever() -> type[RegulatoryDatasetRetriever]:
    """Fixture providing RegulatoryDatasetRetriever class."""
    return RegulatoryDatasetRetriever


@pytest.fixture
def mock_api_key() -> SecretStr:
    """Fixture providing a mock API key."""
    return SecretStr("mock_api_key_for_testing")


@pytest.fixture
def mock_guardian_config() -> dict:
    """Fixture providing mock guardian configuration."""
    return {
        "model": "mock-guardian-model",
        "api_key": "mock_api_key",
        "url": "https://mock-api.example.com",
        "temperature": 0.0,
    }


@pytest.fixture(autouse=True)
def reset_metrics():
    """Fixture to reset metric state between tests."""
    return


@pytest.fixture
def mock_embedding_model() -> str:
    """Fixture providing a mock embedding model name."""
    return "all-MiniLM-L6-v2"


@pytest.fixture
def mock_judge_config() -> dict:
    """Fixture providing mock judge configuration."""
    return {
        "model": "mock-judge-model",
        "temperature": 0.0,
        "chain_of_thought": True,
    }
