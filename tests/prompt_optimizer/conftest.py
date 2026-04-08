import pytest
from unittest.mock import MagicMock

from pygaussia.core.retriever import Retriever
from pygaussia.schemas.common import Batch, Dataset


def _make_batch(qa_id: str, query: str) -> Batch:
    return Batch(
        qa_id=qa_id,
        query=query,
        assistant="",
        ground_truth_assistant=f"Expected answer for: {query}",
        ground_truth_agentic={},
    )


def _make_dataset(session_id: str, context: str, queries: list[str]) -> Dataset:
    return Dataset(
        session_id=session_id,
        assistant_id="test-bot",
        language="english",
        context=context,
        conversation=[_make_batch(f"{session_id}-{i}", q) for i, q in enumerate(queries)],
    )


MOCK_DATASETS = [
    _make_dataset("session-1", "Context about topic A.", ["Query 1A", "Query 2A"]),
    _make_dataset("session-2", "Context about topic B.", ["Query 1B", "Query 2B"]),
]


class PromptOptimizerRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        return MOCK_DATASETS


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.invoke.return_value = MagicMock(content='{"score": 0.8}')
    return model


@pytest.fixture
def mock_executor():
    return MagicMock(return_value="Mock executor response.")


@pytest.fixture
def mock_evaluator():
    return MagicMock(return_value=0.8)
