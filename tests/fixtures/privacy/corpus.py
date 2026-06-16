"""Helpers to build in-memory privacy corpora and retrievers for tests."""

from gaussia.core.retriever import Retriever
from gaussia.schemas.common import Dataset
from gaussia.schemas.privacy import PrivacyBatch, Span


def batch(qa_id: str, query: str, spans: list[Span]) -> PrivacyBatch:
    return PrivacyBatch(
        qa_id=qa_id,
        query=query,
        assistant="",
        ground_truth_assistant="",
        spans=spans,
    )


def dataset(session_id: str, conversation: list[PrivacyBatch], assistant_id: str = "assistant") -> Dataset:
    return Dataset(
        session_id=session_id,
        assistant_id=assistant_id,
        context="",
        conversation=conversation,
    )


def make_retriever(datasets: list[Dataset]) -> type[Retriever]:
    class _PrivacyRetriever(Retriever):
        def load_dataset(self) -> list[Dataset]:
            return datasets

    return _PrivacyRetriever
