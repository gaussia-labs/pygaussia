"""Tests for Gaussia base class streaming and iteration level behavior."""

from collections.abc import Iterator

import pytest

from pygaussia.core.base import Gaussia
from pygaussia.core.retriever import Retriever
from pygaussia.schemas.common import Batch, Dataset, IterationLevel, SessionMetadata, StreamedBatch
from tests.fixtures.mock_data import create_sample_dataset


class RecordingMetric(Gaussia):
    """Concrete Gaussia subclass that records batch calls for assertion."""

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None,
    ) -> None:
        self.metrics.append({"session_id": session_id, "assistant_id": assistant_id, "batch_len": len(batch)})

    def on_process_complete(self) -> None:
        self.metrics.append({"hook": "on_process_complete"})


class FullDatasetRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        return [create_sample_dataset(session_id="s1"), create_sample_dataset(session_id="s2")]


class StreamSessionsRetriever(Retriever):
    @property
    def iteration_level(self) -> IterationLevel:
        return IterationLevel.STREAM_SESSIONS

    def load_dataset(self) -> Iterator[Dataset]:
        yield create_sample_dataset(session_id="s1")
        yield create_sample_dataset(session_id="s2")


class StreamBatchesRetriever(Retriever):
    @property
    def iteration_level(self) -> IterationLevel:
        return IterationLevel.STREAM_BATCHES

    def load_dataset(self) -> Iterator[StreamedBatch]:
        dataset = create_sample_dataset(session_id="s1")
        metadata = SessionMetadata(
            session_id=dataset.session_id,
            assistant_id=dataset.assistant_id,
            language=dataset.language,
            context=dataset.context,
        )
        for batch in dataset.conversation:
            yield StreamedBatch(metadata=metadata, batch=batch)


class IteratorWithoutLevelRetriever(Retriever):
    def load_dataset(self) -> Iterator[Dataset]:
        yield create_sample_dataset()


class TestIterationLevelDetection:
    def test_full_dataset_level_is_default(self):
        metric = RecordingMetric(FullDatasetRetriever)
        assert metric.level == IterationLevel.FULL_DATASET

    def test_stream_sessions_level_detected(self):
        metric = RecordingMetric(StreamSessionsRetriever)
        assert metric.level == IterationLevel.STREAM_SESSIONS

    def test_stream_batches_level_detected(self):
        metric = RecordingMetric(StreamBatchesRetriever)
        assert metric.level == IterationLevel.STREAM_BATCHES

    def test_iterator_without_level_raises(self):
        with pytest.raises(ValueError, match="iteration_level"):
            RecordingMetric(IteratorWithoutLevelRetriever)


class TestFullDatasetMode:
    def test_batch_called_once_per_session(self):
        metrics = RecordingMetric.run(FullDatasetRetriever)
        batch_calls = [m for m in metrics if "session_id" in m]
        assert len(batch_calls) == 2
        assert {m["session_id"] for m in batch_calls} == {"s1", "s2"}

    def test_on_process_complete_called(self):
        metrics = RecordingMetric.run(FullDatasetRetriever)
        assert {"hook": "on_process_complete"} in metrics


class TestStreamSessionsMode:
    def test_batch_called_once_per_session(self):
        metrics = RecordingMetric.run(StreamSessionsRetriever)
        batch_calls = [m for m in metrics if "session_id" in m]
        assert len(batch_calls) == 2
        assert {m["session_id"] for m in batch_calls} == {"s1", "s2"}

    def test_batch_receives_full_conversation(self):
        metrics = RecordingMetric.run(StreamSessionsRetriever)
        batch_calls = [m for m in metrics if "batch_len" in m]
        for call in batch_calls:
            assert call["batch_len"] == 2

    def test_on_process_complete_called(self):
        metrics = RecordingMetric.run(StreamSessionsRetriever)
        assert {"hook": "on_process_complete"} in metrics


class TestStreamBatchesMode:
    def test_batch_called_once_per_qa(self):
        metrics = RecordingMetric.run(StreamBatchesRetriever)
        batch_calls = [m for m in metrics if "batch_len" in m]
        assert len(batch_calls) == 2

    def test_each_batch_contains_single_qa(self):
        metrics = RecordingMetric.run(StreamBatchesRetriever)
        batch_calls = [m for m in metrics if "batch_len" in m]
        for call in batch_calls:
            assert call["batch_len"] == 1

    def test_on_process_complete_called(self):
        metrics = RecordingMetric.run(StreamBatchesRetriever)
        assert {"hook": "on_process_complete"} in metrics
