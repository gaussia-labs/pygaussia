"""Common retrievers for Gaussia examples."""

import json
from collections.abc import Iterator
from pathlib import Path

from gaussia import Retriever
from gaussia.schemas import Dataset, IterationLevel, SessionMetadata, StreamedBatch


class LocalRetriever(Retriever):
    """Retriever that loads all datasets into memory at once (full_dataset mode)."""

    def __init__(self, dataset_path: str | None = None, **kwargs):
        """Initialize the LocalRetriever.

        Args:
            dataset_path: Path to the dataset JSON file. If not provided,
                defaults to examples/data/dataset.json.
            **kwargs: Additional arguments passed to base Retriever.
        """
        super().__init__(**kwargs)
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            examples_dir = Path(__file__).parent.parent
            self.dataset_path = examples_dir / "data" / "dataset.json"

    def load_dataset(self) -> list[Dataset]:
        """Load datasets from the JSON file.

        Returns:
            List of Dataset objects parsed from the JSON file.
        """
        datasets = []
        with open(self.dataset_path) as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset))
        return datasets


class StreamingSessionRetriever(Retriever):
    """Retriever that yields Dataset sessions one at a time (stream_sessions mode).

    Useful for large datasets where loading everything into memory is not feasible.
    Each session is yielded individually as a full Dataset.
    """

    def __init__(self, dataset_path: str | None = None, **kwargs):
        """Initialize the StreamingSessionRetriever.

        Args:
            dataset_path: Path to the dataset JSON file. If not provided,
                defaults to examples/data/dataset.json.
            **kwargs: Additional arguments passed to base Retriever.
        """
        super().__init__(**kwargs)
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            examples_dir = Path(__file__).parent.parent
            self.dataset_path = examples_dir / "data" / "dataset.json"

    @property
    def iteration_level(self) -> IterationLevel:
        return IterationLevel.STREAM_SESSIONS

    def load_dataset(self) -> Iterator[Dataset]:
        """Yield Dataset sessions one at a time.

        Note: json.load() reads the full file into memory before iterating.
        For truly large files, replace with a streaming parser such as ijson:
            for item in ijson.items(f, 'item'): yield Dataset.model_validate(item)

        Yields:
            Dataset objects parsed lazily from the JSON file.
        """
        with open(self.dataset_path) as infile:
            for dataset in json.load(infile):
                yield Dataset.model_validate(dataset)


class StreamingBatchRetriever(Retriever):
    """Retriever that yields individual QA pairs across all sessions (stream_batches mode).

    Useful when processing each QA interaction independently, regardless of session.
    Each yielded item includes the session metadata alongside the individual batch.
    """

    def __init__(self, dataset_path: str | None = None, **kwargs):
        """Initialize the StreamingBatchRetriever.

        Args:
            dataset_path: Path to the dataset JSON file. If not provided,
                defaults to examples/data/dataset.json.
            **kwargs: Additional arguments passed to base Retriever.
        """
        super().__init__(**kwargs)
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            examples_dir = Path(__file__).parent.parent
            self.dataset_path = examples_dir / "data" / "dataset.json"

    @property
    def iteration_level(self) -> IterationLevel:
        return IterationLevel.STREAM_BATCHES

    def load_dataset(self) -> Iterator[StreamedBatch]:
        """Yield individual QA pairs with their session metadata.

        Note: json.load() reads the full file into memory before iterating.
        For truly large files, replace with a streaming parser such as ijson:
            for item in ijson.items(f, 'item'): yield Dataset.model_validate(item)

        Yields:
            StreamedBatch objects, one per QA interaction.
        """
        with open(self.dataset_path) as infile:
            for raw in json.load(infile):
                dataset = Dataset.model_validate(raw)
                metadata = SessionMetadata(
                    session_id=dataset.session_id,
                    assistant_id=dataset.assistant_id,
                    language=dataset.language,
                    context=dataset.context,
                )
                for batch in dataset.conversation:
                    yield StreamedBatch(metadata=metadata, batch=batch)
