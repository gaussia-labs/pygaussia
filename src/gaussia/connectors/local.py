"""Local filesystem corpus connector implementation."""

from pathlib import Path

from loguru import logger

from .base import CorpusConnector, RegulatoryDocument


class LocalCorpusConnector(CorpusConnector):
    """
    Load regulatory corpus from local filesystem.

    Reads all .md files from a specified directory.
    """

    def __init__(self, corpus_dir: Path | str):
        """
        Initialize local corpus connector.

        Args:
            corpus_dir: Directory containing markdown regulatory documents.
        """
        self.corpus_dir = Path(corpus_dir)

    def load_documents(self) -> list[RegulatoryDocument]:
        """
        Load all markdown files from the corpus directory.

        Returns:
            List of RegulatoryDocument objects.

        Raises:
            FileNotFoundError: If corpus directory does not exist.
        """
        if not self.corpus_dir.exists():
            msg = f"Corpus directory does not exist: {self.corpus_dir}"
            raise FileNotFoundError(msg)

        documents: list[RegulatoryDocument] = []
        md_files = list(self.corpus_dir.glob("*.md"))

        if not md_files:
            logger.warning(f"No .md files found in '{self.corpus_dir}'")
            return documents

        for filepath in md_files:
            text = filepath.read_text(encoding="utf-8")
            documents.append(
                RegulatoryDocument(
                    text=text,
                    source=filepath.name,
                )
            )

        logger.info(f"Loaded {len(documents)} document(s) from {self.corpus_dir}")
        return documents


__all__ = ["LocalCorpusConnector"]
