"""HurtLex toxicity dataset loader."""

from importlib.resources import files

import pandas as pd

from gaussia.core.loader import ToxicityLoader
from gaussia.schemas.toxicity import ToxicityDataset


class HurtlexLoader(ToxicityLoader):
    """
    Loads HurtLex multilingual toxicity lexicon.

    HurtLex is a lexicon of offensive, aggressive, and hateful words
    in multiple languages.
    """

    def load(self, language: str) -> list[ToxicityDataset]:
        """
        Load HurtLex toxicity dataset for a specific language.

        Args:
            language: Language code (e.g., "english", "spanish")

        Returns:
            List of ToxicityDataset entries
        """
        # Use importlib.resources instead of deprecated pkg_resources
        toxicity_file = str(files("gaussia").joinpath(f"artifacts/toxicity/hurtlex_{language}.tsv"))
        hurtlex_data = pd.read_csv(
            toxicity_file,
            sep="\t",
            header=0,
        )
        return [ToxicityDataset(word=row["lemma"], category=row["category"]) for _, row in hurtlex_data.iterrows()]
