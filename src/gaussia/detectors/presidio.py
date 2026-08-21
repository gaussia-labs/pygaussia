"""Microsoft Presidio adapter.

Requires the ``privacy-presidio`` extra; the imports below raise a clear
ImportError at module-import time when it is absent (FR-017).
"""

from typing import Any

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from pydantic import PrivateAttr

from gaussia.core.detector import PIIDetector
from gaussia.schemas.privacy import Span

_PRESIDIO_TO_DOMAIN: dict[str, str] = {
    "PERSON": "person",
    "EMAIL_ADDRESS": "email_address",
    "EMAIL": "email_address",
    "PHONE_NUMBER": "phone_number",
    "IBAN_CODE": "iban_code",
    "CREDIT_CARD": "credit_card",
    "DATE_TIME": "date_time",
    "IP_ADDRESS": "ip_address",
    "US_SSN": "us_ssn",
    "LOCATION": "street_address",
    "NRP": "person",
}


class PresidioDetector(PIIDetector):
    """Wraps ``presidio_analyzer.AnalyzerEngine`` with a blank spaCy pipeline (no network)."""

    language: str = "en"

    _engine: Any = PrivateAttr(default=None)

    def setup(self) -> None:
        self._ensure_engine()

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            nlp = spacy.blank(self.language)
            nlp.add_pipe("sentencizer")
            nlp_engine = SpacyNlpEngine(models=[{"lang_code": self.language, "model_name": f"blank_{self.language}"}])
            nlp_engine.nlp = {self.language: nlp}
            self._engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[self.language])
        return self._engine

    @property
    def supported_classes(self) -> frozenset[str]:
        engine = self._ensure_engine()
        entities = engine.get_supported_entities(language=self.language)
        return frozenset(_PRESIDIO_TO_DOMAIN[e] for e in entities if e in _PRESIDIO_TO_DOMAIN)

    def predict(self, text: str) -> list[Span]:
        engine = self._ensure_engine()
        spans: list[Span] = []
        for result in engine.analyze(text=text, language=self.language):
            label = _PRESIDIO_TO_DOMAIN.get(result.entity_type)
            if label is None:
                continue
            spans.append(
                Span(
                    label=label,
                    start=result.start,
                    end=result.end,
                    text=text[result.start : result.end],
                    score=result.score,
                )
            )
        return spans
