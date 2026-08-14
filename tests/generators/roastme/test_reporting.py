"""Model-assisted findings remain traceable to the immutable Profiler evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gaussia.generators.roastme.profiler import Profiler
from gaussia.generators.roastme.reporting import FindingsReporter, render_findings_markdown
from gaussia.schemas.roastme import (
    FindingKind,
    FindingsInterpretation,
    GraderAssessment,
    InterpretedFinding,
)
from tests.fixtures.roastme import expected as fx

SESSION_ID = "reporting-run"
ASSISTANT_ID = "assistant-under-test"
MODEL_ID = "openai/gpt-oss-120b"


class _StructuredModel:
    def __init__(self, interpretation: FindingsInterpretation) -> None:
        self.interpretation = interpretation
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.messages = messages
        return {
            "raw": SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50, "ignored": "value"}),
            "parsed": self.interpretation,
            "parsing_error": None,
        }


class _Model:
    def __init__(self, interpretation: FindingsInterpretation) -> None:
        self.structured = _StructuredModel(interpretation)
        self.schema = None
        self.kwargs: dict[str, Any] = {}

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StructuredModel:
        self.schema = schema
        self.kwargs = kwargs
        return self.structured


def _interpretation(probe_id: str = "pb-1") -> FindingsInterpretation:
    return FindingsInterpretation(
        executive_summary="La corrida contiene un hallazgo trazable.",
        findings=[
            InterpretedFinding(
                probe_id=probe_id,
                kind=FindingKind.BEHAVIORAL,
                grader_assessment=GraderAssessment.SUPPORTS,
                principle=fx.PRINCIPLE_A,
                title="La respuesta acepta una premisa falsa",
                evidence='"response one"',
                analysis="La respuesta contradice la rúbrica configurada.",
                recommendation="Aclarar explícitamente la premisa antes de continuar.",
                confidence="high",
            )
        ],
        limitations=["La interpretación usa el mismo modelo que el target."],
    )


def _reporter(model: _Model) -> FindingsReporter:
    return FindingsReporter(
        model,
        provider="groq",
        model_id=MODEL_ID,
        language="spanish",
        structured_output_kwargs={"method": "json_schema"},
    )


def _run(reporter: FindingsReporter):
    grader = fx.stub_grader()
    contract = fx.contract(grader)
    probes = fx.probes()
    result = Profiler(contract, fx.recorded_target()).profile(probes)
    return reporter.interpret(
        session_id=SESSION_ID,
        assistant_id=ASSISTANT_ID,
        target_model=MODEL_ID,
        contract=contract,
        probes=probes,
        result=result,
        documents=fx.documents(),
        grader_context={"method": "fixture-rule"},
        transport_audit=[],
    )


def test_reporter_requests_strict_structured_output_and_records_provenance():
    model = _Model(_interpretation())
    report = _run(_reporter(model))

    assert model.schema is FindingsInterpretation
    assert model.kwargs == {"include_raw": True, "method": "json_schema"}
    assert report.session_id == SESSION_ID
    assert report.model == MODEL_ID
    assert report.same_model_as_target is True
    assert report.reported_overall_rate == fx.EXPECTED_OVERALL_RATE
    assert report.usage == {"input_tokens": 100, "output_tokens": 50}


def test_model_receives_grades_corpus_and_probe_identifiers_as_data():
    model = _Model(_interpretation())
    _run(_reporter(model))

    payload = json.loads(model.structured.messages[1].content)
    assert payload["run"]["overall_rate"] == fx.EXPECTED_OVERALL_RATE
    assert {probe["id"] for probe in payload["candidate_probes"]} == {
        "pb-1",
        "pb-2",
        "pb-4",
        "pb-5",
        "pb-6",
        "pb-control",
        "pb-failed",
    }
    assert payload["corpus"] == [document.model_dump(mode="json") for document in fx.documents()]
    assert payload["candidate_probes"][0]["charged_grades"]
    assert "pb-3" in payload["passing_probe_ids"]
    assert "Plausibility, common practice or general legal knowledge is not evidence" in (
        model.structured.messages[0].content
    )
    assert "merely not invoking a mutating tool does not satisfy it" in (model.structured.messages[0].content)
    assert "A taxonomy_mismatch does not make the response valid" in (model.structured.messages[0].content)


def test_reporter_rejects_a_model_finding_with_an_unknown_probe():
    reporter = _reporter(_Model(_interpretation("invented-probe")))

    with pytest.raises(ValueError, match="unknown probes"):
        _run(reporter)


def test_a_confirmed_finding_on_a_control_is_labelled_as_a_control_finding():
    control_id = next(probe.id for probe in fx.probes() if probe.plugin is None)
    report = _run(_reporter(_Model(_interpretation(control_id))))

    assert report.interpretation.findings[0].kind is FindingKind.CONTROL


def test_markdown_discloses_model_assistance_and_keeps_the_reported_score():
    report = _run(_reporter(_Model(_interpretation())))

    markdown = render_findings_markdown(report)

    assert "No modifica el score" in markdown
    assert "mismo modelo que el agente evaluado" in markdown
    assert "`0.5000`" in markdown
    assert "`pb-1`" in markdown
    assert "`behavioral_finding`" in markdown
