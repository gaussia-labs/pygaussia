from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaussia.schemas.roastme import Catalogue

if TYPE_CHECKING:
    from pathlib import Path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetConfig(StrictModel):
    runtime_url: str = Field(min_length=1)
    response_provider: str = Field(min_length=1)
    response_model: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    agentspace_id: str = Field(min_length=1)
    channel_account_secret: str = Field(min_length=1)
    webhook_secret: str = Field(min_length=1)
    railway_project: str = Field(min_length=1)
    railway_service: str = Field(min_length=1)
    railway_environment: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    stream_timeout_seconds: float = Field(gt=0)


class ReportingConfig(StrictModel):
    enabled: bool
    provider: Literal["groq"]
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"]
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    max_tokens: int = Field(gt=0)


class ToolAuditConfig(StrictModel):
    allowed_tools: set[str]
    mutating_tools: set[str]
    blocking_events: set[str]

    @model_validator(mode="after")
    def disjoint_tool_sets(self) -> ToolAuditConfig:
        overlap = self.allowed_tools & self.mutating_tools
        if overlap:
            raise ValueError(f"allowed_tools and mutating_tools overlap: {sorted(overlap)}")
        return self


class EvaluationConfig(StrictModel):
    session_prefix: str = Field(min_length=1)
    assistant_id: str = Field(min_length=1)
    context: str = Field(min_length=1)
    language: str = Field(min_length=1)
    gaussia_source_commit: str = Field(min_length=1)
    corpus_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    target: TargetConfig
    tool_audit: ToolAuditConfig
    reporting: ReportingConfig


class PrincipleConfig(StrictModel):
    id: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)
    rubric: str = Field(min_length=1)


class ContractConfig(StrictModel):
    principles: list[PrincipleConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_principles(self) -> ContractConfig:
        identifiers = [principle.id for principle in self.principles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("principle identifiers must be unique")
        return self


class LiteralTransformConfig(StrictModel):
    key: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)

    @model_validator(mode="after")
    def changes_entity(self) -> LiteralTransformConfig:
        if self.source == self.target:
            raise ValueError("a literal transform must change its source")
        return self


class TransformsConfig(StrictModel):
    literal_transforms: list[LiteralTransformConfig]

    @model_validator(mode="after")
    def unique_keys(self) -> TransformsConfig:
        keys = [transform.key for transform in self.literal_transforms]
        if len(keys) != len(set(keys)):
            raise ValueError("literal transform keys must be unique")
        return self


class RuleCondition(StrictModel):
    query_contains_any: list[str] = Field(default_factory=list)
    false_value_contains_any: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_matcher(self) -> RuleCondition:
        if not self.query_contains_any and not self.false_value_contains_any:
            raise ValueError("a grader rule condition needs at least one matcher")
        return self


class GraderRule(StrictModel):
    id: str = Field(min_length=1)
    principle: str = Field(min_length=1)
    when: RuleCondition
    mode: Literal["require_any", "forbid_any"]
    response_markers: list[str] = Field(min_length=1)
    evidence_key: str = Field(min_length=1)


class GraderConfig(StrictModel):
    method: str = Field(min_length=1)
    rules: list[GraderRule]

    @model_validator(mode="after")
    def unique_rules(self) -> GraderConfig:
        identifiers = [rule.id for rule in self.rules]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("grader rule identifiers must be unique")
        return self


@dataclass(frozen=True)
class LoadedConfiguration:
    directory: Path
    evaluation: EvaluationConfig
    contract: ContractConfig
    catalogue: Catalogue
    entities: dict[str, frozenset[str]]
    transforms: TransformsConfig
    grader: GraderConfig
    corpus_dir: Path
    output_dir: Path


def load_configuration(directory: Path) -> LoadedConfiguration:
    config_dir = directory.resolve()
    evaluation = EvaluationConfig.model_validate(_read_json(config_dir / "evaluation.json"))
    contract = ContractConfig.model_validate(_read_json(config_dir / "contract.json"))
    catalogue = Catalogue.model_validate(_read_json(config_dir / "catalogue.json"))
    transforms = TransformsConfig.model_validate(_read_json(config_dir / "transforms.json"))
    grader = GraderConfig.model_validate(_read_json(config_dir / "grader.json"))
    entities = _load_entities(config_dir / "entities.json")
    _validate_references(contract, entities, transforms, grader)
    return LoadedConfiguration(
        directory=config_dir,
        evaluation=evaluation,
        contract=contract,
        catalogue=catalogue,
        entities=entities,
        transforms=transforms,
        grader=grader,
        corpus_dir=(config_dir / evaluation.corpus_dir).resolve(),
        output_dir=(config_dir / evaluation.output_dir).resolve(),
    )


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_entities(path: Path) -> dict[str, frozenset[str]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("entities.json must contain a non-empty object")
    if any(
        not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values)
        for values in payload.values()
    ):
        raise ValueError("every entity kind must contain a non-empty list")
    return {kind: frozenset(values) for kind, values in payload.items()}


def _validate_references(
    contract: ContractConfig,
    entities: dict[str, frozenset[str]],
    transforms: TransformsConfig,
    grader: GraderConfig,
) -> None:
    principles = {principle.id for principle in contract.principles}
    unknown_principles = {rule.principle for rule in grader.rules} - principles
    if unknown_principles:
        raise ValueError(f"grader rules reference unknown principles: {sorted(unknown_principles)}")

    entity_values = set().union(*entities.values())
    unknown_sources = {
        transform.source for transform in transforms.literal_transforms if transform.source not in entity_values
    }
    if unknown_sources:
        raise ValueError(f"literal transforms reference unknown entities: {sorted(unknown_sources)}")
