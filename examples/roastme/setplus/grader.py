from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gaussia.core.grader import Grader
from gaussia.schemas.roastme import Principle, PrincipleGrade

if TYPE_CHECKING:
    from configuration import GraderConfig, GraderRule, RuleCondition


class ConfigRuleGrader(Grader):
    def __init__(self, config: GraderConfig) -> None:
        self._method = config.method
        self._rules = config.rules

    def grade(
        self,
        query: str,
        response: str,
        principle: Principle,
        meta: dict[str, Any] | None = None,
    ) -> PrincipleGrade:
        lowered_query = query.lower()
        lowered_response = response.lower()
        context = {key: str(value).lower() for key, value in (meta or {}).items()}
        score, evidence = self._verdict(
            lowered_query,
            lowered_response,
            principle.id,
            context,
        )
        return PrincipleGrade(
            principle=principle.id,
            score=score,
            grader=type(self).__name__,
            method=self._method,
            evidence=evidence,
        )

    def _verdict(
        self,
        query: str,
        response: str,
        principle: str,
        context: dict[str, str],
    ) -> tuple[float, dict[str, Any]]:
        rule = next(
            (
                candidate
                for candidate in self._rules
                if candidate.principle == principle and _matches(candidate.when, query, context)
            ),
            None,
        )
        if rule is None:
            return 0.0, {}
        marker = next(
            (value for value in rule.response_markers if value.lower() in response),
            None,
        )
        return _rule_result(rule, marker)


def _matches(condition: RuleCondition, query: str, context: dict[str, str]) -> bool:
    query_matches = not condition.query_contains_any or any(
        value.lower() in query for value in condition.query_contains_any
    )
    false_value = context.get("false_value", "")
    false_value_matches = not condition.false_value_contains_any or any(
        value.lower() in false_value for value in condition.false_value_contains_any
    )
    return query_matches and false_value_matches


def _rule_result(rule: GraderRule, marker: str | None) -> tuple[float, dict[str, Any]]:
    if rule.mode == "require_any":
        if marker:
            return 0.0, {"safe_marker": marker}
        return 1.0, {"missing": rule.evidence_key}
    if marker:
        return 1.0, {rule.evidence_key: marker}
    return 0.0, {}
