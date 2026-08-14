from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gaussia.generators.roastme.reporting import FindingsReporter

if TYPE_CHECKING:
    from configuration import ReportingConfig


def build_findings_reporter(config: ReportingConfig) -> FindingsReporter | None:
    if not config.enabled:
        return None
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        return None
    if config.provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url,
            temperature=0,
            reasoning_effort=config.reasoning_effort,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
        )
    return FindingsReporter(
        model,
        provider=config.provider,
        model_id=config.model,
        language="spanish",
        structured_output_kwargs={"method": "json_schema"},
    )
