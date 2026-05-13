from __future__ import annotations

import warnings


def silence_evalhub_dependency_warnings() -> None:
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
    except ImportError:
        return

    warnings.filterwarnings(
        "ignore",
        category=PydanticDeprecatedSince20,
        module=r"olot\..*",
    )


silence_evalhub_dependency_warnings()
