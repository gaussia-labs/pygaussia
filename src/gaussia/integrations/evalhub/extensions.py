from __future__ import annotations

import importlib
import json
import os
from typing import Any


def load_dotted_object(path: str) -> Any:
    module_name, _, object_name = path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(f"Expected a dotted object path, got: {path}")

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Could not import module for dotted object path: {path}") from exc

    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ValueError(f"Could not find object for dotted path: {path}") from exc


def load_json_object_env(name: str) -> dict[str, Any] | None:
    raw = clean_env(name)
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"{name} must be a JSON object")
    return parsed


def clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
