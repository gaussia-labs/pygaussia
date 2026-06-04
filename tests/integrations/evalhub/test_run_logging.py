from __future__ import annotations

from gaussia.integrations.evalhub.run_logging import build_run_logger_from_env


class FakeRunLogger:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_build_run_logger_from_env_returns_none_without_config(monkeypatch) -> None:
    monkeypatch.delenv("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY", raising=False)
    monkeypatch.delenv("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY_KWARGS_JSON", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert build_run_logger_from_env() is None


def test_build_run_logger_from_env_uses_custom_factory(monkeypatch) -> None:
    monkeypatch.setenv("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY", f"{__name__}.FakeRunLogger")
    monkeypatch.setenv("GAUSSIA_EVALHUB_RUN_LOGGER_FACTORY_KWARGS_JSON", '{"sink": "test"}')
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")

    logger = build_run_logger_from_env()

    assert isinstance(logger, FakeRunLogger)
    assert logger.kwargs == {"sink": "test"}
