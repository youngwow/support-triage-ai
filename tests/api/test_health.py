"""Health endpoints through the wired app.

The real ``FaissKnowledgeBase`` is never loaded in tests (warmup is off), so
its ping reports "unavailable" — the ok-readiness cases override the provider
with a ping fake. Env fixtures must be listed *before* ``client`` so the env
is in place when ``create_app()`` caches ``get_settings``.
"""

from typing import Union

import pytest

from src import __version__
from src.config import get_settings
from src.dependencies import get_knowledge_base


class PingProbe:
    """Answers ``ping`` with a preset bool, or raises a preset exception."""

    def __init__(self, result: Union[bool, Exception] = True) -> None:
        self._result = result

    async def ping(self) -> bool:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.fixture
def overrides(app):
    yield app.dependency_overrides
    app.dependency_overrides.clear()


@pytest.fixture
def gemini_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.fixture
def gemini_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")


def test_liveness_is_ok_without_any_io(client):
    response = client.get("/api/v1/health")

    settings = get_settings()
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "checks": {},
    }


def test_readiness_returns_200_when_all_probes_pass(gemini_configured, overrides, client):
    overrides[get_knowledge_base] = lambda: PingProbe(True)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {
        "knowledge_base": "ok",
        "hr_system": "ok",
        "dialog_memory": "ok",
        "gemini_config": "ok",
        "telegram_config": "unconfigured",
    }


def test_unconfigured_telegram_alone_does_not_degrade_readiness(
    gemini_configured, overrides, client
):
    """A missing bot token only disables the webhook — /chat still works."""
    overrides[get_knowledge_base] = lambda: PingProbe(True)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["telegram_config"] == "unconfigured"


def test_readiness_returns_503_when_the_knowledge_base_ping_fails(
    gemini_configured, overrides, client
):
    overrides[get_knowledge_base] = lambda: PingProbe(False)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["knowledge_base"] == "unavailable"


def test_readiness_reports_error_when_a_probe_raises(gemini_configured, overrides, client):
    overrides[get_knowledge_base] = lambda: PingProbe(RuntimeError("index exploded"))

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["knowledge_base"] == "error"


def test_readiness_degrades_when_the_gemini_key_is_missing(
    gemini_unconfigured, overrides, client
):
    overrides[get_knowledge_base] = lambda: PingProbe(True)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["gemini_config"] == "unconfigured"
