"""Coverage for the health service and its probes.

The readiness probe is what an orchestrator uses to decide whether to route
traffic here, so the degraded path matters as much as the happy one.
"""

from collections.abc import Sequence
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import __version__
from src.config import Settings, get_settings
from src.dependencies import get_item_repository
from src.models.domain import Item
from src.repositories.repository_interface import AbstractRepository
from src.services.health import HealthService


class StubRepository(AbstractRepository):
    """A repository whose ``ping`` fails the way a real outage would."""

    def __init__(self, *, ping_result: bool = True, ping_error: Exception | None = None) -> None:
        self._ping_result = ping_result
        self._ping_error = ping_error

    async def list(self, *, limit: int = 50, offset: int = 0) -> Sequence[Item]:
        return []

    async def count(self) -> int:
        return 0

    async def get(self, item_id: UUID) -> Item | None:
        return None

    async def add(self, item: Item) -> Item:
        return item

    async def update(self, item_id: UUID, item: Item) -> Item | None:
        return item

    async def delete(self, item_id: UUID) -> bool:
        return False

    async def ping(self) -> bool:
        if self._ping_error is not None:
            raise self._ping_error
        return self._ping_result


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def test_liveness_does_not_touch_the_repository(settings: Settings) -> None:
    """Liveness answers even when the backing store is unreachable."""
    service = HealthService(settings, StubRepository(ping_error=RuntimeError("down")))

    result = service.liveness()

    assert result.status == "ok"
    assert result.checks == {}


def test_liveness_reports_the_build_identity(settings: Settings) -> None:
    result = HealthService(settings, StubRepository()).liveness()

    assert result.app == settings.app_name
    assert result.environment == settings.environment
    assert result.version == __version__


async def test_readiness_is_ok_when_the_repository_answers(settings: Settings) -> None:
    result = await HealthService(settings, StubRepository()).readiness()

    assert result.status == "ok"
    assert result.checks == {"repository": "ok"}


async def test_readiness_is_degraded_when_the_repository_says_no(settings: Settings) -> None:
    service = HealthService(settings, StubRepository(ping_result=False))

    result = await service.readiness()

    assert result.status == "degraded"
    assert result.checks == {"repository": "unavailable"}


async def test_readiness_is_degraded_when_the_ping_raises(settings: Settings) -> None:
    """A raising dependency must degrade the probe, not crash it."""
    service = HealthService(settings, StubRepository(ping_error=RuntimeError("connection lost")))

    result = await service.readiness()

    assert result.status == "degraded"
    assert result.checks == {"repository": "error"}


def test_liveness_endpoint_reports_settings_backed_metadata(client: TestClient) -> None:
    """Guards the ``get_health_service`` wiring: settings *and* repository injected."""
    settings = get_settings()

    body = client.get("/api/v1/health").json()

    assert body["status"] == "ok"
    assert body["app"] == settings.app_name
    assert body["environment"] == settings.environment
    assert body["version"] == __version__


def test_readiness_endpoint_returns_200_when_healthy(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint_returns_503_when_degraded(app: FastAPI, client: TestClient) -> None:
    """Orchestrators pull the instance out of rotation on a non-200 here."""
    app.dependency_overrides[get_item_repository] = lambda: StubRepository(ping_result=False)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"

    app.dependency_overrides.clear()
