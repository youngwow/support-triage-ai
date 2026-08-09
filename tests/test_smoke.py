"""Smoke tests proving the pytest setup itself works.

They cover the sync client, the async client, and fixture isolation. Real
coverage per module belongs in files mirroring ``src/``.
"""

from uuid import UUID

from fastapi.testclient import TestClient
from httpx2 import AsyncClient

from src.models.domain import Item
from src.repositories.repository_interface import AbstractRepository


def test_liveness_reports_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_checks_the_repository(async_client: AsyncClient) -> None:
    """Plain ``async def`` — no decorator, thanks to ``asyncio_mode = "auto"``."""
    response = await async_client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {"repository": "ok"}


async def test_repository_fixture_shares_state_with_the_app(
    async_client: AsyncClient, item_repository: AbstractRepository
) -> None:
    """Seeding through the fixture must be visible over HTTP."""
    await item_repository.add(Item(id=UUID(int=1), name="seeded"))

    response = await async_client.get("/api/v1/items")

    assert response.status_code == 200
    assert [item["name"] for item in response.json()["items"]] == ["seeded"]


def test_each_test_starts_from_an_empty_repository(client: TestClient) -> None:
    """Fails if ``_reset_singletons`` stops clearing the cached repository."""
    assert client.get("/api/v1/items").json()["total"] == 0
