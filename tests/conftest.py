"""Fixtures shared by the whole suite."""

from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient

from src.config import get_settings
from src.dependencies import get_item_repository
from src.main import create_app
from src.repositories.repository_interface import AbstractRepository


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Drop the cached settings and repository around every test.

    Both providers are ``lru_cache``d, so without this the in-memory store
    carries rows from one test into the next and tests pass or fail depending
    on the order they run in.
    """
    get_settings.cache_clear()
    get_item_repository.cache_clear()
    yield
    get_settings.cache_clear()
    get_item_repository.cache_clear()


@pytest.fixture
def app() -> FastAPI:
    """A fresh application instance per test."""
    return create_app()


@pytest.fixture
def item_repository() -> AbstractRepository:
    """The same repository instance the app under test will resolve.

    Use it to seed state directly instead of driving the API to set up
    preconditions.
    """
    return get_item_repository()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Synchronous client. Entering the context manager runs the lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Async client for tests that must await application code directly.

    ASGITransport does not run startup/shutdown events — use ``client`` when a
    test depends on anything the lifespan sets up.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
