"""Fixtures shared by the whole suite."""

from collections.abc import AsyncIterator, Iterator, Sequence

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient

from src.config import get_settings
from src.dependencies import (
    get_audit_log,
    get_draft_graph,
    get_draft_queue,
    get_embedder,
    get_knowledge_base,
    get_llm_client,
    get_ticket_repository,
    get_topic_classifier,
)
from src.main import create_app
from src.repositories.audit_log import AbstractAuditLog
from src.repositories.draft_queue import DraftQueue
from src.repositories.ticket_repository import AbstractTicketRepository


#: Every ``lru_cache`` provider in ``src/dependencies.py`` or ``src/config.py``.
#: Adding a cached provider without adding it here leaks state between tests and
#: makes the suite order-dependent — keep this list exhaustive.
_CACHED_PROVIDERS = (
    get_settings,
    get_embedder,
    get_knowledge_base,
    get_topic_classifier,
    get_ticket_repository,
    get_audit_log,
    get_llm_client,
    get_draft_graph,
    get_draft_queue,
)


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Drop every cached singleton around each test."""
    for provider in _CACHED_PROVIDERS:
        provider.cache_clear()
    yield
    for provider in _CACHED_PROVIDERS:
        provider.cache_clear()


@pytest.fixture(autouse=True)
def _fast_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the application's startup free of weights, network and background tasks.

    ``WARMUP_ON_STARTUP=false`` stops the lifespan from loading the 3B embedding
    model. ``DRAFT_WORKER_ENABLED=false`` stops the background worker, so draft
    tests call ``process_next()`` explicitly instead of racing a task — a task
    still pending at teardown plus ``filterwarnings=error`` is a flake waiting
    to happen. Blanking the key makes ``get_llm_client()`` return the null
    client even though a real ``.env`` may hold credentials.
    """
    monkeypatch.setenv("WARMUP_ON_STARTUP", "false")
    monkeypatch.setenv("DRAFT_WORKER_ENABLED", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "")


class FakeEmbedder:
    """Deterministic bag-of-stems embedder: no torch, no weights, no network.

    One axis per stem that actually occurs in ``data/knowledge_base``, plus a
    constant bias axis so no vector is ever all-zero — L2-normalising a zero row
    yields NaN and FAISS then returns nonsense. Counts are L2-normalised, so
    ``IndexFlatIP`` behaves as cosine, exactly like the real model.
    """

    _VOCAB = (
        "карт",
        "оплат",
        "подписк",
        "списан",
        "возврат",
        "шлюз",
        "заказ",
        "восстанов",
        "аккаунт",
        "парол",
        "паспорт",
        "персональн",
        "502",
        "сбой",
        "инцидент",
        "недоступ",
    )
    _BIAS = 0.1

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = [
            [float(text.lower().count(stem)) for stem in self._VOCAB] + [self._BIAS]
            for text in texts
        ]
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def app() -> FastAPI:
    """A fresh application instance per test."""
    return create_app()


@pytest.fixture
def ticket_repository() -> AbstractTicketRepository:
    """The same instance the app under test resolves — seed through it."""
    return get_ticket_repository()


@pytest.fixture
def audit_log() -> AbstractAuditLog:
    """The same instance the app under test resolves."""
    return get_audit_log()


@pytest.fixture
def draft_queue() -> DraftQueue:
    """The same instance the app under test resolves."""
    return get_draft_queue()


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
