"""Fixtures shared by the whole suite."""

from collections.abc import AsyncIterator, Iterator, Sequence

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient

from src.config import get_settings
from src.dependencies import (
    get_agent_graph,
    get_bot,
    get_dialog_memory,
    get_dispatcher,
    get_embedder,
    get_hr_system,
    get_knowledge_base,
    get_llm_client,
    get_update_deduplicator,
)
from src.main import create_app


_CACHED_PROVIDERS = (
    get_settings,
    get_embedder,
    get_knowledge_base,
    get_hr_system,
    get_dialog_memory,
    get_llm_client,
    get_agent_graph,
    get_bot,
    get_dispatcher,
    get_update_deduplicator,
)


class FakeEmbedder:
    """Deterministic bag-of-stems stand-in for the real GPU embedder.

    One axis per Russian stem of the knowledge-base corpus plus a constant
    bias axis (so no vector is ever all-zero), occurrence-counted over the
    lowercased text and L2-normalized — FAISS inner product then behaves as
    cosine similarity, exactly like the real model. Queries need no
    instruction prefix here; the same counting is a good enough proxy.
    """

    _VOCAB = (
        "отпуск",
        "предупре",
        "больничн",
        "командировк",
        "суточн",
        "ноутбук",
        "монитор",
        "фишинг",
        "зарплат",
        "грейд",
        "инцидент",
        "оборудован",
    )
    _BIAS = 0.1

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float(lowered.count(stem)) for stem in self._VOCAB] + [self._BIAS])
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Drop every ``lru_cache``d provider around each test.

    Without this, cached singletons carry state from one test into the next
    and the suite becomes order-dependent. Every new ``lru_cache`` provider
    in ``src/dependencies.py`` or ``src/config.py`` must be added to
    ``_CACHED_PROVIDERS``.
    """
    for provider in _CACHED_PROVIDERS:
        provider.cache_clear()
    yield
    for provider in _CACHED_PROVIDERS:
        provider.cache_clear()


@pytest.fixture(autouse=True)
def _fast_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the app lifespan cheap and offline in tests.

    WARMUP_ON_STARTUP=false stops the lifespan from loading the 3B embedding
    model; blanking the Telegram env vars stops it from constructing a real
    Bot or registering a webhook even though ``.env`` may hold real values.
    """
    monkeypatch.setenv("WARMUP_ON_STARTUP", "false")
    monkeypatch.setenv("TELEGRAM_BOT_API_KEY", "")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")


@pytest.fixture
def app() -> FastAPI:
    """A fresh application instance per test."""
    return create_app()


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
