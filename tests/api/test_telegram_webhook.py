"""POST /api/v1/telegram/webhook with a fake bot and dispatcher.

The bot is a plain object — the route never calls its methods, only threads it
through validation context and ``feed_update``. ``TestClient`` runs background
tasks before returning, so the fake dispatcher's recorded calls are visible
right after each request. Env fixtures are listed before ``client`` so they
apply before ``create_app()`` caches ``get_settings``.
"""

import logging
from typing import Optional

import pytest
from aiogram.types import Update

from src.dependencies import get_bot, get_dispatcher


SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"


class FakeDispatcher:
    def __init__(self, *, error: Optional[Exception] = None) -> None:
        self._error = error
        self.calls: list[tuple[object, Update]] = []

    async def feed_update(self, bot, update) -> None:
        self.calls.append((bot, update))
        if self._error is not None:
            raise self._error


def update_payload(update_id: int, text: str = "hi") -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 1,
            "date": 1754700000,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 7, "is_bot": False, "first_name": "T"},
            "text": text,
        },
    }


@pytest.fixture(autouse=True)
def _no_webhook_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on whatever secret ``.env`` may hold."""
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")


@pytest.fixture
def webhook_secret(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    return "s3cret"


@pytest.fixture
def fake_bot() -> object:
    return object()


@pytest.fixture
def fake_dispatcher() -> FakeDispatcher:
    return FakeDispatcher()


@pytest.fixture
def configured_telegram(app, fake_bot, fake_dispatcher):
    app.dependency_overrides[get_bot] = lambda: fake_bot
    app.dependency_overrides[get_dispatcher] = lambda: fake_dispatcher
    yield
    app.dependency_overrides.clear()


def test_acks_and_feeds_the_dispatcher_exactly_once(
    configured_telegram, client, fake_bot, fake_dispatcher
):
    response = client.post("/api/v1/telegram/webhook", json=update_payload(1001))

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(fake_dispatcher.calls) == 1
    bot, update = fake_dispatcher.calls[0]
    assert bot is fake_bot
    assert isinstance(update, Update)
    assert update.update_id == 1001


def test_duplicate_update_id_is_acked_but_not_fed_again(
    configured_telegram, client, fake_dispatcher
):
    first = client.post("/api/v1/telegram/webhook", json=update_payload(2002))
    second = client.post("/api/v1/telegram/webhook", json=update_payload(2002))

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == {"ok": True}
    assert len(fake_dispatcher.calls) == 1


def test_a_new_update_id_is_fed_again(configured_telegram, client, fake_dispatcher):
    client.post("/api/v1/telegram/webhook", json=update_payload(1))
    client.post("/api/v1/telegram/webhook", json=update_payload(2))

    assert [update.update_id for _, update in fake_dispatcher.calls] == [1, 2]


@pytest.mark.parametrize(
    "headers",
    [{}, {SECRET_HEADER: "wrong"}],
    ids=["missing-header", "wrong-header"],
)
def test_bad_secret_is_rejected_with_403(
    webhook_secret, configured_telegram, client, fake_dispatcher, headers
):
    response = client.post(
        "/api/v1/telegram/webhook", json=update_payload(1), headers=headers
    )

    assert response.status_code == 403
    assert response.json()["code"] == "webhook_forbidden"
    assert fake_dispatcher.calls == []


def test_correct_secret_is_accepted(
    webhook_secret, configured_telegram, client, fake_dispatcher
):
    response = client.post(
        "/api/v1/telegram/webhook",
        json=update_payload(1),
        headers={SECRET_HEADER: webhook_secret},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(fake_dispatcher.calls) == 1


def test_returns_503_when_the_bot_is_not_configured(client):
    """Default test env: empty token, so ``get_bot()`` resolves to ``None``."""
    response = client.post("/api/v1/telegram/webhook", json=update_payload(1))

    assert response.status_code == 503
    assert response.json()["code"] == "telegram_not_configured"


def test_a_crashing_handler_is_logged_and_never_breaks_the_ack(
    app, fake_bot, client, caplog
):
    dispatcher = FakeDispatcher(error=RuntimeError("handler blew up"))
    app.dependency_overrides[get_bot] = lambda: fake_bot
    app.dependency_overrides[get_dispatcher] = lambda: dispatcher
    try:
        with caplog.at_level(logging.ERROR, logger="src.api.routes.telegram"):
            response = client.post("/api/v1/telegram/webhook", json=update_payload(3003))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(dispatcher.calls) == 1
    assert "Failed to process Telegram update 3003" in caplog.text
