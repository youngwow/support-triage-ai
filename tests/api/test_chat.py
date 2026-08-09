"""POST /api/v1/chat with the assistant service stubbed at the seam.

Every test overrides ``get_assistant_service`` — FastAPI resolves endpoint
dependencies even for requests that fail body validation, and the real
provider would build the Gemini client and the FAISS knowledge base.
"""

import pytest

from src.dependencies import get_assistant_service
from src.models.domain import AssistantReply


class StubAssistant:
    def __init__(self, reply: AssistantReply) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def handle_message(self, *, chat_id, user_id, text) -> AssistantReply:
        self.calls.append({"chat_id": chat_id, "user_id": user_id, "text": text})
        return self.reply


@pytest.fixture
def stub_assistant(app):
    stub = StubAssistant(
        AssistantReply(
            text="Передаю оператору.",
            route="escalate",
            escalated=True,
            escalation_reason="инцидент безопасности",
            sources=["security_policy.md"],
        )
    )
    app.dependency_overrides[get_assistant_service] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


def test_returns_the_reply_as_wire_format(stub_assistant, client):
    response = client.post(
        "/api/v1/chat",
        json={"chat_id": "chat-1", "user_id": "employee-7", "text": "Потерял ноутбук"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Передаю оператору.",
        "route": "escalate",
        "escalated": True,
        "escalation_reason": "инцидент безопасности",
        "sources": ["security_policy.md"],
    }
    assert stub_assistant.calls == [
        {"chat_id": "chat-1", "user_id": "employee-7", "text": "Потерял ноутбук"}
    ]


def test_omitted_user_id_reaches_the_service_as_none(stub_assistant, client):
    response = client.post("/api/v1/chat", json={"chat_id": "chat-1", "text": "Вопрос"})

    assert response.status_code == 200
    assert stub_assistant.calls == [{"chat_id": "chat-1", "user_id": None, "text": "Вопрос"}]


@pytest.mark.parametrize(
    ("payload", "field", "error_type"),
    [
        ({"chat_id": "chat-1"}, "text", "missing"),
        ({"chat_id": "chat-1", "text": ""}, "text", "string_too_short"),
        (
            {"chat_id": "chat-1", "text": "Вопрос", "role": "admin"},
            "role",
            "extra_forbidden",
        ),
    ],
    ids=["missing-text", "empty-text", "unknown-extra-field"],
)
def test_invalid_bodies_are_rejected_with_422(
    stub_assistant, client, payload, field, error_type
):
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        error["type"] == error_type and error["loc"][-1] == field for error in errors
    ), errors
    # the service was never reached
    assert stub_assistant.calls == []
