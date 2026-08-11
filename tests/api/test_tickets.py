"""HTTP contract of the synchronous triage path.

Most tests replace ``get_triage_service`` with a hand-written fake, so what is
under test is the route: status codes, body shape, validation and the 404 path.
The last test deliberately keeps the real wiring — with no ``GEMINI_API_KEY``
the app degrades to the rule classifier, which is offline — because "raw PII
never reaches the response" is only worth asserting end to end.
"""

from collections.abc import Iterator
from datetime import timedelta
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dependencies import get_triage_service
from src.models.domain import AuditRecord, Channel, Ticket, TicketStatus, TriageDecision
from src.utils import utc_now


# T-1004 from data/tickets/historical_tickets.json, raw and unmasked.
T_1004_RAW = (
    "Помогите восстановить аккаунт. "
    "Мой логин test@test.com, паспорт серия 1234 номер 567890."
)

TICKET_RESPONSE_FIELDS = {
    "id",
    "channel",
    "text",
    "external_id",
    "status",
    "triage",
    "draft",
    "created_at",
    "updated_at",
}


def make_ticket(
    *,
    text: str = "Как привязать карту?",
    channel: Channel = "chat",
    external_id: Optional[str] = None,
    status: TicketStatus = "queued",
    decision: Optional[TriageDecision] = None,
) -> Ticket:
    return Ticket(
        id=uuid4(),
        channel=channel,
        text=text,
        external_id=external_id,
        status=status,
        decision=decision
        if decision is not None
        else TriageDecision(
            category="billing/faq",
            category_confidence=0.91,
            risk="low",
            route="auto_answer",
            auto_send_allowed=True,
            classifier="llm",
            rule_hits=("billing_faq",),
            pii_types=(),
            injection_flagged=False,
            reason="вопрос по привязке карты",
            latency_ms=12.5,
        ),
    )


class FakeTriageService:
    """Stands in for ``TriageService`` and records every call the route makes."""

    def __init__(self) -> None:
        self.ticket: Optional[Ticket] = None
        self.created: bool = True
        self.records: list[AuditRecord] = []
        self.triage_calls: list[dict[str, Any]] = []

    async def triage(
        self, *, channel: str, text: str, external_id: Optional[str] = None
    ) -> tuple[Ticket, bool]:
        self.triage_calls.append(
            {"channel": channel, "text": text, "external_id": external_id}
        )
        assert self.ticket is not None, "the test must set fake_service.ticket first"
        return self.ticket, self.created

    async def get(self, ticket_id: UUID) -> Optional[Ticket]:
        if self.ticket is not None and self.ticket.id == ticket_id:
            return self.ticket
        return None

    async def audit_trail(self, ticket_id: UUID) -> list[AuditRecord]:
        return [record for record in self.records if record.ticket_id == ticket_id]


@pytest.fixture
def fake_service(app: FastAPI) -> Iterator[FakeTriageService]:
    """Install the fake and always take the override back down again."""
    service = FakeTriageService()
    app.dependency_overrides[get_triage_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_triage_service, None)


# --- POST /tickets -------------------------------------------------------


def test_create_returns_201_with_the_triage_decision(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    fake_service.ticket = make_ticket(text="Как привязать карту?")

    response = client.post(
        "/api/v1/tickets", json={"channel": "chat", "text": "Как привязать карту?"}
    )

    assert response.status_code == 201
    body = response.json()
    assert set(body) == TICKET_RESPONSE_FIELDS
    assert body["id"] == str(fake_service.ticket.id)
    assert body["channel"] == "chat"
    assert body["text"] == "Как привязать карту?"
    assert body["external_id"] is None
    assert body["status"] == "queued"
    assert body["draft"] is None
    assert body["triage"] == {
        "category": "billing/faq",
        "category_confidence": pytest.approx(0.91),
        "risk": "low",
        "route": "auto_answer",
        "auto_send_allowed": True,
        "classifier": "llm",
        "rule_hits": ["billing_faq"],
        "pii_types": [],
        "injection_flagged": False,
        "reason": "вопрос по привязке карты",
        "latency_ms": pytest.approx(12.5),
    }


def test_create_forwards_the_request_fields_to_the_service(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    fake_service.ticket = make_ticket(channel="email", external_id="msg-42")

    client.post(
        "/api/v1/tickets",
        json={"channel": "email", "text": "Не могу войти", "external_id": "msg-42"},
    )

    assert fake_service.triage_calls == [
        {"channel": "email", "text": "Не могу войти", "external_id": "msg-42"}
    ]


def test_repeated_external_id_returns_200_instead_of_201(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    fake_service.ticket = make_ticket(external_id="msg-42")
    fake_service.created = False

    response = client.post(
        "/api/v1/tickets",
        json={"channel": "chat", "text": "Как привязать карту?", "external_id": "msg-42"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(fake_service.ticket.id)
    assert response.json()["external_id"] == "msg-42"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"channel": "chat"}, id="missing-text"),
        pytest.param({"channel": "chat", "text": ""}, id="empty-text"),
        pytest.param({"text": "Как привязать карту?"}, id="missing-channel"),
        pytest.param(
            {"channel": "telegram", "text": "Как привязать карту?"}, id="unknown-channel"
        ),
        pytest.param(
            {"channel": "chat", "text": "Как привязать карту?", "txet": "typo"},
            id="unknown-extra-field",
        ),
        pytest.param(
            {"channel": "chat", "text": "Как привязать карту?", "external_id": ""},
            id="empty-external-id",
        ),
    ],
)
def test_invalid_payload_is_rejected_before_the_service_is_called(
    fake_service: FakeTriageService, client: TestClient, payload: dict[str, Any]
) -> None:
    response = client.post("/api/v1/tickets", json=payload)

    assert response.status_code == 422
    assert fake_service.triage_calls == []


def test_validation_error_names_the_offending_field(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    response = client.post("/api/v1/tickets", json={"channel": "chat", "text": ""})

    assert response.status_code == 422
    locations = [tuple(error["loc"]) for error in response.json()["detail"]]
    assert ("body", "text") in locations


# --- GET /tickets/{id} ---------------------------------------------------


def test_get_returns_the_stored_ticket(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    fake_service.ticket = make_ticket(text="Сайт не работает", status="draft_ready")

    response = client.get(f"/api/v1/tickets/{fake_service.ticket.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(fake_service.ticket.id)
    assert body["text"] == "Сайт не работает"
    assert body["status"] == "draft_ready"


def test_get_unknown_ticket_returns_404(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    unknown = uuid4()

    response = client.get(f"/api/v1/tickets/{unknown}")

    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "detail": f"Ticket {unknown} does not exist",
    }


def test_get_ticket_with_a_malformed_uuid_is_422(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    response = client.get("/api/v1/tickets/not-a-uuid")

    assert response.status_code == 422


# --- GET /tickets/{id}/audit ---------------------------------------------


def test_audit_returns_the_decision_trail_oldest_first(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    ticket = make_ticket()
    fake_service.ticket = ticket
    triaged_at = utc_now()
    fake_service.records = [
        AuditRecord(
            ticket_id=ticket.id, event="triaged", details={"route": "auto_answer"}, at=triaged_at
        ),
        AuditRecord(
            ticket_id=ticket.id,
            event="queued",
            details={"pending": 1},
            at=triaged_at + timedelta(milliseconds=2),
        ),
    ]

    response = client.get(f"/api/v1/tickets/{ticket.id}/audit")

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == str(ticket.id)
    assert [record["event"] for record in body["records"]] == ["triaged", "queued"]
    assert body["records"][0]["details"] == {"route": "auto_answer"}


def test_audit_for_unknown_ticket_returns_404(
    fake_service: FakeTriageService, client: TestClient
) -> None:
    unknown = uuid4()

    response = client.get(f"/api/v1/tickets/{unknown}/audit")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


# --- end to end, with the real service ------------------------------------


def test_response_body_never_contains_the_raw_pii_that_was_posted(
    client: TestClient,
) -> None:
    """Real wiring: the LLM is unconfigured, so this runs entirely offline."""
    response = client.post("/api/v1/tickets", json={"channel": "chat", "text": T_1004_RAW})

    assert response.status_code == 201
    assert "test@test.com" not in response.text
    assert "1234 номер 567890" not in response.text
    body = response.json()
    assert body["text"] == (
        "Помогите восстановить аккаунт. Мой логин [EMAIL], паспорт [PASSPORT]."
    )
    assert body["triage"]["pii_types"] == ["EMAIL", "PASSPORT"]
    assert body["triage"]["risk"] == "medium"
    assert body["triage"]["auto_send_allowed"] is False
