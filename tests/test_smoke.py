"""The one suite that proves the pieces fit together.

Everything else tests a seam in isolation. This runs the real application —
lifespan, router, services, repositories, audit log — with exactly one thing
replaced: the topic classifier, so no test needs a Gemini key or a network.
"""

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.schemas import TicketClassification
from src.config import get_settings
from src.dependencies import get_topic_classifier
from src.repositories.draft_queue import DraftQueue


HEALTH_URL = "/api/v1/health"
METRICS_URL = "/api/v1/metrics"
TICKETS_URL = "/api/v1/tickets"

#: No rule fires and there is no personal data, so this ticket's route is
#: decided by the classifier alone — the happy path.
ROUTINE_TEXT = "Здравствуйте, подскажите пожалуйста по тарифам."

EXPECTED_PATHS = {
    "/api/v1/health",
    "/api/v1/health/ready",
    "/api/v1/tickets",
    "/api/v1/tickets/{ticket_id}",
    "/api/v1/tickets/{ticket_id}/audit",
    "/api/v1/metrics",
}


class StubClassifier:
    """A deterministic ``TopicClassifier`` standing in for the Gemini call."""

    def __init__(self, classification: TicketClassification) -> None:
        self.classification = classification
        self.calls: list[str] = []

    async def classify(self, text: str) -> TicketClassification:
        self.calls.append(text)
        return self.classification


@pytest.fixture
def stub_classifier(app: FastAPI) -> Iterator[StubClassifier]:
    stub = StubClassifier(
        TicketClassification(
            category="billing/faq",
            risk="low",
            confidence=0.93,
            reason="стаб",
        )
    )
    app.dependency_overrides[get_topic_classifier] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_topic_classifier, None)


@pytest.fixture
def historical_tickets() -> dict[str, dict]:
    """The case's own dataset, keyed by ticket id."""
    path = get_settings().historical_tickets_path
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["ticket_id"]: record for record in records}


def test_the_application_starts_and_answers_the_liveness_probe(
    client: TestClient,
) -> None:
    response = client.get(HEALTH_URL)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_openapi_schema_lists_every_public_route(client: TestClient) -> None:
    response = client.get("/openapi.json")

    schema = response.json()
    assert response.status_code == 200
    assert EXPECTED_PATHS <= set(schema["paths"])
    assert "201" in schema["paths"][TICKETS_URL]["post"]["responses"]


def test_a_routine_ticket_flows_from_intake_to_audit_trail_to_metrics(
    stub_classifier: StubClassifier,
    client: TestClient,
) -> None:
    created = client.post(TICKETS_URL, json={"channel": "chat", "text": ROUTINE_TEXT})

    assert created.status_code == 201, created.text
    ticket = created.json()
    ticket_id = ticket["id"]
    assert ticket["status"] == "queued"
    assert ticket["draft"] is None
    assert ticket["triage"]["category"] == "billing/faq"
    assert ticket["triage"]["route"] == "auto_answer"
    assert ticket["triage"]["auto_send_allowed"] is True
    assert ticket["triage"]["classifier"] == "llm"
    assert stub_classifier.calls == [ROUTINE_TEXT]

    fetched = client.get(f"{TICKETS_URL}/{ticket_id}")

    assert fetched.status_code == 200
    assert fetched.json() == ticket

    audit = client.get(f"{TICKETS_URL}/{ticket_id}/audit")

    assert audit.status_code == 200
    trail = audit.json()
    assert trail["ticket_id"] == ticket_id
    assert [record["event"] for record in trail["records"]] == ["triaged", "queued"]
    triaged = trail["records"][0]["details"]
    assert triaged["category"] == "billing/faq"
    assert triaged["route"] == "auto_answer"
    assert triaged["auto_send_allowed"] is True

    metrics = client.get(METRICS_URL)

    assert metrics.status_code == 200
    counts = metrics.json()
    assert counts["tickets_total"] == 1
    assert counts["by_category"] == {"billing/faq": 1}
    assert counts["by_route"] == {"auto_answer": 1}
    assert counts["llm_classifications"] == 1
    assert counts["triage_latency_ms"]["count"] == 1


def test_a_risky_ticket_escalates_and_is_never_queued_for_a_draft(
    historical_tickets: dict[str, dict],
    stub_classifier: StubClassifier,
    client: TestClient,
    draft_queue: DraftQueue,
) -> None:
    """T-1003: abuse, a fourfold charge and a threat to go to the police.

    The classifier is told to say "low risk" on purpose. The rules must win —
    risk only ever escalates — and a senior escalation must not spend a
    generation call, so nothing reaches the draft queue.
    """
    text = historical_tickets["T-1003"]["user_query"]
    stub_classifier.classification = TicketClassification(
        category="billing/refund",
        risk="low",
        confidence=0.95,
        reason="стаб",
    )

    created = client.post(TICKETS_URL, json={"channel": "email", "text": text})

    assert created.status_code == 201, created.text
    ticket = created.json()
    triage = ticket["triage"]
    assert triage["risk"] == "high"
    assert triage["route"] == "senior_escalation"
    assert triage["auto_send_allowed"] is False
    assert set(triage["rule_hits"]) == {"legal_threat", "refund_demand", "abuse"}
    assert ticket["status"] == "awaiting_operator"

    audit = client.get(f"{TICKETS_URL}/{ticket['id']}/audit")

    assert [record["event"] for record in audit.json()["records"]] == ["triaged"]
    assert draft_queue.pending() == 0
