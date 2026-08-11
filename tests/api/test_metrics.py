"""``GET /api/v1/metrics`` — the audit-log projection served over HTTP.

The aggregation itself is pinned in ``tests/services/test_metrics_service.py``.
What these tests add is the wiring: the endpoint answers with a body that
matches ``MetricsResponse``, and the numbers move when real tickets go through
the real application.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.schemas import TicketClassification
from src.config import get_settings
from src.dependencies import get_topic_classifier
from src.models.responses import MetricsResponse


METRICS_URL = "/api/v1/metrics"
TICKETS_URL = "/api/v1/tickets"

#: No rule fires on this one and it holds no personal data, so the routing is
#: decided by the classifier alone.
NEUTRAL_TEXT = "Здравствуйте, подскажите пожалуйста по тарифам."


class StubClassifier:
    """A deterministic ``TopicClassifier``: no provider, no network, no latency."""

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


def post_ticket(client: TestClient, text: str = NEUTRAL_TEXT) -> dict:
    response = client.post(TICKETS_URL, json={"channel": "chat", "text": text})
    assert response.status_code == 201, response.text
    return response.json()


def test_metrics_return_a_body_matching_the_response_model(client: TestClient) -> None:
    response = client.get(METRICS_URL)

    body = response.json()
    assert response.status_code == 200
    assert MetricsResponse.model_validate(body).model_dump() == body


def test_metrics_are_all_zero_before_any_ticket_arrives(client: TestClient) -> None:
    response = client.get(METRICS_URL)

    result = MetricsResponse.model_validate(response.json())
    assert result.tickets_total == 0
    assert (result.by_category, result.by_risk, result.by_route) == ({}, {}, {})
    assert result.triage_latency_ms.count == 0
    assert result.over_budget_ratio == 0.0


def test_metrics_report_the_configured_hot_path_budget(client: TestClient) -> None:
    response = client.get(METRICS_URL)

    result = MetricsResponse.model_validate(response.json())
    assert result.hot_path_budget_ms == get_settings().hot_path_budget_ms


def test_metrics_count_the_tickets_posted_through_the_app(
    stub_classifier: StubClassifier,
    client: TestClient,
) -> None:
    post_ticket(client)
    post_ticket(client)

    result = MetricsResponse.model_validate(client.get(METRICS_URL).json())

    assert result.tickets_total == 2
    assert result.by_category == {"billing/faq": 2}
    assert result.by_risk == {"low": 2}
    assert result.by_route == {"auto_answer": 2}
    assert (result.llm_classifications, result.rule_fallbacks) == (2, 0)
    assert result.auto_send_allowed == 2


def test_metrics_time_every_triaged_ticket(
    stub_classifier: StubClassifier,
    client: TestClient,
) -> None:
    post_ticket(client)
    post_ticket(client)

    latency = MetricsResponse.model_validate(client.get(METRICS_URL).json()).triage_latency_ms

    assert latency.count == 2
    assert latency.max_ms >= latency.p95_ms >= latency.p50_ms > 0.0


def test_metrics_report_the_rule_fallback_when_the_llm_is_unconfigured(
    client: TestClient,
) -> None:
    """No override here: the null LLM client is what an unkeyed deployment gets.

    The whole degraded path is therefore exercised end to end — the classifier
    raises, the rules answer, and the projection reports one failure and one
    fallback rather than losing the ticket.
    """
    post_ticket(client)

    result = MetricsResponse.model_validate(client.get(METRICS_URL).json())

    assert result.tickets_total == 1
    assert (result.rule_fallbacks, result.llm_classifications) == (1, 0)
    assert (result.llm_calls, result.llm_failures) == (0, 1)
    assert result.low_confidence == 1
    assert result.auto_send_allowed == 0
    assert result.by_route == {"operator_queue": 1}


def test_metrics_ignore_a_deduplicated_redelivery(
    stub_classifier: StubClassifier,
    client: TestClient,
) -> None:
    payload = {"channel": "chat", "text": NEUTRAL_TEXT, "external_id": "msg-42"}
    first = client.post(TICKETS_URL, json=payload)
    second = client.post(TICKETS_URL, json=payload)

    result = MetricsResponse.model_validate(client.get(METRICS_URL).json())

    assert (first.status_code, second.status_code) == (201, 200)
    assert result.tickets_total == 1
    assert stub_classifier.calls == [NEUTRAL_TEXT]
