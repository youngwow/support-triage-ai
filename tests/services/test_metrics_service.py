"""``MetricsService`` is a pure projection over the audit log.

There is no counter registry to desynchronise: every number below is recomputed
from the same ``AuditRecord``s that document the decisions. So the tests seed an
``InMemoryAuditLog`` directly and never go through HTTP — the seam under test is
"records in, aggregates out".
"""

from typing import Any
from uuid import uuid4

import pytest

from src.config import Settings
from src.models.domain import AuditRecord
from src.models.responses import MetricsResponse
from src.repositories.audit_log import InMemoryAuditLog
from src.services.metrics_service import MetricsService


BUDGET_MS = 500
MIN_CONFIDENCE = 0.75


@pytest.fixture
def settings() -> Settings:
    """Thresholds pinned in code, so the expected numbers never depend on ``.env``."""
    return Settings(hot_path_budget_ms=BUDGET_MS, min_topic_confidence=MIN_CONFIDENCE)


async def snapshot_of(records: list[AuditRecord], settings: Settings) -> MetricsResponse:
    audit = InMemoryAuditLog()
    for record in records:
        await audit.append(record)
    return await MetricsService(audit=audit, settings=settings).snapshot()


def triaged(
    *,
    latency_ms: float = 10.0,
    category: str = "billing/faq",
    risk: str = "low",
    route: str = "auto_answer",
    classifier: str = "llm",
    auto_send_allowed: bool = False,
    pii_types: list[str] | None = None,
    injection_flagged: bool = False,
    category_confidence: float = 0.9,
) -> AuditRecord:
    """A ``triaged`` record shaped exactly like the one ``TriageService`` writes."""
    return AuditRecord(
        ticket_id=uuid4(),
        event="triaged",
        details={
            "category": category,
            "category_confidence": category_confidence,
            "risk": risk,
            "route": route,
            "auto_send_allowed": auto_send_allowed,
            "classifier": classifier,
            "rule_hits": [],
            "pii_types": pii_types or [],
            "injection_flagged": injection_flagged,
            "latency_ms": latency_ms,
            "reason": "",
        },
    )


def drafted(*, status: str = "draft_ready", degraded: bool = False) -> AuditRecord:
    return AuditRecord(
        ticket_id=uuid4(),
        event="drafted",
        details={"status": status, "degraded": degraded, "escalated": False},
    )


def event(name: str, **details: Any) -> AuditRecord:
    return AuditRecord(ticket_id=uuid4(), event=name, details=details)


# --- the empty log -------------------------------------------------------


async def test_an_empty_log_projects_to_all_zeros(settings: Settings) -> None:
    result = await snapshot_of([], settings)

    assert result.tickets_total == 0
    assert (result.by_category, result.by_risk, result.by_route) == ({}, {}, {})
    assert (result.llm_calls, result.llm_failures, result.rule_fallbacks) == (0, 0, 0)
    assert (result.drafts_ready, result.drafts_degraded, result.queue_rejected) == (0, 0, 0)


async def test_an_empty_log_reports_zero_latency_without_dividing_by_zero(
    settings: Settings,
) -> None:
    result = await snapshot_of([], settings)

    latency = result.triage_latency_ms
    assert (latency.count, latency.p50_ms, latency.p95_ms, latency.max_ms) == (0, 0.0, 0.0, 0.0)
    assert (result.over_budget, result.over_budget_ratio) == (0, 0.0)


# --- latency percentiles -------------------------------------------------


async def test_percentiles_interpolate_linearly_between_samples(
    settings: Settings,
) -> None:
    """p95 over five samples sits at position 3.8, i.e. 80% of the way from 400 to 5000."""
    records = [
        triaged(latency_ms=value) for value in (5000.0, 100.0, 400.0, 200.0, 300.0)
    ]

    latency = (await snapshot_of(records, settings)).triage_latency_ms

    assert latency.count == 5
    assert latency.p50_ms == pytest.approx(300.0)
    assert latency.p95_ms == pytest.approx(4080.0)
    assert latency.max_ms == pytest.approx(5000.0)


async def test_percentiles_of_a_single_sample_are_that_sample(settings: Settings) -> None:
    latency = (await snapshot_of([triaged(latency_ms=137.5)], settings)).triage_latency_ms

    assert latency.count == 1
    assert (latency.p50_ms, latency.p95_ms, latency.max_ms) == (137.5, 137.5, 137.5)


async def test_percentiles_of_two_samples_interpolate_across_the_pair(
    settings: Settings,
) -> None:
    records = [triaged(latency_ms=1000.0), triaged(latency_ms=200.0)]

    latency = (await snapshot_of(records, settings)).triage_latency_ms

    # p50 is the midpoint, p95 is 95% of the way from 200 to 1000.
    assert latency.p50_ms == pytest.approx(600.0)
    assert latency.p95_ms == pytest.approx(960.0)
    assert latency.max_ms == pytest.approx(1000.0)


# --- the hot-path budget -------------------------------------------------


async def test_over_budget_counts_only_latencies_strictly_above_the_budget(
    settings: Settings,
) -> None:
    records = [
        triaged(latency_ms=120.0),
        triaged(latency_ms=float(BUDGET_MS)),  # exactly at budget is not over it
        triaged(latency_ms=500.1),
        triaged(latency_ms=1200.0),
    ]

    result = await snapshot_of(records, settings)

    assert result.hot_path_budget_ms == BUDGET_MS
    assert result.over_budget == 2
    assert result.over_budget_ratio == pytest.approx(0.5)


async def test_over_budget_ratio_is_one_when_every_ticket_misses_the_budget(
    settings: Settings,
) -> None:
    records = [triaged(latency_ms=900.0), triaged(latency_ms=1800.0)]

    result = await snapshot_of(records, settings)

    assert (result.over_budget, result.over_budget_ratio) == (2, 1.0)


# --- distributions -------------------------------------------------------


async def test_tickets_are_bucketed_by_category_risk_and_route(
    settings: Settings,
) -> None:
    records = [
        triaged(category="billing/refund", risk="high", route="senior_escalation"),
        triaged(category="billing/refund", risk="high", route="senior_escalation"),
        triaged(category="technical_issue", risk="low", route="auto_answer"),
        triaged(category="account_recovery", risk="medium", route="operator_queue"),
    ]

    result = await snapshot_of(records, settings)

    assert result.tickets_total == 4
    assert result.by_category == {
        "billing/refund": 2,
        "technical_issue": 1,
        "account_recovery": 1,
    }
    assert result.by_risk == {"high": 2, "low": 1, "medium": 1}
    assert result.by_route == {
        "senior_escalation": 2,
        "auto_answer": 1,
        "operator_queue": 1,
    }


async def test_only_triaged_records_count_as_tickets(settings: Settings) -> None:
    """``queued``/``deduplicated``/``drafted`` share the log but are not tickets."""
    records = [
        triaged(category="billing/faq"),
        event("queued", pending=1),
        event("deduplicated", external_id="msg-1"),
        drafted(),
    ]

    result = await snapshot_of(records, settings)

    assert result.tickets_total == 1
    assert result.by_category == {"billing/faq": 1}


@pytest.mark.parametrize(
    ("field", "detail", "truthy", "falsy"),
    [
        ("auto_send_allowed", "auto_send_allowed", True, False),
        ("pii_detected", "pii_types", ["EMAIL"], []),
        ("injection_flagged", "injection_flagged", True, False),
    ],
    ids=["auto_send_allowed", "pii_detected", "injection_flagged"],
)
async def test_flag_counters_count_the_records_carrying_the_flag(
    field: str,
    detail: str,
    truthy: object,
    falsy: object,
    settings: Settings,
) -> None:
    records = [
        triaged(**{detail: truthy}),
        triaged(**{detail: truthy}),
        triaged(**{detail: falsy}),
    ]

    result = await snapshot_of(records, settings)

    assert getattr(result, field) == 2


async def test_low_confidence_counts_records_below_the_topic_threshold(
    settings: Settings,
) -> None:
    records = [
        triaged(category_confidence=0.2),
        triaged(category_confidence=0.6),
        triaged(category_confidence=MIN_CONFIDENCE),  # the threshold itself is confident
        triaged(category_confidence=0.9),
    ]

    result = await snapshot_of(records, settings)

    assert result.low_confidence == 2


# --- LLM usage -----------------------------------------------------------


async def test_rule_fallbacks_count_tickets_the_model_did_not_classify(
    settings: Settings,
) -> None:
    records = [
        triaged(classifier="rules"),
        triaged(classifier="rules"),
        triaged(classifier="llm"),
    ]

    result = await snapshot_of(records, settings)

    assert (result.rule_fallbacks, result.llm_classifications) == (2, 1)


async def test_llm_calls_sum_classifications_and_non_degraded_drafts(
    settings: Settings,
) -> None:
    records = [
        triaged(classifier="llm"),
        triaged(classifier="llm"),
        triaged(classifier="rules"),  # no call was made for this one
        drafted(degraded=False),
        drafted(degraded=False),
        drafted(degraded=True),  # the fallback path: no generation call either
    ]

    result = await snapshot_of(records, settings)

    assert result.llm_calls == 4


async def test_llm_failures_sum_classification_failures_and_degraded_drafts(
    settings: Settings,
) -> None:
    records = [
        event("llm_classification_failed", fallback="rules"),
        event("llm_classification_failed", fallback="rules"),
        drafted(status="awaiting_operator", degraded=True),
        drafted(status="draft_ready", degraded=False),
    ]

    result = await snapshot_of(records, settings)

    assert result.llm_failures == 3
    assert result.drafts_degraded == 1


# --- drafts and backpressure ---------------------------------------------


async def test_drafts_ready_counts_only_drafts_that_passed_every_gate(
    settings: Settings,
) -> None:
    records = [
        drafted(status="draft_ready"),
        drafted(status="awaiting_operator"),
        drafted(status="awaiting_operator", degraded=True),
    ]

    result = await snapshot_of(records, settings)

    assert result.drafts_ready == 1


async def test_queue_rejected_counts_the_backpressure_events(settings: Settings) -> None:
    records = [
        triaged(),
        event("queue_rejected", pending=100),
        event("queue_rejected", pending=100),
    ]

    result = await snapshot_of(records, settings)

    assert result.queue_rejected == 2


# --- robustness ----------------------------------------------------------


async def test_a_triaged_record_without_details_falls_back_to_neutral_values(
    settings: Settings,
) -> None:
    """The projection reads a shape it does not enforce — it must not explode."""
    result = await snapshot_of([AuditRecord(ticket_id=uuid4(), event="triaged")], settings)

    assert result.tickets_total == 1
    assert (result.by_category, result.by_risk, result.by_route) == (
        {"": 1},
        {"": 1},
        {"": 1},
    )
    assert result.triage_latency_ms.count == 1
    assert result.triage_latency_ms.max_ms == 0.0
    assert result.over_budget == 0
    assert result.low_confidence == 1
