from datetime import datetime
from typing import Any, Literal, Optional, Self
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.domain import AuditRecord, Ticket


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)


class TriageResponse(BaseModel):
    category: str
    category_confidence: float
    risk: str
    route: str
    auto_send_allowed: bool
    classifier: str
    rule_hits: list[str]
    pii_types: list[str]
    injection_flagged: bool
    reason: str
    latency_ms: float


class DraftResponse(BaseModel):
    text: str
    sources: list[str]
    is_grounded: bool
    confidence: float
    degraded: bool


class TicketResponse(BaseModel):
    id: UUID
    channel: str
    #: Already masked — the raw body is never stored, so it cannot be returned.
    text: str
    external_id: Optional[str]
    status: str
    triage: Optional[TriageResponse]
    draft: Optional[DraftResponse]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, ticket: Ticket) -> Self:
        """Single place where a domain entity becomes wire format."""
        decision = ticket.decision
        draft = ticket.draft
        return cls(
            id=ticket.id,
            channel=ticket.channel,
            text=ticket.text,
            external_id=ticket.external_id,
            status=ticket.status,
            triage=(
                None
                if decision is None
                else TriageResponse(
                    category=decision.category,
                    category_confidence=decision.category_confidence,
                    risk=decision.risk,
                    route=decision.route,
                    auto_send_allowed=decision.auto_send_allowed,
                    classifier=decision.classifier,
                    rule_hits=list(decision.rule_hits),
                    pii_types=list(decision.pii_types),
                    injection_flagged=decision.injection_flagged,
                    reason=decision.reason,
                    latency_ms=decision.latency_ms,
                )
            ),
            draft=(
                None
                if draft is None
                else DraftResponse(
                    text=draft.text,
                    sources=list(draft.sources),
                    is_grounded=draft.is_grounded,
                    confidence=draft.confidence,
                    degraded=draft.degraded,
                )
            ),
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
        )


class AuditRecordResponse(BaseModel):
    ticket_id: UUID
    event: str
    details: dict[str, Any]
    at: datetime

    @classmethod
    def from_domain(cls, record: AuditRecord) -> Self:
        return cls(
            ticket_id=record.ticket_id,
            event=record.event,
            details=record.details,
            at=record.at,
        )


class AuditResponse(BaseModel):
    ticket_id: UUID
    records: list[AuditRecordResponse]


class LatencyStats(BaseModel):
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float


class MetricsResponse(BaseModel):
    """A JSON projection of the audit log — not a Prometheus exposition format.

    Everything here is recomputed from the decision trail on each request, so a
    number shown can always be traced back to the records that produced it.
    """

    tickets_total: int
    by_category: dict[str, int]
    by_risk: dict[str, int]
    by_route: dict[str, int]
    auto_send_allowed: int
    low_confidence: int
    pii_detected: int
    injection_flagged: int
    llm_classifications: int
    rule_fallbacks: int
    llm_calls: int
    llm_failures: int
    drafts_ready: int
    drafts_degraded: int
    queue_rejected: int
    triage_latency_ms: LatencyStats
    hot_path_budget_ms: int
    over_budget: int
    over_budget_ratio: float


class ErrorResponse(BaseModel):
    code: str
    detail: str
