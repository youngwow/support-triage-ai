"""
Domain entities — the "Model" of Model-Service-Repository.

These are the objects services reason about and repositories persist. They are
deliberately independent of both HTTP (no FastAPI import) and storage (no ORM,
no driver), so the same entity can serve the REST API and any other frontend.

Every entity is frozen: state changes produce a new instance, which keeps the
repository the only component that decides when stored state moves.
"""

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.utils import utc_now


# The taxonomy is fixed by data/tickets/historical_tickets.json plus a catch-all.
# Adding a label here means re-running the golden-set evaluation.
Category = Literal[
    "billing/faq",
    "billing/payment_issue",
    "billing/refund",
    "account_recovery",
    "technical_issue",
    "other",
]
RiskLevel = Literal["low", "medium", "high"]
Channel = Literal["chat", "email", "web_form", "mobile"]

#: Where the ticket goes after triage. ``auto_answer`` still means "a draft may
#: be generated", never "an answer was sent" — nothing in this PoC replies to a
#: user without a human.
Route = Literal["auto_answer", "operator_queue", "senior_escalation"]

TicketStatus = Literal[
    "triaged",  # classified and routed, no draft requested
    "queued",  # waiting for the draft worker
    "draft_ready",  # a grounded draft is attached
    "awaiting_operator",  # a human must act; may or may not have a draft
]

_RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def max_risk(left: RiskLevel, right: RiskLevel) -> RiskLevel:
    """Risk only ever escalates.

    A rule that fired must never be softened by a model that disagrees, so
    every place that combines two opinions combines them through here.
    """
    return left if _RISK_ORDER[left] >= _RISK_ORDER[right] else right


class DocumentChunk(BaseModel):
    """One retrievable piece of the knowledge base."""

    model_config = ConfigDict(frozen=True)

    id: int
    source: str
    heading: str
    text: str


class RetrievedChunk(BaseModel):
    """A chunk plus the similarity score that surfaced it."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class TriageDecision(BaseModel):
    """The output of the synchronous path: what this ticket is and where it goes."""

    model_config = ConfigDict(frozen=True)

    category: Category
    category_confidence: float = Field(ge=0.0, le=1.0)
    risk: RiskLevel
    route: Route
    #: True only when *every* guard passed. See TriageService for the conjunction.
    auto_send_allowed: bool
    #: "llm" or "rules" — which classifier produced `category`. Distinguishing a
    #: degraded decision from a normal one is what makes the audit log useful.
    classifier: Literal["llm", "rules"]
    rule_hits: tuple[str, ...] = ()
    pii_types: tuple[str, ...] = ()
    injection_flagged: bool = False
    reason: str = ""
    latency_ms: float = 0.0


class DraftAnswer(BaseModel):
    """The output of the asynchronous path: a suggested reply, never a sent one."""

    model_config = ConfigDict(frozen=True)

    text: str
    sources: tuple[str, ...] = ()
    is_grounded: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: True when the draft was produced without the LLM (retrieval-only or a
    #: canned apology), i.e. the fallback path ran.
    degraded: bool = False


class Ticket(BaseModel):
    """A support request as the system knows it.

    ``text`` is always the PII-masked text. The raw body is never stored in this
    PoC — masking happens before anything else touches the ticket.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    channel: Channel
    text: str
    external_id: Optional[str] = None
    status: TicketStatus = "triaged"
    decision: Optional[TriageDecision] = None
    draft: Optional[DraftAnswer] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def with_decision(self, decision: TriageDecision, *, status: TicketStatus) -> "Ticket":
        return self.model_copy(
            update={"decision": decision, "status": status, "updated_at": utc_now()}
        )

    def with_draft(self, draft: DraftAnswer, *, status: TicketStatus) -> "Ticket":
        return self.model_copy(
            update={"draft": draft, "status": status, "updated_at": utc_now()}
        )


class AuditRecord(BaseModel):
    """One immutable line in the decision trail.

    Every automatic decision writes one. The metrics projection is computed by
    reading these back, so there is a single source of truth for "what happened"
    rather than a counter registry that can drift from the log.
    """

    model_config = ConfigDict(frozen=True)

    ticket_id: UUID
    event: str
    details: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utc_now)
