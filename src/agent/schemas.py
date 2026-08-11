"""
Structured-output contracts and the draft graph's state.

These are what the LLM is *forced* to return (`response_schema`), so they are
kept separate from both the domain entities and the wire models: an LLM contract
changing must not silently reshape the API.
"""

from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from src.models.domain import Category, RetrievedChunk, RiskLevel


class TicketClassification(BaseModel):
    """One structured call on the triage path: what is this ticket, how risky."""

    category: Category
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    #: Short justification, stored in the audit trail so a human reviewing an
    #: automatic decision can see why it was made.
    reason: str = ""


class GroundedDraft(BaseModel):
    """One structured call on the draft path: the answer plus its own verdict.

    ``is_grounded`` and ``confidence`` are requested in the *same* call as the
    answer rather than in a second verification call — one LLM round trip per
    ticket is the cost ceiling this PoC commits to.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)


class DraftState(TypedDict):
    """State threaded through the draft graph. Nodes return partial dicts."""

    ticket_id: str
    question: str
    category: Category
    risk: RiskLevel
    chunks: list[RetrievedChunk]
    generation: Optional[GroundedDraft]
    answer: Optional[str]
    sources: list[str]
    escalated: bool
    escalation_reason: Optional[str]


def initial_draft_state(
    *,
    ticket_id: str,
    question: str,
    category: Category,
    risk: RiskLevel,
) -> DraftState:
    return DraftState(
        ticket_id=ticket_id,
        question=question,
        category=category,
        risk=risk,
        chunks=[],
        generation=None,
        answer=None,
        sources=[],
        escalated=False,
        escalation_reason=None,
    )
