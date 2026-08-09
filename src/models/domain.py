"""
Domain entities — the "Model" of Model-Service-Repository.

These are the objects services reason about and repositories persist. They are
deliberately independent of both HTTP (no FastAPI import) and storage (no ORM,
no driver), so the same entity can serve the REST API and the Telegram bot.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.utils import utc_now


class DocumentChunk(BaseModel):
    """A fragment of one knowledge-base document, the retrieval unit."""

    model_config = ConfigDict(frozen=True)

    id: int
    source: str
    heading: Optional[str] = None
    text: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """A chunk together with its similarity to the query."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class ChatMessage(BaseModel):
    """One turn of a dialog, kept for context and escalation dumps."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    text: str
    at: datetime = Field(default_factory=utc_now)

    def as_line(self) -> str:
        return f"{self.role}: {self.text}"


class Employee(BaseModel):
    """What the external HR system knows about a user."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    full_name: str
    grade: int = Field(ge=1)
    vacation_balance_days: int = Field(ge=0)


class AssistantReply(BaseModel):
    """The agent's final verdict for one incoming message."""

    model_config = ConfigDict(frozen=True)

    text: str
    route: Literal["knowledge_base", "escalate", "degraded"]
    escalated: bool = False
    escalation_reason: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
