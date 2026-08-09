from typing import Literal, Optional, Self

from pydantic import BaseModel, Field

from src.models.domain import AssistantReply


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    checks: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    route: Literal["knowledge_base", "escalate", "degraded"]
    escalated: bool
    escalation_reason: Optional[str]
    sources: list[str]

    @classmethod
    def from_domain(cls, reply: AssistantReply) -> Self:
        """Single place where a domain entity becomes wire format."""
        return cls(
            reply=reply.text,
            route=reply.route,
            escalated=reply.escalated,
            escalation_reason=reply.escalation_reason,
            sources=reply.sources,
        )


class WebhookAck(BaseModel):
    """Telegram only needs a fast 200 — processing happens in the background."""

    ok: bool = True


class ErrorResponse(BaseModel):
    code: str
    detail: str
